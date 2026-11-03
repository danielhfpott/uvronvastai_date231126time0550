#!/usr/bin/env python3
"""
UVR on Vast.ai - make a dry, studio-sounding vocal from a lead vocal stem.

Pick an audio file and press "Make dry": the app rents the cheapest suitable verified GPU on
vast.ai, runs Reverb_HQ_By_FoxJoy.onnx and UVR-DeEcho-DeReverb.pth on it separately with
python-audio-separator, downloads each model's dry stem as 24-bit FLAC and destroys the machine.

Run:  python uvr_vast_gui.py      (installs `requests` + `paramiko` on first run)
Settings: ~/.uvr_vast_gui.json  (saved API key + SSH key path; the output folder is remembered there)
"""

import io
import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText


def _ensure_deps():
    missing = []
    for mod in ("requests", "paramiko"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print("Installing", *missing, "...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


_ensure_deps()
import paramiko  # noqa: E402
import requests  # noqa: E402

APP = "UVR on Vast.ai"
CONFIG_PATH = Path.home() / ".uvr_vast_gui.json"
API = "https://console.vast.ai/api/v0"
API_V1 = "https://console.vast.ai/api/v1"
IMAGE = "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime"
UVR_LABEL = "uvr"  # instances with this label are offered for destruction when the window closes
RUNTYPE = "ssh_direc ssh_proxy"
READY_MARKER = "/root/uvr_ready"
SETUP_LOG = "/root/uvr_setup.log"
PATH_PREFIX = "export PATH=/opt/conda/bin:/usr/local/bin:$PATH PYTHONUNBUFFERED=1; "
REMOTE_DIR = "/root/uvr"
DISK_GB = 40.0
READY_TIMEOUT = 30 * 60  # seconds for boot + setup
MODELS = ["Reverb_HQ_By_FoxJoy.onnx", "UVR-DeEcho-DeReverb.pth"]
SAMPLE_RATE = "48000"  # the input is decoded to this and the models run at it; the FLACs keep it
q = shlex.quote

FAILED_MARKER = "/root/uvr_failed"
# Installs everything audio-separator needs. build-essential: diffq (a dependency) compiles C code
# and the pytorch runtime image ships without gcc. onnxruntime-gpu is pinned because newer builds
# want CUDA 13 libraries while the image has CUDA 12.8, which silently drops MDX-Net (.onnx) to CPU
# (1.25.1 verified on GPU). Also re-run over SSH by the app if boot setup failed.
SETUP_SH = r"""#!/bin/bash
export PATH=/opt/conda/bin:/usr/local/bin:$PATH DEBIAN_FRONTEND=noninteractive
rm -f /root/uvr_ready /root/uvr_failed
echo "== setup started $(date)"
fail() { echo "== SETUP FAILED: $1"; touch /root/uvr_failed; exit 1; }
if ! command -v ffmpeg >/dev/null || ! command -v gcc >/dev/null; then
  apt-get update -y && apt-get install -y --no-install-recommends ffmpeg build-essential || fail "apt-get install ffmpeg build-essential"
fi
python3 -m pip install --progress-bar off --root-user-action=ignore "audio-separator[gpu]" "onnxruntime-gpu==1.25.1" || fail "pip install audio-separator[gpu]"
B=$(command -v audio-separator) || fail "audio-separator is not on PATH after install"
[ "$B" != /usr/local/bin/audio-separator ] && ln -sf "$B" /usr/local/bin/audio-separator
mkdir -p /root/uvr/models
nvidia-smi -L
audio-separator --version || fail "audio-separator does not start"
python3 - <<'PY' || echo "== WARNING: ONNX GPU check failed - MDX-Net (.onnx) models may run on CPU"
import torch, onnxruntime as ort
from onnx import helper, TensorProto
v = lambda n: helper.make_tensor_value_info(n, TensorProto.FLOAT, [4])
m = helper.make_model(helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "g", [v("x")], [v("y")]),
                      opset_imports=[helper.make_opsetid("", 13)])
m.ir_version = 8
s = ort.InferenceSession(m.SerializeToString(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print("== GPU:", torch.cuda.get_device_name(0), "| ONNX Runtime", ort.__version__, "uses", s.get_providers()[0])
PY
touch /root/uvr_ready
echo "== setup done $(date)"
"""
ONSTART = ("cat > /root/uvr_setup.sh <<'UVR_SETUP_EOF'\n" + SETUP_SH + "UVR_SETUP_EOF\n"
           "bash /root/uvr_setup.sh >> /root/uvr_setup.log 2>&1\n")

# The machine the job needs (the full GUI's filter values): cheapest verified on-demand offer first.
OFFER_QUERY = {
    "rentable": {"eq": True}, "rented": {"eq": False}, "external": {"eq": False}, "verified": {"eq": True},
    "gpu_name": {"in": ["RTX 4090", "RTX 3090"]}, "num_gpus": {"eq": 1}, "gpu_ram": {"gte": 12000.0},
    "dph_total": {"lte": 0.6}, "reliability2": {"gte": 0.95}, "disk_space": {"gte": DISK_GB},
    "inet_down": {"gte": 200.0}, "cuda_max_good": {"gte": 12.8}, "cpu_ram": {"gte": 16000.0},
    "order": [["dph_total", "asc"]], "type": "on-demand", "limit": 64, "allocated_storage": DISK_GB,
}
# audio-separator settings the job has been running with: general, MDX-Net (.onnx), VR Arch (.pth).
# Stems stay WAV on the box; the dry one is encoded to FLAC by ffmpeg.
SEPARATOR_ARGS = ["--output_format", "WAV", "--log_level", "info",
                  "--normalization", "0.9", "--amplification", "0.0", "--sample_rate", SAMPLE_RATE,
                  "--mdx_segment_size", "256", "--mdx_overlap", "0.25", "--mdx_batch_size", "1",
                  "--mdx_hop_length", "1024",
                  "--vr_batch_size", "1", "--vr_window_size", "512", "--vr_aggression", "5",
                  "--vr_post_process_threshold", "0.2"]
FLAC_24 = ["-c:a", "flac", "-compression_level", "5", "-sample_fmt", "s32", "-bits_per_raw_sample", "24"]
AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus", ".aiff", ".aif",
             ".wma", ".mp4", ".mkv", ".webm", ".alac"}
OUT_RE = re.compile(r"^f(\d+)_\((.+?)\)_.+\.[A-Za-z0-9]+$")
LOOSE_RE = re.compile(r"^f(\d+)\W?(.*)\.[A-Za-z0-9]+$")


class StopJob(Exception):
    pass


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def safe_name(s):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", s).strip(" .")[:180] or "output"


def _offer_gone(err):
    return any(s in str(err) for s in ("no_such_ask", "not available", "already rented"))


def offer_desc(o):
    return (f"#{o['id']}: {o.get('num_gpus')}x {o.get('gpu_name')} ({(o.get('gpu_ram') or 0) / 1024:.0f} GB) "
            f"@ ${float(o.get('dph_total') or 0):.3f}/hr - {o.get('geolocation') or '?'}")


def load_or_create_key(path, log=print):
    """Return the OpenSSH public key line for `path`, generating an RSA key pair if missing."""
    p = Path(path).expanduser()
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        k = paramiko.RSAKey.generate(3072)
        k.write_private_key_file(str(p))
        pub = f"ssh-rsa {k.get_base64()} uvr-vast-gui"
        Path(str(p) + ".pub").write_text(pub + "\n")
        log(f"Generated new SSH key: {p}")
        return pub
    pub_path = Path(str(p) + ".pub")
    if pub_path.exists():
        return pub_path.read_text().strip()
    for cls in (paramiko.Ed25519Key, paramiko.ECDSAKey, paramiko.RSAKey):
        try:
            k = cls.from_private_key_file(str(p))
            return f"{k.get_name()} {k.get_base64()} uvr-vast-gui"
        except Exception:
            pass
    raise RuntimeError(f"Could not load SSH key {p} (passphrase-protected keys are not supported).")


def ssh_targets(inst, mode="auto"):
    direct = proxy = None
    try:
        port = int(inst["ports"]["22/tcp"][0]["HostPort"])
        ip = (inst.get("public_ipaddr") or "").strip()
        if ip:
            direct = (ip, port)
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    if inst.get("ssh_host") and inst.get("ssh_port"):
        port = int(inst["ssh_port"]) + (1 if "jupyter" in str(inst.get("image_runtype") or "") else 0)
        proxy = (inst["ssh_host"], port)
    picks = {"direct": [direct], "proxy": [proxy]}.get(mode, [direct, proxy])
    return [t for t in picks if t]


def uvr_args(model, out_dir):
    """audio-separator CLI arguments (everything except the input file)."""
    return ["-m", model, "--output_dir", out_dir, "--model_file_dir", f"{REMOTE_DIR}/models", *SEPARATOR_ARGS]


class Vast:
    def __init__(self, key):
        if not key.strip():
            raise RuntimeError(f"No saved Vast.ai API key - it is read from {CONFIG_PATH}.")
        self.key = key.strip()

    def req(self, method, path, **kw):
        headers = {"Authorization": f"Bearer {self.key}", "Accept": "application/json"}
        for base in (API, API_V1):  # Vast retires v0 endpoints with HTTP 410 -> retry on v1
            r = requests.request(method, base + path, timeout=60, headers=headers, **kw)
            if not (r.status_code == 410 and base == API):
                break
        if r.status_code >= 400:
            raise RuntimeError(f"Vast API {method} {path} -> HTTP {r.status_code}: {r.text[:400]}")
        try:
            return r.json()
        except ValueError:
            return {}

    def search(self, query):
        try:
            d = self.req("POST", "/bundles/", json=query)
        except RuntimeError:
            d = self.req("GET", "/bundles/", params={"q": json.dumps(query)})
        return d.get("offers", [])

    def create(self, offer_id, body):
        return self.req("PUT", f"/asks/{offer_id}/", json=body)

    def instances(self):
        d = self.req("GET", "/instances/", params={"owner": "me"})
        if isinstance(d, list):
            return d
        return d.get("instances") or d.get("results") or []

    def instance(self, iid):
        try:
            d = self.req("GET", f"/instances/{iid}/", params={"owner": "me"})
            inst = d.get("instances", d) if isinstance(d, dict) else d
            if isinstance(inst, list):
                inst = inst[0] if inst else None
            if inst:
                return inst
        except RuntimeError:
            pass
        return next((i for i in self.instances() if int(i.get("id", -1)) == int(iid)), None)

    def destroy(self, iid):
        return self.req("DELETE", f"/instances/{iid}/")


class Remote:
    def __init__(self, host, port, key_path):
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(host, port=port, username="root", key_filename=key_path, timeout=20,
                            banner_timeout=30, auth_timeout=30, look_for_keys=False, allow_agent=False)
        self.client.get_transport().set_keepalive(30)

    def run(self, cmd, timeout=120):
        chan = self.client.get_transport().open_session(timeout=30)
        chan.set_combine_stderr(True)
        chan.settimeout(timeout)
        chan.exec_command(cmd)
        data = b""
        while True:
            d = chan.recv(65536)
            if not d:
                break
            data += d
        return chan.recv_exit_status(), data.decode("utf-8", "replace")

    def stream(self, cmd, on_line, stop):
        chan = self.client.get_transport().open_session(timeout=30)
        chan.set_combine_stderr(True)
        chan.exec_command(cmd)
        buf = b""
        while True:
            if stop.is_set():
                chan.close()
                try:
                    self.run("pkill -f audio-separator; pkill ffmpeg", timeout=15)
                except Exception:
                    pass
                return -1
            if chan.recv_ready():
                buf += chan.recv(65536)
                *lines, buf = re.split(rb"[\r\n]", buf)
                for ln in lines:
                    s = ln.decode("utf-8", "replace").rstrip()
                    if s:
                        on_line(s)
            elif chan.exit_status_ready():
                break
            else:
                time.sleep(0.1)
        if buf.strip():
            on_line(buf.decode("utf-8", "replace").rstrip())
        return chan.recv_exit_status()

    def close(self):
        self.client.close()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP)
        icon = Path(__file__).with_name("uvr_vast.ico")
        if icon.exists():
            try:
                self.iconbitmap(default=str(icon))
            except tk.TclError:
                pass
        self.geometry("900x620")
        self.minsize(640, 420)
        self.saved = load_json(CONFIG_PATH)
        self.ssh_key = str(Path(self.saved.get("ssh_key_path") or Path.home() / ".ssh" / "uvr_vast_rsa").expanduser())
        self.q = queue.Queue()
        self.inst_cache = {}
        self.stop_event = threading.Event()
        self.job = None
        self.job_iid = None  # the instance the running job rented, until it is confirmed destroyed
        self._build()
        self.after(100, self._pump)
        self.after(800, self.on_find_instances)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- small helpers ----------
    def log(self, msg):
        self.q.put(("log", f"[{time.strftime('%H:%M:%S')}] {msg}"))

    def status(self, msg, error=False):
        self.q.put(("status", (msg, error)))

    def ui(self, fn, *a):
        self.q.put(("call", lambda: fn(*a)))

    def _pump(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "log":
                    self.logbox.insert("end", val + "\n")
                    self.logbox.see("end")
                elif kind == "status":
                    msg, error = val
                    self.status_lbl.config(text=msg, foreground="red" if error else "")
                elif kind == "call":
                    try:
                        val()
                    except Exception as e:
                        self.logbox.insert("end", f"UI error: {e}\n")
        except queue.Empty:
            pass
        self.after(100, self._pump)

    def _save(self):
        """Remember the output folder; everything else in the settings file is kept as it is."""
        self.saved["local_out"] = self.out_var.get().strip()
        try:
            CONFIG_PATH.write_text(json.dumps(self.saved, indent=2), encoding="utf-8")
        except OSError as e:
            messagebox.showerror(APP, f"Could not save settings: {e}")

    def bg(self, fn, *a, job=False):
        if job:
            if self.job and self.job.is_alive():
                messagebox.showwarning(APP, "A job is already running - press Stop first.")
                return
            self.stop_event.clear()

        def run():
            try:
                fn(*a)
            except RuntimeError as e:
                self.log(f"ERROR: {e}")
            except Exception as e:
                self.log(f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")

        t = threading.Thread(target=run, daemon=True)
        if job:
            self.job = t
        t.start()

    def check_stop(self):
        if self.stop_event.is_set():
            raise StopJob()

    def nap(self, secs):
        if self.stop_event.wait(secs):
            raise StopJob()

    # ---------- layout ----------
    def _build(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=(8, 4))
        top.columnconfigure(1, weight=1)
        self.in_var = tk.StringVar()
        self.out_var = tk.StringVar(value=self.saved.get("local_out") or str(Path.home() / "Music" / "UVR_output"))
        for r, (label, v, cmd) in enumerate((("Input audio file", self.in_var, self.on_browse_in),
                                             ("Output folder", self.out_var, self.on_browse_out))):
            ttk.Label(top, text=label).grid(row=r, column=0, sticky="w", padx=(0, 6), pady=2)
            ttk.Entry(top, textvariable=v).grid(row=r, column=1, sticky="we", pady=2)
            ttk.Button(top, text="Browse…", command=cmd).grid(row=r, column=2, padx=(6, 0), pady=2)

        act = ttk.Frame(self)
        act.pack(fill="x", padx=8, pady=4)
        ttk.Button(act, text="Make dry", command=self.on_make_dry).pack(side="left")
        ttk.Button(act, text="Stop", command=self.on_stop).pack(side="left", padx=6)
        self.status_lbl = ttk.Label(self, text="Ready.", anchor="w")
        self.status_lbl.pack(fill="x", padx=8)
        self.logbox = ScrolledText(self, height=12, font=("Consolas", 9))
        self.logbox.pack(fill="both", expand=True, padx=8, pady=(4, 8))

    def on_browse_in(self):
        exts = " ".join("*" + e for e in sorted(AUDIO_EXT))
        p = filedialog.askopenfilename(title="Lead vocal stem", filetypes=[("Audio", exts), ("All", "*.*")])
        if p:
            self.in_var.set(p)

    def on_browse_out(self):
        p = filedialog.askdirectory(title="Output folder")
        if p:
            self.out_var.set(p)
            self._save()

    # ---------- instances ----------
    def on_find_instances(self):
        key = self.saved.get("api_key", "")
        if key.strip():
            self.bg(lambda: self.ui(self._got_instances, Vast(key).instances()))

    def _uvr_ids(self):
        return {int(k) for k, i in self.inst_cache.items() if i.get("label") == UVR_LABEL}

    def _got_instances(self, insts):
        self.inst_cache = {str(i.get("id")): i for i in insts}
        mine = [i for i in insts if i.get("label") == UVR_LABEL]
        others = [i for i in insts if i.get("label") != UVR_LABEL]
        for i in mine:
            self.log(f"UVR instance #{i.get('id')}: {i.get('actual_status') or '?'} · {i.get('num_gpus', 1)}x "
                     f"{i.get('gpu_name', '?')} · ${float(i.get('dph_total') or 0):.3f}/hr - billing until destroyed")
        if others:
            self.log("Also billing on your account, not managed here: " + ", ".join(
                f"#{i.get('id')} '{i.get('label') or '-'}' ${float(i.get('dph_total') or 0):.3f}/hr" for i in others))

    def _destroy(self, vast, iid):
        """Destroy `iid` and wait until it is gone from the account. Returns False if that can't be confirmed."""
        self.status(f"Destroying instance #{iid}...")
        deleted, err, deadline = False, None, time.time() + 120
        while time.time() < deadline:
            try:
                if not deleted:
                    vast.destroy(iid)
                    deleted = True
                    self.status(f"Destroying instance #{iid} - checking that it is gone...")
                if all(int(i.get("id", -1)) != iid for i in vast.instances()):
                    self.log(f"Destroyed instance #{iid} - confirmed gone from your account. Billing stopped.")
                    return True
                err = "it is still listed on your account"
            except Exception as e:
                err = e
            time.sleep(4)
        self.log(f"!! Could not confirm that instance #{iid} was destroyed ({err}). "
                 "IT MAY STILL BE BILLING - destroy it at cloud.vast.ai/instances!")
        return False

    # ---------- the job ----------
    def on_make_dry(self):
        src, out = self.in_var.get().strip(), self.out_var.get().strip()
        if not os.path.isfile(src):
            messagebox.showerror(APP, f"Input file not found:\n{src}" if src else "Choose an input audio file first.")
            return
        if not out:
            messagebox.showerror(APP, "Choose an output folder first.")
            return
        self._save()
        self.bg(self._make_dry, src, Path(out).expanduser(), job=True)

    def on_stop(self):
        if self.job and self.job.is_alive():
            self.stop_event.set()
            self.status("Stopping - the instance will be destroyed...")

    def _make_dry(self, src, out_local):
        self.job_iid, vast, t0, dph = None, None, time.time(), 0.0
        result, failed = "Stopped", False
        try:
            vast = Vast(self.saved.get("api_key", ""))
            pub = load_or_create_key(self.ssh_key, self.log)
            out_local.mkdir(parents=True, exist_ok=True)
            self._ensure_account_key(vast, pub)
            iid = self._rent_any(vast)
            r, dph = self._connect_ready(vast, iid, pub)
            try:
                made, problems = self._run_job(r, src, out_local)
            finally:
                r.close()
            result = f"Done - {made} dry file(s) in {out_local}" + (" (with problems, see log)" if problems else "")
            self.ui(self.bell)
        except StopJob:
            self.log("Stopped.")
        except RuntimeError as e:
            result, failed = f"FAILED: {e}", True
            self.log(f"ERROR: {e}")
        except Exception as e:
            result, failed = f"FAILED: {type(e).__name__}: {e}", True
            self.log(f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        finally:
            mins = (time.time() - t0) / 60
            if dph:
                self.log(f"Job time {mins:.1f} min ≈ ${dph * mins / 60:.3f} at ${dph:.3f}/hr")
            iid = self.job_iid
            if not iid:
                self.status(f"{result} - no instance was rented.", failed)
            elif self._destroy(vast, iid):
                self.job_iid = None
                self.status(f"{result} - instance #{iid} destroyed, confirmed gone.", failed)
            else:
                self.status(f"{result} - !! INSTANCE #{iid} MAY STILL BE BILLING - destroy it at "
                            "cloud.vast.ai/instances", True)
                self.ui(messagebox.showerror, APP, f"Could not confirm that instance #{iid} was destroyed.\n\n"
                        "IT MAY STILL BE BILLING.\n\nDestroy it now at cloud.vast.ai/instances.")
            self.ui(self.on_find_instances)

    def _ensure_account_key(self, vast, pub):
        body = pub.split()[1]
        try:
            if body in json.dumps(vast.req("GET", "/ssh/")):
                return
        except Exception as e:
            self.log(f"(could not list account SSH keys: {e})")
        try:
            vast.req("POST", "/ssh/", json={"ssh_key": pub})
            self.log("Added SSH public key to your Vast account.")
        except Exception as e:
            self.log(f"(could not add SSH key to account: {e})")

    def _rent_any(self, vast):
        """Rent the cheapest matching offer; if it was taken meanwhile, move on to the next one."""
        tried, pool = set(), []
        for _ in range(6):
            if not pool:
                self.status("Looking for the cheapest suitable verified GPU...")
                pool = sorted((o for o in vast.search(OFFER_QUERY) if o["id"] not in tried),
                              key=lambda o: float(o.get("dph_total") or "inf"))
                if not pool:
                    raise RuntimeError("No verified GPU offer matches the job's requirements right now "
                                       "(1x RTX 4090 / RTX 3090, up to $0.60/hr) - try again later.")
            o = pool.pop(0)
            tried.add(o["id"])
            self.check_stop()
            try:
                return self._rent(vast, o)
            except RuntimeError as e:
                if not _offer_gone(e):
                    raise
                self.log(f"Offer #{o['id']} is no longer available - trying the next cheapest match...")
        raise RuntimeError("Could not rent any of the matching offers - try again.")

    def _rent(self, vast, offer):
        body = {"client_id": "me", "image": IMAGE, "disk": DISK_GB, "label": UVR_LABEL,
                "runtype": RUNTYPE, "onstart": ONSTART, "env": {}}
        self.log("Renting " + offer_desc(offer) + " ...")
        res = vast.create(offer["id"], body)
        iid = res.get("new_contract")
        if iid:
            self.job_iid = int(iid)
        if not iid or res.get("success") is False:
            raise RuntimeError(f"Rent failed: {res}")
        self.log(f"Rented instance #{iid}. Billing has started.")
        return int(iid)

    # ---------- SSH / readiness ----------
    def _wait_running(self, vast, iid, deadline):
        last = None
        while True:
            inst = vast.instance(iid)
            if not inst:
                raise RuntimeError(f"Instance #{iid} not found (destroyed?).")
            st = inst.get("actual_status") or "pending"
            msg = (inst.get("status_msg") or "").strip()
            if (st, msg) != last:
                self.log(f"Instance #{iid}: {st}" + (f" - {msg[:200]}" if msg else ""))
                last = (st, msg)
            if st == "running" and ssh_targets(inst):
                return inst
            if inst.get("intended_status") == "stopped":
                raise RuntimeError(f"Instance #{iid} is stopped - start it on cloud.vast.ai or rent a new one.")
            if time.time() > deadline:
                raise RuntimeError(f"Instance #{iid} not running before the timeout.")
            self.status(f"Waiting for instance #{iid} to boot ({st})...")
            self.nap(8)

    def _ssh(self, inst, deadline):
        targets = ssh_targets(inst)
        if not targets:
            raise RuntimeError("Instance has no SSH endpoint yet.")
        last = None
        while True:
            for host, port in targets:
                try:
                    self.status(f"Connecting SSH {host}:{port} ...")
                    r = Remote(host, port, self.ssh_key)
                    self.log(f"SSH connected: root@{host} -p {port}")
                    return r
                except Exception as e:
                    last = e
            if time.time() > deadline:
                raise RuntimeError(f"SSH failed: {type(last).__name__}: {last}")
            self.status(f"SSH not up yet ({type(last).__name__}); retrying...")
            self.nap(10)

    def _connect_ready(self, vast, iid, pub):
        deadline = time.time() + READY_TIMEOUT
        inst = self._wait_running(vast, iid, deadline)
        try:
            vast.req("POST", f"/instances/{iid}/ssh/", json={"ssh_key": pub})
        except Exception as e:
            self.log(f"(attach SSH key to instance: {e})")
        r = self._ssh(inst, deadline)
        # [o]/[u] keep pgrep from matching the shell that runs this very check
        probe = (f"if [ -f {READY_MARKER} ]; then echo __READY__; elif [ -f {FAILED_MARKER} ]; then echo __FAILED__; "
                 f"elif pgrep -f '[o]nstart.sh|[u]vr_setup.sh' >/dev/null 2>&1; then echo __RUNNING__; "
                 f"elif [ -f {SETUP_LOG} ]; then echo __FAILED__; else echo __NOLOG__; fi; "
                 f"tail -n 2 {SETUP_LOG} 2>/dev/null")
        try:
            last, repaired, since = "", False, time.time()
            while True:
                _, out = r.run(probe, timeout=30)
                state, _, tail = out.strip().partition("\n")
                if state == "__READY__":
                    self.log("Remote setup ready (ffmpeg + audio-separator installed).")
                    break
                if state == "__FAILED__" or (state == "__NOLOG__" and time.time() - since > 120):
                    if repaired:
                        raise RuntimeError("Remote setup failed again - see the setup errors above.")
                    self._repair_setup(r)
                    repaired = True
                    continue
                if tail.strip() and tail != last:
                    self.log("setup: " + tail.strip().replace("\n", " | ")[-300:])
                    last = tail
                if time.time() > deadline:
                    raise RuntimeError("Remote setup did not finish before the timeout - see the setup lines above.")
                self.status("Waiting for remote setup (installing ffmpeg, compiler + audio-separator)...")
                self.nap(8)
        except BaseException:
            r.close()
            raise
        return r, float(inst.get("dph_total") or 0)

    def _repair_setup(self, r):
        """Boot-time setup failed or never ran: run it over SSH."""
        self.log("Remote setup failed or never ran - installing over SSH now...")
        sftp = r.client.open_sftp()
        try:
            sftp.putfo(io.BytesIO(SETUP_SH.encode()), "/root/uvr_setup.sh")
        finally:
            sftp.close()

        def line(s):
            if s.startswith("==") or re.search(r"\berror\b|failed|Successfully installed", s, re.I):
                self.log("  setup: " + s[-300:])
            else:
                self.status("setup: " + s[-140:])
        r.stream(f"bash /root/uvr_setup.sh 2>&1 | tee -a {SETUP_LOG}", line, self.stop_event)
        self.check_stop()

    # ---------- separation ----------
    def _progress(self, label):
        last = [0.0]

        def cb(done, total):
            now = time.time()
            if now - last[0] > 0.3 or done == total:
                last[0] = now
                self.status(f"{label}: {done / 1e6:.1f}/{total / 1e6:.1f} MB")
            if self.stop_event.is_set():
                raise StopJob()
        return cb

    def _remote_line(self, line):
        if "%|" in line:  # tqdm progress bar -> status line instead of log
            self.status(line.strip()[-140:])
        else:
            self.log("  " + line)

    def _separate(self, r, sftp, model, in_path, out_dir):
        """Run audio-separator on one remote WAV named f<n>.wav; returns {stem: remote path} for every stem."""
        args = uvr_args(model, out_dir) + [in_path]
        self.log("$ audio-separator " + " ".join(q(x) for x in args))
        code = r.stream(PATH_PREFIX + f"mkdir -p {q(out_dir)} && audio-separator "
                        + " ".join(q(x) for x in args), self._remote_line, self.stop_event)
        self.check_stop()
        try:
            names = sorted(sftp.listdir(out_dir))
        except IOError:
            names = []
        stems = {}
        for name in names:
            m = OUT_RE.match(name)
            if m:
                stem = m.group(2)
            else:
                m = LOOSE_RE.match(name)
                if not m:
                    continue
                paren = re.search(r"\(([^()]+)\)", m.group(2))
                stem = paren.group(1) if paren else (m.group(2) or "stem")
            stems[stem] = f"{out_dir}/{name}"
        if code != 0 or not stems:
            self.log(f"!! {model}: exit code {code}, {len(stems)} stem file(s) produced")
        return stems

    def _run_job(self, r, src, out_local):
        """Upload `src`, run each model on it separately and download each model's dry ("No …") stem
        as 24-bit FLAC. Returns (files made, problems)."""
        base = f"{REMOTE_DIR}/job_{time.strftime('%Y%m%d_%H%M%S')}"
        r.run(f"mkdir -p {q(base)}/final {q(REMOTE_DIR)}/models")
        sftp = r.client.open_sftp()
        song, made, problems = Path(src).stem, 0, 0
        try:
            # 1) upload, then decode to stereo float WAV
            up, wav = f"{base}/src{Path(src).suffix.lower()}", f"{base}/f1.wav"
            self.log(f"Upload: {Path(src).name} ({os.path.getsize(src) / 1e6:.1f} MB)")
            sftp.put(src, up, callback=self._progress(f"Uploading {Path(src).name}"))
            self.log(f"Decoding → {SAMPLE_RATE} Hz stereo")
            ff = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", up,
                  "-ar", SAMPLE_RATE, "-ac", "2", "-c:a", "pcm_f32le", wav]
            r.stream(" ".join(q(x) for x in ff), self._remote_line, self.stop_event)
            self.check_stop()
            try:
                size = sftp.stat(wav).st_size or 0
            except IOError:
                size = 0
            if size < 4096:  # missing, or header only
                raise RuntimeError(f"Could not decode {Path(src).name} - no audio came out.")

            # 2) each model on the input separately; encode + download only its dry stem
            for i, model in enumerate(MODELS, 1):
                self.check_stop()
                self.log(f"Model {i}/{len(MODELS)}: {model}")
                stems = self._separate(r, sftp, model, wav, f"{base}/m{i}")
                dry = next((s for s in sorted(stems) if s.lower().startswith("no ")), None)
                if not dry:
                    self.log(f"!! {model}: no dry ('No …') stem among {sorted(stems)}")
                    problems += 1
                    continue
                dst = out_local / f"{safe_name(f'{song}_({dry})_{Path(model).stem}')}.flac"
                remote = f"{base}/final/{i}.flac"
                ff = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", stems[dry],
                      *FLAC_24, remote]
                self.log(f"Encoding {dry} → 24-bit FLAC ...")
                r.stream(" ".join(q(x) for x in ff), self._remote_line, self.stop_event)
                self.check_stop()
                try:
                    sftp.stat(remote)
                    sftp.get(remote, str(dst), callback=self._progress(f"Downloading {dst.name}"))
                    self.log(f"  ↓ {dst}")
                    made += 1
                except IOError as e:
                    problems += 1
                    self.log(f"!! could not download {dst.name}: {e}")
        finally:
            sftp.close()
        if not made:
            raise RuntimeError("No dry file was made - see the log.")
        return made, problems

    # ---------- shutdown ----------
    def _on_close(self):
        self._save()
        alive = sorted(self._uvr_ids() | ({self.job_iid} if self.job_iid else set()))
        if alive:
            ans = messagebox.askyesnocancel(APP, "These UVR instances may still be running (and billing): "
                                            + ", ".join(f"#{i}" for i in alive) + "\n\nDestroy them now?")
            if ans is None:
                return
            if ans:
                try:
                    vast = Vast(self.saved.get("api_key", ""))
                    for i in alive:
                        vast.destroy(i)
                except Exception as e:
                    messagebox.showerror(APP, f"Destroy failed: {e}\nDestroy them at cloud.vast.ai/instances!")
                    return
        self.stop_event.set()
        self.destroy()


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
            # own taskbar button/icon instead of grouping under python
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("uvr.vast.gui")
        except Exception:
            pass
    App().mainloop()
