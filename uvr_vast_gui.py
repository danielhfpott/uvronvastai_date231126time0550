#!/usr/bin/env python3
"""
UVR on Vast.ai - split a finished mixdown back into stems, or make a dry vocal from a vocal stem.

Pick an audio file, pick what to make and press Start. The app uses its running vast.ai instance, or
rents the GPU you picked in the list - only vacant, verified RTX 4090 / RTX 5090 machines in Europe
with at least 99.5% reliability and fast internet both ways are listed.
  All stems: BS-Roformer-SW splits the mix into vocals, drums, bass, guitar, piano and other; a karaoke
    model splits the vocals into lead + backing, and DrumSep splits the drums into kick, snare, toms,
    hi-hat, ride and crash. Every stem is downloaded as 24-bit FLAC into one folder per song.
  Dry vocal: every de-reverb version in RECIPES, each as 24-bit FLAC.
The instance keeps running for the next file until you press "Destroy instance".

Run:  python uvr_vast_gui.py      (installs `requests` + `paramiko` on first run)
Settings: ~/.uvr_vast_gui.json  (SSH key path; output folder and instance ID are remembered there)
The Vast.ai API key is typed into the window on every launch and is never saved.
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
UVR_LABEL = "uvr"  # label of the instances this app rents
RUNTYPE = "ssh_direc ssh_proxy"
READY_MARKER = "/root/uvr_ready"
SETUP_LOG = "/root/uvr_setup.log"
PATH_PREFIX = "export PATH=/opt/conda/bin:/usr/local/bin:$PATH PYTHONUNBUFFERED=1; "
REMOTE_DIR = "/root/uvr"
DISK_GB = 40.0
READY_TIMEOUT = 30 * 60  # seconds for boot + setup
ANVUEW = "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"  # strongest de-reverb
SUCIAL = "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt"  # de-reverb + de-echo (delays)
# Each recipe is one dry FLAC. A recipe with several models runs each on the previous one's dry
# stem; recipes that start the same share those runs.
RECIPES = [("Reverb_HQ_By_FoxJoy.onnx",), ("UVR-DeEcho-DeReverb.pth",), (ANVUEW,), (SUCIAL,), (ANVUEW, SUCIAL)]
# The dry stem each model writes, as audio-separator names it (its models-scores.json; compared lower-case)
DRY_STEM = {"Reverb_HQ_By_FoxJoy.onnx": "no reverb", "UVR-DeEcho-DeReverb.pth": "no reverb",
            ANVUEW: "noreverb", SUCIAL: "dry"}
# The models are trained on 44.1 kHz, and VR Arch (.pth) models write 44.1 kHz audio labelled with
# this rate - anything else detunes them. Dry FLACs are then resampled to OUT_RATE; stems go back to
# the input file's own rate, so they line up with the mixdown in FL Studio.
SAMPLE_RATE = "44100"
OUT_RATE = "48000"
q = shlex.quote

# ---- "All stems": a cascade of the strongest open models audio-separator ships (0.48) ----
SW = "BS-Roformer-SW.ckpt"  # 6 stems: bass, drums, other, vocals, guitar, piano
KARAOKE = "bs_roformer_karaoke_frazer_becruily.ckpt"  # on a vocal stem: Vocals = lead, Instrumental = backing
DRUMSEP = "MDX23C-DrumSep-aufr33-jarredou.ckpt"  # on a drum stem: kick, snare, toms, hh, ride, crash
# Each step: (model, what it runs on) - None = the mixdown, else (earlier model, its stem, lower-case)
STEM_STEPS = [(SW, None), (KARAOKE, (SW, "vocals")), (DRUMSEP, (SW, "drums"))]
# The FLACs, in this order: (file name, model, its stem lower-case). The "(all)" stems are kept too,
# for when the finer split of a song isn't clean.
STEM_FILES = [
    ("Lead Vocal", KARAOKE, "vocals"), ("Backing Vocals", KARAOKE, "instrumental"), ("Vocals (all)", SW, "vocals"),
    ("Kick", DRUMSEP, "kick"), ("Snare", DRUMSEP, "snare"), ("Toms", DRUMSEP, "toms"), ("Hi-Hat", DRUMSEP, "hh"),
    ("Ride", DRUMSEP, "ride"), ("Crash", DRUMSEP, "crash"), ("Drums (all)", SW, "drums"),
    ("Bass", SW, "bass"), ("Guitar", SW, "guitar"), ("Piano", SW, "piano"), ("Other - synths, pads, FX", SW, "other"),
]
# Stems keep the mix's level (normalization 1.0 only scales down a stem that would clip), so they sum
# back to the mix. SW's own overlap is 2; 4 removes more chunk-boundary artefacts.
STEM_ARGS = ["--output_format", "WAV", "--log_level", "info", "--normalization", "1.0",
             "--amplification", "0.0", "--sample_rate", SAMPLE_RATE]
STEM_EXTRA = {SW: ["--mdxc_overlap", "4"]}
MODES = {"stems": "All stems from a mixdown (lead/backing vocals, kick, snare, hats, bass, guitar, piano, synths…)",
         "dry": "Dry vocal from a vocal stem (de-reverb / de-echo versions)"}

FAILED_MARKER = "/root/uvr_failed"
# Installs everything audio-separator needs. build-essential: diffq (a dependency) compiles C code
# and the pytorch runtime image ships without gcc. onnxruntime-gpu is pinned because newer builds
# want CUDA 13 libraries while the image has CUDA 12.8, which silently drops MDX-Net (.onnx) to CPU
# (1.25.1 verified on GPU). librosa 1.0 dropped audioread, which audio-separator 0.48 still imports, so
# librosa stays below 1. Also re-run over SSH by the app if boot setup failed.
SETUP_SH = r"""#!/bin/bash
export PATH=/opt/conda/bin:/usr/local/bin:$PATH DEBIAN_FRONTEND=noninteractive
rm -f /root/uvr_ready /root/uvr_failed
echo "== setup started $(date)"
fail() { echo "== SETUP FAILED: $1"; touch /root/uvr_failed; exit 1; }
if ! command -v ffmpeg >/dev/null || ! command -v gcc >/dev/null; then
  apt-get update -y && apt-get install -y --no-install-recommends ffmpeg build-essential || fail "apt-get install ffmpeg build-essential"
fi
python3 -m pip install --progress-bar off --root-user-action=ignore "audio-separator[gpu]" "onnxruntime-gpu==1.25.1" "librosa<1" || fail "pip install audio-separator[gpu]"
B=$(command -v audio-separator) || fail "audio-separator is not on PATH after install"
[ "$B" != /usr/local/bin/audio-separator ] && ln -sf "$B" /usr/local/bin/audio-separator
mkdir -p /root/uvr/models
nvidia-smi -L
audio-separator --version || fail "audio-separator does not start"
python3 -c "from audio_separator.separator import Separator" || fail "audio-separator cannot load its separator"
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

# The only machines the GPU list offers: vacant, verified, single RTX 4090 / 5090 in Europe, at least
# 99.5% reliable, with fast internet both ways (uploads + model downloads). Vast filters by these;
# offer_ok() checks them again so nothing else can ever be listed or rented.
GPU_NAMES = ("RTX 4090", "RTX 5090")
MIN_RELIABILITY = 0.995
MIN_NET_MBPS = 500.0  # both download and upload, Mbit/s
EUROPE = ("AL", "AT", "BA", "BE", "BG", "CH", "CY", "CZ", "DE", "DK", "EE", "ES", "FI", "FR", "GB", "GR", "HR",
          "HU", "IE", "IS", "IT", "LI", "LT", "LU", "LV", "MD", "ME", "MK", "MT", "NL", "NO", "PL", "PT", "RO",
          "RS", "SE", "SI", "SK", "UA")
OFFER_QUERY = {
    "rentable": {"eq": True}, "rented": {"eq": False}, "external": {"eq": False}, "verified": {"eq": True},
    "gpu_name": {"in": list(GPU_NAMES)}, "num_gpus": {"eq": 1}, "reliability2": {"gte": MIN_RELIABILITY},
    "geolocation": {"in": list(EUROPE)}, "inet_down": {"gte": MIN_NET_MBPS}, "inet_up": {"gte": MIN_NET_MBPS},
    "disk_space": {"gte": DISK_GB}, "cuda_max_good": {"gte": 12.8}, "cpu_ram": {"gte": 16000.0},
    "order": [["dph_total", "asc"]], "type": "on-demand", "limit": 200, "allocated_storage": DISK_GB,
}
# audio-separator settings: general, MDX-Net (.onnx), VR Arch (.pth); Roformer (.ckpt) models use their
# own config. Stems stay WAV on the box; the dry one is encoded to FLAC by ffmpeg.
SEPARATOR_ARGS = ["--output_format", "WAV", "--log_level", "info",
                  "--normalization", "0.9", "--amplification", "0.0", "--sample_rate", SAMPLE_RATE,
                  "--mdx_segment_size", "256", "--mdx_overlap", "0.25", "--mdx_batch_size", "1",
                  "--mdx_hop_length", "1024",
                  "--vr_batch_size", "1", "--vr_window_size", "512", "--vr_aggression", "5",
                  "--vr_post_process_threshold", "0.2"]
AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus", ".aiff", ".aif",
             ".wma", ".mp4", ".mkv", ".webm", ".alac"}
OUT_RE = re.compile(r"^f(\d+)_\((.+?)\)_.+\.[A-Za-z0-9]+$")
LOOSE_RE = re.compile(r"^f(\d+)\W?(.*)\.[A-Za-z0-9]+$")


def flac_24(rate):
    """ffmpeg output options: 24-bit FLAC at `rate` Hz."""
    return ["-af", f"aresample={rate}:resampler=soxr:precision=28",
            "-c:a", "flac", "-compression_level", "5", "-sample_fmt", "s32", "-bits_per_raw_sample", "24"]


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


def _num(o, k):
    try:
        return float(o.get(k) or 0)
    except (TypeError, ValueError):
        return 0.0


def offer_country(o):
    """Two-letter country code of an offer ('Norway, NO' -> 'NO'), '' if unknown."""
    return str(o.get("geolocation") or "").rpartition(",")[2].strip().upper()


def offer_ok(o):
    """True if the offer meets every GPU-list requirement (checked again here, not only by Vast's filter)."""
    return (o.get("gpu_name") in GPU_NAMES and int(o.get("num_gpus") or 0) == 1
            and offer_country(o) in EUROPE and _num(o, "reliability2") >= MIN_RELIABILITY
            and _num(o, "inet_down") >= MIN_NET_MBPS and _num(o, "inet_up") >= MIN_NET_MBPS
            and o.get("rentable", True) and not o.get("rented") and o.get("verification", "verified") == "verified")


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


def uvr_args(model, out_dir, settings=SEPARATOR_ARGS):
    """audio-separator CLI arguments (everything except the input file)."""
    return ["-m", model, "--output_dir", out_dir, "--model_file_dir", f"{REMOTE_DIR}/models", *settings]


class Vast:
    def __init__(self, key, anonymous=False):
        """`anonymous`: allowed without an API key (only the offer search works without one)."""
        if not key.strip() and not anonymous:
            raise RuntimeError("No Vast.ai API key - enter it at the top of the window and press Connect.")
        self.key = key.strip()

    def req(self, method, path, **kw):
        headers = {"Accept": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
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

    def gpu_offers(self):
        """The GPUs the list may offer (OFFER_QUERY + offer_ok), cheapest first."""
        return sorted((o for o in self.search(OFFER_QUERY) if offer_ok(o)), key=lambda o: _num(o, "dph_total"))

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
        self.geometry("1000x760")
        self.minsize(760, 560)
        self.saved = load_json(CONFIG_PATH)
        self.api_key = ""  # typed into the window on every launch; never written to the settings file
        old_key = self.saved.pop("api_key", None) is not None
        self.ssh_key = str(Path(self.saved.get("ssh_key_path") or Path.home() / ".ssh" / "uvr_vast_rsa").expanduser())
        self.q = queue.Queue()
        self.stop_event = threading.Event()
        self.job = None
        self.closing = self.destroying = False
        self.attached = set()
        # the instance jobs run on: {"id", "state", "dph", "ready"} - ready = set up, as far as the app knows
        saved_id = str(self.saved.get("instance_id") or "").strip()
        self.inst = ({"id": int(saved_id), "state": "not checked yet", "dph": None, "ready": True}
                     if saved_id.isdigit() else None)
        self.status_msg, self.status_err, self.inst_txt = "Ready.", False, self._inst_text()
        self.offers = []  # the GPU list, in display order
        self.offer_sort = ("dph_total", False)
        self.offers_busy = False
        self._build()
        if old_key:
            self._save()  # deletes the previously saved API key from the settings file
        self.after(100, self._pump)
        self.after(300, self.on_refresh_gpus)  # the offer search needs no API key
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- small helpers ----------
    def log(self, msg):
        self.q.put(("log", f"[{time.strftime('%H:%M:%S')}] {msg}"))

    def status(self, msg, error=False):
        self.q.put(("status", (msg, error)))

    def ui(self, fn, *a):
        self.q.put(("call", lambda: fn(*a)))

    def _inst_text(self):
        i = self.inst
        if not i:
            return "No instance running"
        return f"Instance #{i['id']} {i['state']} · " + ("$?/hr" if i["dph"] is None else f"${i['dph']:.3f}/hr")

    def _set_inst(self, iid, state="", dph=None, ready=True):
        """Switch to instance `iid` (None = no instance); it is shown on the status line and remembered."""
        self.inst = {"id": iid, "state": state, "dph": dph, "ready": ready} if iid else None
        self.ui(self._save)
        self.q.put(("inst", self._inst_text()))

    def _update_inst(self, state=None, dph=None, ready=None):
        """Update what is known about the current instance (None / 0 = unchanged)."""
        i = self.inst
        if i:
            i["state"] = state or i["state"]
            i["dph"] = dph or i["dph"]
            i["ready"] = i["ready"] if ready is None else ready
            self.q.put(("inst", self._inst_text()))

    def _pump(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "log":
                    self.logbox.insert("end", val + "\n")
                    self.logbox.see("end")
                elif kind in ("status", "inst"):  # status line = instance state + latest message
                    if kind == "inst":
                        self.inst_txt = val
                    else:
                        self.status_msg, self.status_err = val
                    self.status_lbl.config(text=f"{self.inst_txt} — {self.status_msg}",
                                           foreground="red" if self.status_err else "")
                elif kind == "call":
                    try:
                        val()
                    except Exception as e:
                        self.logbox.insert("end", f"UI error: {e}\n")
        except queue.Empty:
            pass
        self.after(100, self._pump)

    def _save(self):
        """Remember the output folder and instance ID; everything else in the settings file is kept as it is."""
        self.saved["local_out"] = self.out_var.get().strip()
        self.saved["mode"] = self.mode_var.get()
        self.saved["instance_id"] = str(self.inst["id"]) if self.inst else ""
        try:
            CONFIG_PATH.write_text(json.dumps(self.saved, indent=2), encoding="utf-8")
        except OSError as e:
            messagebox.showerror(APP, f"Could not save settings: {e}")

    def bg(self, fn, *a, job=False):
        if job:
            if self.job and self.job.is_alive():
                messagebox.showwarning(APP, "Already busy - wait for it to finish or press Stop first.")
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
        self.key_var = tk.StringVar()
        self.key_var.trace_add("write", lambda *_: setattr(self, "api_key", self.key_var.get().strip()))
        ttk.Label(top, text="Vast.ai API key").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=2)
        key_entry = ttk.Entry(top, textvariable=self.key_var, show="•")
        key_entry.grid(row=0, column=1, sticky="we", pady=2)
        key_entry.bind("<Return>", lambda e: self.on_connect())
        ttk.Button(top, text="Connect", command=self.on_connect).grid(row=0, column=2, sticky="we", padx=(6, 0), pady=2)
        key_entry.focus_set()
        self.in_var = tk.StringVar()
        self.out_var = tk.StringVar(value=self.saved.get("local_out") or str(Path.home() / "Music" / "UVR_output"))
        for r, (label, v, cmd) in enumerate((("Input audio file", self.in_var, self.on_browse_in),
                                             ("Output folder", self.out_var, self.on_browse_out)), start=1):
            ttk.Label(top, text=label).grid(row=r, column=0, sticky="w", padx=(0, 6), pady=2)
            ttk.Entry(top, textvariable=v).grid(row=r, column=1, sticky="we", pady=2)
            ttk.Button(top, text="Browse…", command=cmd).grid(row=r, column=2, padx=(6, 0), pady=2)

        mode = ttk.LabelFrame(self, text="What to make")
        mode.pack(fill="x", padx=8, pady=4)
        self.mode_var = tk.StringVar(value=self.saved.get("mode") if self.saved.get("mode") in MODES else "stems")
        for val, text in MODES.items():
            ttk.Radiobutton(mode, text=text, value=val, variable=self.mode_var,
                            command=self._save).pack(anchor="w", padx=6, pady=1)

        gpu = ttk.LabelFrame(self, text=f"GPU to rent - Europe · {' / '.join(GPU_NAMES)} · ≥{MIN_RELIABILITY:.1%} reliable · "
                                        f"≥{MIN_NET_MBPS:.0f} Mbit/s down + up · vacant (a running instance is used first)")
        gpu.pack(fill="both", padx=8, pady=4)
        bar = ttk.Frame(gpu)
        bar.pack(fill="x", padx=4, pady=(2, 0))
        ttk.Button(bar, text="Refresh", command=self.on_refresh_gpus).pack(side="left")
        self.gpu_lbl = ttk.Label(bar, text="Looking for GPUs...")
        self.gpu_lbl.pack(side="left", padx=8)
        cols = (("gpu_name", "GPU", 90), ("geolocation", "Where", 150), ("dph_total", "$/hour", 70),
                ("reliability2", "Reliability", 80), ("inet_down", "Down Mbit/s", 90), ("inet_up", "Up Mbit/s", 90),
                ("cpu_ram", "RAM", 60), ("id", "Offer", 90))
        box = ttk.Frame(gpu)
        box.pack(fill="both", expand=True, padx=4, pady=4)
        self.gpu_tree = ttk.Treeview(box, columns=[c for c, _, _ in cols], show="headings", height=7,
                                     selectmode="browse")
        for c, title, w in cols:
            self.gpu_tree.heading(c, text=title, command=lambda c=c: self._sort_gpus(c))
            self.gpu_tree.column(c, width=w, anchor="w" if c in ("gpu_name", "geolocation") else "e")
        sb = ttk.Scrollbar(box, orient="vertical", command=self.gpu_tree.yview)
        self.gpu_tree.configure(yscrollcommand=sb.set)
        self.gpu_tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")

        act = ttk.Frame(self)
        act.pack(fill="x", padx=8, pady=4)
        ttk.Button(act, text="Start", command=self.on_start).pack(side="left")
        ttk.Button(act, text="Stop", command=self.on_stop).pack(side="left", padx=6)
        ttk.Button(act, text="Destroy instance", command=self.on_destroy).pack(side="left")
        self.status_lbl = ttk.Label(self, text=f"{self.inst_txt} — {self.status_msg}", anchor="w")
        self.status_lbl.pack(fill="x", padx=8)
        self.logbox = ScrolledText(self, height=10, font=("Consolas", 9))
        self.logbox.pack(fill="both", expand=True, padx=8, pady=(4, 8))

    def on_browse_in(self):
        exts = " ".join("*" + e for e in sorted(AUDIO_EXT))
        title = "Mixdown to split into stems" if self.mode_var.get() == "stems" else "Lead vocal stem"
        p = filedialog.askopenfilename(title=title, filetypes=[("Audio", exts), ("All", "*.*")])
        if p:
            self.in_var.set(p)

    # ---------- GPU list ----------
    def on_refresh_gpus(self):
        if self.offers_busy or self.closing:
            return
        self.offers_busy = True
        self.gpu_lbl.config(text="Looking for GPUs...")
        key = self.api_key

        def work():
            try:
                offers = Vast(key, anonymous=True).gpu_offers()
                self.ui(self._got_offers, offers, None)
            except Exception as e:
                self.ui(self._got_offers, None, e)
        threading.Thread(target=work, daemon=True).start()

    def _got_offers(self, offers, err):
        self.offers_busy = False
        if err is not None:
            self.gpu_lbl.config(text="Could not load the GPU list - press Refresh. (see the log)")
            self.log(f"GPU list: {err}")
            return
        self.offers = offers
        self._show_offers()
        when = time.strftime("%H:%M")
        self.gpu_lbl.config(text=f"{len(offers)} GPUs match right now (checked {when}) - click one to pick it."
                            if offers else f"No GPU matches right now (checked {when}) - press Refresh in a few minutes.")

    def _sort_gpus(self, col):
        key, desc = self.offer_sort
        self.offer_sort = (col, not desc if col == key else col in ("reliability2", "inet_down", "inet_up", "cpu_ram"))
        self._show_offers()

    def _show_offers(self):
        """Fill the list in the chosen sort order; the picked GPU stays picked if it is still there."""
        col, desc = self.offer_sort
        text_cols = ("gpu_name", "geolocation")
        self.offers.sort(key=lambda o: str(o.get(col) or "") if col in text_cols else _num(o, col), reverse=desc)
        picked = self.gpu_tree.selection()
        self.gpu_tree.delete(*self.gpu_tree.get_children())
        for o in self.offers:
            self.gpu_tree.insert("", "end", iid=str(o["id"]), values=(
                o.get("gpu_name"), o.get("geolocation") or "?", f"${_num(o, 'dph_total'):.3f}",
                f"{_num(o, 'reliability2') * 100:.2f}%", f"{_num(o, 'inet_down'):.0f}", f"{_num(o, 'inet_up'):.0f}",
                f"{_num(o, 'cpu_ram') / 1024:.0f} GB", o["id"]))
        keep = [i for i in picked if self.gpu_tree.exists(i)]
        if keep:
            self.gpu_tree.selection_set(keep)
            self.gpu_tree.see(keep[0])

    def _picked_offers(self):
        """The picked offer, then - in case someone rents it first - the same GPU model listed at the same
        price or cheaper, cheapest first. Nothing pricier or of the other model is ever rented unasked."""
        sel = self.gpu_tree.selection()
        picked = next((o for o in self.offers if sel and str(o["id"]) == sel[0]), None)
        if not picked:
            return []
        same = [o for o in self.offers if o is not picked and o.get("gpu_name") == picked.get("gpu_name")
                and _num(o, "dph_total") <= _num(picked, "dph_total") + 1e-6]
        return [picked] + sorted(same, key=lambda o: _num(o, "dph_total"))

    def on_browse_out(self):
        p = filedialog.askdirectory(title="Output folder")
        if p:
            self.out_var.set(p)
            self._save()

    # ---------- instances ----------
    def on_connect(self):
        """Connect with the API key typed in the window (and check the saved instance, if any)."""
        if not self.api_key:
            messagebox.showerror(APP, "Enter your Vast.ai API key first.")
            return
        i = self.inst
        self.on_find_instances(i["id"] if i and i["state"] == "not checked yet" else None, connect=True)

    def on_find_instances(self, check=None, connect=False):
        """Log what is billing on the account; `check` = the instance ID saved at launch, to reconnect to."""
        key = self.api_key
        if not key:
            return

        def work():
            try:
                insts = Vast(key).instances()
            except Exception:
                if check or connect:
                    self.status("Could not connect - see the log." if connect else "Could not check it - see the log.",
                                True)
                raise
            if connect:
                self.log("Connected to Vast.ai - API key accepted.")
                self.status("Connected - choose a file, pick a GPU and press Start.")
            self.ui(self._got_instances, insts, check)
        if check:
            self.status("Checking that it still exists...")
        elif connect:
            self.status("Connecting to Vast.ai...")
        self.bg(work)

    def _got_instances(self, insts, check=None):
        by_id = {int(i.get("id", -1)): i for i in insts}
        cur = self.inst
        # reconnect to the saved instance, or forget it if gone (a running job checks for itself)
        startup = bool(check and cur and cur["id"] == check and not (self.job and self.job.is_alive()))
        if startup and check in by_id:
            i = by_id[check]
            self._update_inst(i.get("actual_status") or "?", float(i.get("dph_total") or 0))
            self.log(f"Reconnected to instance #{cur['id']} ({cur['state']}, ${cur['dph'] or 0:.3f}/hr) - "
                     "it keeps billing until you press Destroy instance.")
            self.status("Reconnected - choose a file and press Start.")
        elif startup:
            self.log(f"Saved instance #{cur['id']} no longer exists - the GPU you pick is rented on the next Start.")
            self._set_inst(None)
            self.status("Ready.")
        cur_id = self.inst["id"] if self.inst else None
        mine = [i for i in insts if i.get("label") == UVR_LABEL and int(i.get("id", -1)) != cur_id]
        others = [i for i in insts if i.get("label") != UVR_LABEL and int(i.get("id", -1)) != cur_id]
        for i in mine:
            self.log(f"UVR instance #{i.get('id')}: {i.get('actual_status') or '?'} · {i.get('num_gpus', 1)}x "
                     f"{i.get('gpu_name', '?')} · ${float(i.get('dph_total') or 0):.3f}/hr - billing until destroyed "
                     "(not the one this app uses)")
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

    def _destroy_inst(self, vast):
        """Destroy the current instance: the status line confirms it is gone, or a loud error says it may bill."""
        iid = self.inst["id"]
        self.destroying = True
        try:
            ok = vast is not None and self._destroy(vast, iid)
        finally:
            self.destroying = False
        if ok:
            self._set_inst(None)
            self.status(f"Instance #{iid} destroyed - confirmed gone.")
        else:
            self.status(f"!! Instance #{iid} MAY STILL BE BILLING - destroy it at cloud.vast.ai/instances", True)
            self.ui(messagebox.showerror, APP, f"Could not confirm that instance #{iid} was destroyed.\n\n"
                    "IT MAY STILL BE BILLING.\n\nDestroy it now at cloud.vast.ai/instances.")
        return ok

    def _vast_or_none(self):
        try:
            return Vast(self.api_key)
        except RuntimeError as e:
            self.log(f"ERROR: {e}")
            return None

    def on_destroy(self):
        if self.closing:
            return
        if not self.inst:
            self.status("No instance to destroy.")
            return
        self.bg(self._destroy_job, job=True)

    def _destroy_job(self):
        self._destroy_inst(self._vast_or_none())
        self.ui(self.on_find_instances)
        self.ui(self.on_refresh_gpus)

    # ---------- the job ----------
    def on_start(self):
        if self.closing:
            return
        src, out, mode = self.in_var.get().strip(), self.out_var.get().strip(), self.mode_var.get()
        if not self.api_key:
            messagebox.showerror(APP, "Enter your Vast.ai API key first.")
            return
        if not os.path.isfile(src):
            messagebox.showerror(APP, f"Input file not found:\n{src}" if src else "Choose an input audio file first.")
            return
        if not out:
            messagebox.showerror(APP, "Choose an output folder first.")
            return
        picks = self._picked_offers()
        if not self.inst and not picks:
            messagebox.showerror(APP, "Pick a GPU in the list first (click a row)." if self.offers else
                                 "No GPU matches right now - press Refresh, then pick one.")
            return
        self._save()
        self.bg(self._job, src, Path(out).expanduser(), mode, picks, job=True)

    def on_stop(self):
        """Stop the job. A machine that is still being rented or set up is destroyed; a set-up one keeps running."""
        if self.job and self.job.is_alive() and not self.destroying and not self.stop_event.is_set():
            self.stop_event.set()
            i = self.inst
            self.status("Stopping - this half-set-up instance will be destroyed..." if i and not i["ready"]
                        else "Stopping - the instance keeps running..." if i else "Stopping...")

    def _job(self, src, out_local, mode, picks):
        vast, t0, dph = None, time.time(), 0.0
        result, failed = "Stopped.", False
        try:
            vast = Vast(self.api_key)
            pub = load_or_create_key(self.ssh_key, self.log)
            out_local.mkdir(parents=True, exist_ok=True)
            if self.inst:
                self.status(f"Checking instance #{self.inst['id']}...")
                if all(int(i.get("id", -1)) != self.inst["id"] for i in vast.instances()):
                    self.log(f"Instance #{self.inst['id']} no longer exists - renting the GPU you picked.")
                    self._set_inst(None)
                else:
                    self.log(f"Using the running instance #{self.inst['id']} (the GPU list is for renting a new one).")
            if not self.inst:
                if not picks:
                    raise RuntimeError("No GPU picked - click one in the list, then press Start again.")
                self._ensure_account_key(vast, pub)
                self._rent_picked(vast, picks)
            r, dph = self._connect_ready(vast, self.inst["id"], pub)
            try:
                if mode == "stems":
                    made, problems, where = self._run_stems(r, src, out_local)
                    result = f"Done - {made} stems in {where}"
                else:
                    made, problems = self._run_job(r, src, out_local)
                    where, result = out_local, f"Done - {made} dry file(s) in {out_local}"
            finally:
                r.close()
            result += " (with problems, see log)" if problems else ""
            self.ui(self.bell)
            self.ui(os.startfile, str(where))  # open the folder, ready to drag into FL Studio
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
            i = self.inst
            if self.stop_event.is_set() and i and not i["ready"]:
                self.log(f"Stopped during rental/setup - destroying instance #{i['id']} so it doesn't keep billing.")
                if self._destroy_inst(vast):
                    self.status(f"Stopped - instance #{i['id']} destroyed, confirmed gone.")
            else:
                self.status(result, failed)
            self.ui(self.on_find_instances)
            self.ui(self.on_refresh_gpus)

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

    def _rent_picked(self, vast, picks):
        """Rent the picked offer; if someone rented it first, the next of `picks` (same GPU, not pricier)."""
        self.status(f"Renting the picked {picks[0].get('gpu_name')} in {picks[0].get('geolocation') or '?'}...")
        for n, o in enumerate(picks[:6]):
            self.check_stop()
            if not offer_ok(o):  # never rent anything outside the list's requirements
                continue
            if n:
                self.log(f"Trying {offer_desc(o)} instead (same GPU, same price or cheaper)...")
            try:
                return self._rent(vast, o)
            except RuntimeError as e:
                if not _offer_gone(e):
                    raise
                self.log(f"Offer #{o['id']} was just rented by someone else.")
        raise RuntimeError("The GPU you picked (and the same-or-cheaper ones like it) were just rented by others - "
                           "press Refresh and pick another.")

    def _rent(self, vast, offer):
        body = {"client_id": "me", "image": IMAGE, "disk": DISK_GB, "label": UVR_LABEL,
                "runtype": RUNTYPE, "onstart": ONSTART, "env": {}}
        self.log("Renting " + offer_desc(offer) + " ...")
        res = vast.create(offer["id"], body)
        iid = res.get("new_contract")
        if iid:
            self._set_inst(int(iid), "starting", float(offer.get("dph_total") or 0) or None, ready=False)
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
            self._update_inst(st, float(inst.get("dph_total") or 0))
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
        if iid not in self.attached:
            try:
                vast.req("POST", f"/instances/{iid}/ssh/", json={"ssh_key": pub})
            except Exception as e:
                self.log(f"(attach SSH key to instance: {e})")
            self.attached.add(iid)
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
                    self._update_inst("running", ready=True)
                    break
                self._update_inst("setting up", ready=False)
                if state == "__FAILED__" or (state == "__NOLOG__" and time.time() - since > 120):
                    if repaired:
                        self._update_inst("setup failed")
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

    def _separate(self, r, sftp, model, in_path, out_dir, settings=SEPARATOR_ARGS):
        """Run audio-separator on one remote WAV named f<n>.wav; returns {stem: remote path} for every stem."""
        args = uvr_args(model, out_dir, settings) + [in_path]
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

    def _upload_decode(self, r, sftp, src, base):
        """Upload `src` into `base` and decode it to the stereo float WAV f1.wav.
        Returns (WAV path, the file's own sample rate - None if unknown)."""
        up, wav = f"{base}/src{Path(src).suffix.lower()}", f"{base}/f1.wav"
        self.log(f"Upload: {Path(src).name} ({os.path.getsize(src) / 1e6:.1f} MB)")
        sftp.put(src, up, callback=self._progress(f"Uploading {Path(src).name}"))
        _, rate = r.run("ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of csv=p=0 "
                        + q(up), timeout=60)
        rate = rate.strip().splitlines()[0].strip() if rate.strip() else ""
        self.log(f"Decoding → {SAMPLE_RATE} Hz stereo" + (f" (the file is {rate} Hz)" if rate.isdigit() else ""))
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
        return wav, (rate if rate.isdigit() and 8000 <= int(rate) <= 384000 else None)

    def _fetch_flac(self, r, sftp, path, remote, dst, rate):
        """Encode remote WAV `path` to 24-bit FLAC at `rate` and download it to `dst`. Returns True if done."""
        ff = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", path, *flac_24(rate), remote]
        r.stream(" ".join(q(x) for x in ff), self._remote_line, self.stop_event)
        self.check_stop()
        try:
            sftp.stat(remote)
            sftp.get(remote, str(dst), callback=self._progress(f"Downloading {dst.name}"))
            self.log(f"  ↓ {dst}")
            return True
        except IOError as e:
            self.log(f"!! could not download {dst.name}: {e}")
            return False

    def _run_stems(self, r, src, out_local):
        """Upload a mixdown, run the STEM_STEPS cascade and download every STEM_FILES stem as 24-bit FLAC
        into a new '<song> stems' folder. Returns (stems made, problems, folder)."""
        base = f"{REMOTE_DIR}/job_{time.strftime('%Y%m%d_%H%M%S')}"
        r.run(f"mkdir -p {q(base)}/final {q(REMOTE_DIR)}/models")
        sftp = r.client.open_sftp()
        song, made, problems = Path(src).stem, 0, 0
        try:
            wav, rate = self._upload_decode(r, sftp, src, base)
            out_rate = rate or OUT_RATE
            # 1) the cascade: each model on the mixdown or on a stem an earlier model made
            stems = {}  # model -> {stem lower-case: remote WAV}
            for n, (model, feed) in enumerate(STEM_STEPS, 1):
                self.check_stop()
                inp = wav
                if feed:
                    prev = stems.get(feed[0], {}).get(feed[1])
                    if not prev:
                        self.log(f"!! step {n}/{len(STEM_STEPS)} ({model}) skipped - no '{feed[1]}' stem to run it on")
                        continue
                    inp = f"{base}/f{n}.wav"  # audio-separator names its outputs after the input: keep it short
                    r.run(f"ln -sf {q(prev)} {q(inp)}")
                what = "the mixdown" if not feed else f"the {feed[1]} stem"
                self.log(f"Step {n}/{len(STEM_STEPS)}: {model} on {what}")
                self.status(f"Separating - step {n}/{len(STEM_STEPS)} ({Path(model).stem} on {what})...")
                got = self._separate(r, sftp, model, inp, f"{base}/m{n}", STEM_ARGS + STEM_EXTRA.get(model, []))
                stems[model] = {s.lower(): p for s, p in got.items()}
                if got:
                    self.log(f"  → {', '.join(sorted(got))}")

            # 2) one folder per song, numbered files in a fixed order
            tail = " stems"
            keep = max(10, min(120, 240 - len(str(out_local)) - len(tail) - 40))  # Windows: 260-char paths
            folder = out_local / safe_name(song[:keep] + tail)
            for k in range(2, 100):
                if not folder.exists():
                    break
                folder = out_local / safe_name(f"{song[:keep]}{tail} ({k})")
            folder.mkdir(parents=True)
            self.log(f"Saving stems as 24-bit FLAC at {out_rate} Hz into {folder}")
            for n, (name, model, stem) in enumerate(STEM_FILES, 1):
                self.check_stop()
                path = stems.get(model, {}).get(stem)
                if not path:
                    self.log(f"!! no '{stem}' stem from {model} - '{name}' is missing")
                    problems += 1
                    continue
                self.status(f"Encoding + downloading {name} ({n}/{len(STEM_FILES)})...")
                if self._fetch_flac(r, sftp, path, f"{base}/final/{n}.flac", folder / f"{n:02d} {name}.flac", out_rate):
                    made += 1
                else:
                    problems += 1
        finally:
            sftp.close()
        if not made:
            raise RuntimeError("No stem was made - see the log.")
        return made, problems, folder

    def _run_job(self, r, src, out_local):
        """Upload `src`, make each RECIPES dry version of it and download each as 24-bit FLAC.
        Returns (files made, problems)."""
        base = f"{REMOTE_DIR}/job_{time.strftime('%Y%m%d_%H%M%S')}"
        r.run(f"mkdir -p {q(base)}/final {q(REMOTE_DIR)}/models")
        sftp = r.client.open_sftp()
        song, made, problems = Path(src).stem, 0, 0
        try:
            # 1) upload, then decode to stereo float WAV
            wav, _ = self._upload_decode(r, sftp, src, base)

            # 2) each recipe: its models in turn, each on the previous one's dry stem; encode + download the last
            done, runs = {(): ("input", wav)}, 0  # models run so far -> (dry stem, remote path), None = failed
            for i, recipe in enumerate(RECIPES, 1):
                self.log(f"Dry version {i}/{len(RECIPES)}: {' → '.join(recipe)}")
                for n in range(1, len(recipe) + 1):
                    key, prev = recipe[:n], done.get(recipe[:n - 1])
                    if key in done or prev is None:
                        done.setdefault(key, None)
                        continue
                    self.check_stop()
                    runs += 1
                    inp = wav
                    if n > 1:  # audio-separator names its outputs after the input: give it a short f<n>.wav name
                        inp = f"{base}/f{runs}.wav"
                        r.run(f"ln -sf {q(prev[1])} {q(inp)}")
                    model = key[-1]
                    stems = self._separate(r, sftp, model, inp, f"{base}/m{runs}")
                    dry = next((s for s in stems if s.lower() == DRY_STEM[model]), None)
                    if not dry:
                        self.log(f"!! {model}: no '{DRY_STEM[model]}' stem among {sorted(stems)}")
                    done[key] = (dry, stems[dry]) if dry else None
                if not done[recipe]:
                    self.log(f"!! dry version {i} failed - see above")
                    problems += 1
                    continue
                dry, path = done[recipe]
                tail = f"_({dry})_{' + '.join(Path(m).stem for m in recipe)}"
                keep = max(10, min(180, 250 - len(str(out_local)) - len(".flac")) - len(tail))  # Windows: 260-char paths
                dst = out_local / f"{safe_name(song[:keep] + tail)}.flac"
                self.log(f"Encoding {dry} → 24-bit FLAC ...")
                if self._fetch_flac(r, sftp, path, f"{base}/final/{i}.flac", dst, OUT_RATE):
                    made += 1
                else:
                    problems += 1
        finally:
            sftp.close()
        if not made:
            raise RuntimeError("No dry file was made - see the log.")
        return made, problems

    # ---------- shutdown ----------
    def _on_close(self):
        if self.closing:
            return
        choice = self._ask_close() if self.inst else "keep"
        if choice is None:
            return
        self.closing = True
        busy = bool(self.job and self.job.is_alive())
        if busy and choice == "keep" and self.inst and not self.inst["ready"]:
            busy = False  # leave a machine that is still being set up alone: it finishes setup by itself
        if busy:
            self.stop_event.set()
            self.status("Stopping the job before closing...")
        threading.Thread(target=self._close_work, args=(choice, busy), daemon=True).start()

    def _ask_close(self):
        """Ask what to do with the running instance: 'destroy', 'keep' or None (cancel)."""
        i, ans = self.inst, []
        d = tk.Toplevel(self)
        d.title(APP)
        d.transient(self)
        d.resizable(False, False)
        dph = "$?" if i["dph"] is None else f"${i['dph']:.3f}"
        ttk.Label(d, text=f"Instance #{i['id']} still billing at {dph}/hr. Destroy it or keep it running?").pack(
            padx=16, pady=(16, 10))
        row = ttk.Frame(d)
        row.pack(padx=16, pady=(0, 14))
        for text, val in (("Destroy", "destroy"), ("Keep running", "keep"), ("Cancel", None)):
            ttk.Button(row, text=text, command=lambda v=val: (ans.append(v), d.destroy())).pack(side="left", padx=4)
        d.grab_set()
        self.wait_window(d)
        return ans[0] if ans else None

    def _close_work(self, choice, busy):
        if busy:
            self.job.join()
        if choice == "destroy" and self.inst and not self._destroy_inst(self._vast_or_none()):
            self.closing = False  # loud error shown - stay open
            return
        self.ui(self._finish_close)

    def _finish_close(self):
        self._save()
        self.after(0, self.destroy)  # after the current queue pass; no stop here, a kept machine may be setting up


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
