#!/usr/bin/env python3
"""
UVR on Vast.ai - one-file GUI.

Shape the whole Ultimate Vocal Remover pipeline - input slicing, one or many
models, UVR-style ensembling and inversion, chained post-processing passes
(de-reverb, de-noise, karaoke, Demucs band splits...), every per-architecture
parameter and the final encode - then rent a GPU on vast.ai, run it there with
python-audio-separator, download the stems and destroy the machine.

Run:  python uvr_vast_gui.py      (installs `requests` + `paramiko` on first run)
Settings: ~/.uvr_vast_gui.json    Presets: ~/.uvr_vast_presets.json
"""

import fnmatch
import io
import itertools
import json
import os
import posixpath
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
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
PRESETS_PATH = Path.home() / ".uvr_vast_presets.json"
API = "https://console.vast.ai/api/v0"
API_V1 = "https://console.vast.ai/api/v1"
IMAGE = "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime"
UVR_LABEL = "uvr"  # only instances with this label are auto-picked or offered for destruction
RUNTYPE = "ssh_direc ssh_proxy"
READY_MARKER = "/root/uvr_ready"
SETUP_LOG = "/root/uvr_setup.log"
PATH_PREFIX = "export PATH=/opt/conda/bin:/usr/local/bin:$PATH PYTHONUNBUFFERED=1; "
DEFAULT_TEMPLATE = "{base}_({lineage})_{model}"
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

# Runs on the GPU box: UVR-style ensembling and mix-minus-stem inversion (numpy/scipy/soundfile
# all come with audio-separator).
FX_PY = r'''
import json, sys
import numpy as np
import soundfile as sf
from scipy.signal import stft, istft


def load(path):
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.T
    if y.shape[0] == 1:
        y = np.repeat(y, 2, axis=0)
    return y[:2], sr


def spec(y, n_fft, hop):
    return stft(y, nperseg=n_fft, noverlap=n_fft - hop)[2]


def unspec(z, n_fft, hop, n):
    y = istft(z, nperseg=n_fft, noverlap=n_fft - hop)[1].astype(np.float32)[:n]
    return np.pad(y, (0, n - len(y)))


def ensemble(op):
    loaded = [load(p) for p in op["inputs"]]
    sr = loaded[0][1]
    n = min(y.shape[1] for y, _ in loaded)
    Y = np.stack([y[:, :n] for y, _ in loaded])
    w = np.ones(len(loaded), np.float32)
    for i, v in enumerate((op.get("weights") or [])[:len(loaded)]):
        w[i] = v
    w /= w.sum()
    algo, n_fft, hop = op["algo"], op["n_fft"], op["hop"]
    if algo == "avg_wave":
        out = np.tensordot(w, Y, axes=1)
    elif algo == "median_wave":
        out = np.median(Y, axis=0)
    elif algo in ("max_wave", "min_wave"):
        idx = (np.argmax if algo == "max_wave" else np.argmin)(np.abs(Y), axis=0)
        out = np.take_along_axis(Y, idx[None], axis=0)[0]
    else:
        chans = []
        for ch in range(2):
            S = np.stack([spec(y[ch], n_fft, hop) for y in Y])
            mag = np.abs(S)
            if algo == "avg_spec":
                Z = np.tensordot(w, mag, axes=1) * np.exp(1j * np.angle(np.tensordot(w, S, axes=1)))
            else:
                if algo == "max_spec":
                    idx = np.argmax(mag, axis=0)
                elif algo == "min_spec":
                    idx = np.argmin(mag, axis=0)
                else:
                    idx = np.argsort(mag, axis=0)[len(loaded) // 2]
                Z = np.take_along_axis(S, idx[None], axis=0)[0]
            chans.append(unspec(Z, n_fft, hop, n))
        out = np.stack(chans)
    sf.write(op["out"], out.T, sr, subtype="FLOAT")


def invert(op):
    y, sr = load(op["stem"])
    m, _ = load(op["mix"])
    n = min(y.shape[1], m.shape[1])
    y, m = y[:, :n], m[:, :n]
    if op.get("method") == "spec":
        chans = []
        for ch in range(2):
            M, S = spec(m[ch], op["n_fft"], op["hop"]), spec(y[ch], op["n_fft"], op["hop"])
            Z = np.maximum(np.abs(M) - np.abs(S), 0) * np.exp(1j * np.angle(M))
            chans.append(unspec(Z, op["n_fft"], op["hop"], n))
        out = np.stack(chans)
    else:
        out = m - y
    sf.write(op["out"], out.T, sr, subtype="FLOAT")


for op in json.load(open(sys.argv[1])):
    try:
        {"ensemble": ensemble, "invert": invert}[op["op"]](op)
        print("fx ok:", op["op"], op["out"].rsplit("/", 1)[-1], flush=True)
    except Exception as e:
        print("!! fx failed:", op["op"], op["out"], type(e).__name__, e, flush=True)
'''

GPUS = ["", "RTX 4090", "RTX 5090", "RTX 3090", "RTX 4090, RTX 3090", "RTX 4090, RTX 5090",
        "RTX 4080", "RTX 3080", "RTX A6000", "A40", "L40S", "A100 SXM4", "A100 PCIE", "H100 SXM", "H100 PCIE"]
ORDERS = ["dph_total asc", "score desc", "dlperf_per_dphtotal desc", "reliability2 desc",
          "inet_down desc", "dlperf desc", "gpu_ram desc"]
ROFORMER = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
MODEL_PRESETS = [
    ROFORMER,
    "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
    "vocals_mel_band_roformer.ckpt",
    "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
    "MDX23C-8KFFT-InstVoc_HQ.ckpt",
    "UVR-MDX-NET-Inst_HQ_4.onnx",
    "UVR-MDX-NET-Inst_HQ_3.onnx",
    "UVR-MDX-NET-Voc_FT.onnx",
    "Kim_Vocal_2.onnx",
    "UVR_MDXNET_KARA_2.onnx",
    "Reverb_HQ_By_FoxJoy.onnx",
    "UVR-MDX-NET-Crowd_HQ_1.onnx",
    "htdemucs_ft.yaml",
    "htdemucs_6s.yaml",
    "htdemucs.yaml",
    "hdemucs_mmi.yaml",
    "6_HP-Karaoke-UVR.pth",
    "5_HP-Karaoke-UVR.pth",
    "UVR-DeEcho-DeReverb.pth",
    "UVR-De-Echo-Normal.pth",
    "UVR-DeNoise.pth",
    "1_HP-UVR.pth",
    "9_HP2-UVR.pth",
    "17_HP-Wind_Inst-UVR.pth",
    "deverb_bs_roformer_8_384dim_10depth.ckpt",
    "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
]
STEMS = ["", "Vocals", "Instrumental", "Drums", "Bass", "Other", "Guitar", "Piano",
         "No Reverb", "Reverb", "No Echo", "Echo", "No Noise", "Noise", "Karaoke"]
STEM_FILTERS = ["Vocals", "Instrumental", "No *", "*", "Vocals, Instrumental", "Drums", "Bass", "Other",
                "Guitar", "Piano", "No Vocals"]
ENS_ALGOS = ["avg_wave", "median_wave", "max_wave", "min_wave", "avg_spec", "median_spec", "max_spec", "min_spec"]
FORMATS = {"flac": "flac", "wav": "wav", "aiff": "aiff", "alac": "m4a", "mp3": "mp3", "ogg": "ogg",
           "opus": "opus", "m4a (aac)": "m4a"}
AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus", ".aiff", ".aif",
             ".wma", ".mp4", ".mkv", ".webm", ".alac"}
OUT_RE = re.compile(r"^f(\d+)_\((.+?)\)_.+\.[A-Za-z0-9]+$")
LOOSE_RE = re.compile(r"^f(\d+)\W?(.*)\.[A-Za-z0-9]+$")


def _step(src, stems, model, dl=True, single="", args=""):
    return {"src": src, "stems": stems, "model": model, "single": single, "args": args, "dl": dl}


BUILTIN_PRESETS = {
    "★ Vocals + instrumental (BS-Roformer)": {
        "models": ROFORMER + "\n", "primary_mode": "separate", "single_stem": "", "invert_stems": "", "chain": []},
    "★ Clean lead vocal: Roformer → De-Echo/De-Reverb → De-Noise": {
        "models": ROFORMER + "\n", "primary_mode": "separate", "single_stem": "", "invert_stems": "", "chain": [
            _step("primary", "Vocals", "UVR-DeEcho-DeReverb.pth", dl=False),
            _step("previous", "No *", "UVR-DeNoise.pth")]},
    "★ Karaoke: lead vs backing vocals": {
        "models": ROFORMER + "\n", "primary_mode": "separate", "single_stem": "", "invert_stems": "", "chain": [
            _step("primary", "Vocals", "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt")]},
    "★ Full split: Roformer vocals + Demucs 6-stem band": {
        "models": ROFORMER + "\n", "primary_mode": "separate", "single_stem": "", "invert_stems": "", "chain": [
            _step("primary", "Instrumental", "htdemucs_6s.yaml")]},
    "★ Ensemble: Roformer + MDX23C + Inst HQ4 (avg_spec)": {
        "models": f"{ROFORMER}\nMDX23C-8KFFT-InstVoc_HQ.ckpt\nUVR-MDX-NET-Inst_HQ_4.onnx\n",
        "primary_mode": "ensemble", "ens_algo": "avg_spec", "single_stem": "", "invert_stems": "", "chain": []},
    "★ Ensemble max_spec vocals + inverted instrumental": {
        "models": f"{ROFORMER}\nMDX23C-8KFFT-InstVoc_HQ.ckpt\n", "primary_mode": "ensemble", "ens_algo": "max_spec",
        "single_stem": "Vocals", "invert_stems": "Vocals", "chain": []},
    "★ Test slice: 30 s from 1:00": {"sample_start": "60", "sample_duration": "30"},
    "★ Whole files (no slicing)": {"sample_start": "", "sample_duration": ""},
    "★ Delivery: 24-bit WAV 48 kHz": {"final_format": "wav", "bit_depth": "24", "final_sr": "48000"},
    "★ Delivery: MP3 320 at -14 LUFS": {"final_format": "mp3", "lossy_bitrate": "320k", "lossy_vbr": "",
                                        "loudnorm_lufs": "-14"},
}


class StopJob(Exception):
    pass


def INT(s):
    return int(float(s))


def num(c, key, cast=float, required=False):
    s = str(c.get(key, "")).strip()
    if not s:
        if required:
            raise RuntimeError(f"Setting '{key}' is required.")
        return None
    try:
        return cast(s)
    except ValueError:
        raise RuntimeError(f"Bad value for '{key}': {s!r}")


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def split_list(s):
    return [x.strip() for x in re.split(r"[,;]", s or "") if x.strip()]


def stem_match(stem, patterns):
    return not patterns or any(fnmatch.fnmatch(stem.lower(), p.lower()) for p in patterns)


def safe_name(s):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", s).strip(" .")[:180] or "output"


def parse_models(c):
    return [m.strip() for m in c["models"].splitlines() if m.strip() and not m.strip().startswith("#")]


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


def offer_query(c):
    qy = {"rentable": {"eq": True}, "rented": {"eq": False}, "external": {"eq": False}}
    if c["verified_only"]:
        qy["verified"] = {"eq": True}
    gpus = [g.strip() for g in c["gpu_names"].split(",") if g.strip()]
    if len(gpus) == 1:
        qy["gpu_name"] = {"eq": gpus[0]}
    elif gpus:
        qy["gpu_name"] = {"in": gpus}
    for key, field, op, cast, mult in (
        ("num_gpus", "num_gpus", "eq", INT, 1),
        ("min_vram_gb", "gpu_ram", "gte", float, 1000),
        ("max_dph", "dph_total", "lte", float, 1),
        ("min_reliability", "reliability2", "gte", float, 1),
        ("disk_gb", "disk_space", "gte", float, 1),
        ("min_inet_down", "inet_down", "gte", float, 1),
        ("min_cuda", "cuda_max_good", "gte", float, 1),
        ("min_cpu_ram_gb", "cpu_ram", "gte", float, 1000),
    ):
        v = num(c, key, cast)
        if v is not None:
            qy[field] = {op: v * mult}
    field, _, direction = c["order_by"].strip().partition(" ")
    qy["order"] = [[field or "dph_total", direction.strip() or "asc"]]
    qy["type"] = "bid" if c["offer_type"] == "interruptible" else "on-demand"
    qy["limit"] = num(c, "limit", INT) or 64
    qy["allocated_storage"] = num(c, "disk_gb", float) or 10
    return qy


def uvr_args(c, model, out_dir, model_dir, single_stem="", extra=""):
    """audio-separator CLI arguments (everything except the input files). Stems are always WAV;
    the final format is produced by the ffmpeg encode stage."""
    a = ["-m", model, "--output_dir", out_dir, "--model_file_dir", model_dir,
         "--output_format", "WAV", "--log_level", c["log_level"] or "info"]

    def opt(flag, key, cast=None):
        v = num(c, key, cast) if cast else (str(c.get(key, "")).strip() or None)
        if v is not None:
            a.extend([flag, str(v)])

    def flag(f, key):
        if c.get(key):
            a.append(f)

    if (single_stem or "").strip():
        a += ["--single_stem", single_stem.strip()]
    opt("--normalization", "normalization", float)
    opt("--amplification", "amplification", float)
    opt("--sample_rate", "sample_rate", INT)
    flag("--invert_spect", "invert_spect")
    flag("--use_soundfile", "use_soundfile")
    flag("--use_autocast", "use_autocast")
    # MDX-Net
    opt("--mdx_segment_size", "mdx_segment_size", INT)
    opt("--mdx_overlap", "mdx_overlap", float)
    opt("--mdx_batch_size", "mdx_batch_size", INT)
    opt("--mdx_hop_length", "mdx_hop_length", INT)
    flag("--mdx_enable_denoise", "mdx_enable_denoise")
    # VR Arch
    opt("--vr_batch_size", "vr_batch_size", INT)
    opt("--vr_window_size", "vr_window_size", INT)
    opt("--vr_aggression", "vr_aggression", INT)
    flag("--vr_enable_tta", "vr_enable_tta")
    flag("--vr_high_end_process", "vr_high_end_process")
    flag("--vr_enable_post_process", "vr_enable_post_process")
    opt("--vr_post_process_threshold", "vr_post_process_threshold", float)
    # Demucs
    opt("--demucs_segment_size", "demucs_segment_size")
    opt("--demucs_shifts", "demucs_shifts", INT)
    opt("--demucs_overlap", "demucs_overlap", float)
    if not c.get("demucs_segments_enabled", True):
        a.append("--demucs_segments_enabled=")  # argparse type=bool: empty string -> False
    # MDXC / Roformer
    opt("--mdxc_segment_size", "mdxc_segment_size", INT)
    flag("--mdxc_override_model_segment_size", "mdxc_override_model_segment_size")
    opt("--mdxc_overlap", "mdxc_overlap", INT)
    opt("--mdxc_batch_size", "mdxc_batch_size", INT)
    opt("--mdxc_pitch_shift", "mdxc_pitch_shift", INT)
    for s in (c.get("extra_args", ""), extra):
        try:
            a += shlex.split(s or "")
        except ValueError as e:
            raise RuntimeError(f"Bad extra args {s!r}: {e}")
    return a


def encode_args(c):
    """ffmpeg output arguments for the final encode -> (args, file extension)."""
    fmt, depth = c["final_format"], c["bit_depth"]
    if fmt not in FORMATS:
        raise RuntimeError(f"Unknown output format {fmt!r}")
    lossless = fmt in ("wav", "aiff", "flac", "alac")
    filters = []
    gain = num(c, "gain_db", float)
    if gain:
        filters.append(f"volume={gain}dB")
    lufs = num(c, "loudnorm_lufs", float)
    if lufs is not None:
        filters.append(f"loudnorm=I={lufs}:TP={num(c, 'loudnorm_tp', float) or -1.0}"
                       f":LRA={num(c, 'loudnorm_lra', float) or 11}")
    if c["extra_af"].strip():
        filters.append(c["extra_af"].strip())
    sr = c["final_sr"].strip()
    sr_val = INT(sr) if sr and sr != "keep" else None
    if lufs is not None and not sr_val:  # loudnorm outputs 192 kHz unless told otherwise
        sr_val = num(c, "sample_rate", INT) or 44100
    res = [str(sr_val)] if sr_val else []
    if c["hq_resample"] and res:
        res += ["resampler=soxr", "precision=28"]
    if lossless and depth == "16" and c["dither"] != "none":
        res += [f"out_sample_fmt={'s16p' if fmt == 'alac' else 's16'}", f"dither_method={c['dither']}"]
    if res:
        filters.append("aresample=" + ":".join(res))
    a = ["-af", ",".join(filters)] if filters else []
    if c["final_channels"] == "mono":
        a += ["-ac", "1"]
    elif c["final_channels"] == "stereo":
        a += ["-ac", "2"]
    br = c["lossy_bitrate"].strip() or "320k"
    vbr = num(c, "lossy_vbr", float)
    if fmt == "wav":
        a += ["-c:a", {"16": "pcm_s16le", "24": "pcm_s24le"}.get(depth, "pcm_f32le")]
    elif fmt == "aiff":
        a += ["-c:a", {"16": "pcm_s16be", "24": "pcm_s24be"}.get(depth, "pcm_s32be")]
    elif fmt == "flac":
        level = num(c, "flac_level", INT)
        a += ["-c:a", "flac", "-compression_level", str(5 if level is None else level)]
        a += ["-sample_fmt", "s16"] if depth == "16" else ["-sample_fmt", "s32", "-bits_per_raw_sample", "24"]
    elif fmt == "alac":
        a += ["-c:a", "alac", "-sample_fmt", "s16p" if depth == "16" else "s32p"]
    elif fmt == "mp3":
        a += ["-c:a", "libmp3lame"] + (["-q:a", str(INT(vbr))] if vbr is not None else ["-b:a", br])
    elif fmt == "ogg":
        a += ["-c:a", "libvorbis"] + (["-q:a", str(vbr)] if vbr is not None else ["-b:a", br])
    elif fmt == "opus":
        a += ["-c:a", "libopus", "-b:a", br]
    else:
        a += ["-c:a", "aac", "-b:a", br]
    return a, FORMATS[fmt]


def out_name(c, m):
    lin = m["lineage"]
    vals = defaultdict(str, base=m["song"], stem=lin[-1] if lin else "", lineage=" - ".join(lin),
                       first=lin[0] if lin else "", model=m["model"], step=m["step"])
    try:
        name = (c["name_template"].strip() or DEFAULT_TEMPLATE).format_map(vals)
    except Exception as e:
        raise RuntimeError(f"Bad file name template: {e}")
    return safe_name(name)


def unique_path(folder, name, ext, used, policy):
    p, n = folder / f"{name}.{ext}", 2
    while p in used or (policy == "keep both" and p.exists()):
        p, n = folder / f"{name} ({n}).{ext}", n + 1
    used.add(p)
    return p


def validate(c):
    """Check every setting before any money is spent; returns the primary model list."""
    models = parse_models(c)
    for m in models:
        uvr_args(c, m, "/o", "/m", c["single_stem"])
    for i, st in enumerate(c["chain"], 1):
        if not st.get("model"):
            raise RuntimeError(f"Chain step {i} has no model.")
        uvr_args(c, st["model"], "/o", "/m", st.get("single", ""), st.get("args", ""))
    for w in split_list(c["ens_weights"]):
        try:
            float(w)
        except ValueError:
            raise RuntimeError(f"Bad ensemble weight {w!r}")
    nfft, hop = num(c, "ens_nfft", INT) or 2048, num(c, "ens_hop", INT) or 512
    if not 0 < hop < nfft:
        raise RuntimeError("STFT hop must be smaller than the STFT size.")
    for k in ("sample_start", "sample_duration"):
        num(c, k, float)
    encode_args(c)
    out_name(c, {"song": "x", "lineage": ["Vocals"], "model": "m", "step": "primary"})
    return models


class Vast:
    def __init__(self, key):
        if not key.strip():
            raise RuntimeError("Enter your Vast.ai API key first (tab 1).")
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

    def me(self):
        return self.req("GET", "/users/current/")

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
                    self.run("pkill -f audio-separator; pkill -f uvr_fx.py; pkill ffmpeg", timeout=15)
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
        self.geometry("1400x950")
        self.minsize(1100, 760)
        self.saved = load_json(CONFIG_PATH)
        self.saved.setdefault("final_format", str(self.saved.get("output_format", "flac")).lower())
        self.user_presets = load_json(PRESETS_PATH)
        self.vars, self.texts = {}, {}
        self.preset_keys, self._preset_section = {"models", "chain"}, False
        self.chain = [dict(s) for s in self.saved.get("chain", [])]
        self.q = queue.Queue()
        self.offers, self.inst_cache = {}, {}
        self.stop_event = threading.Event()
        self.job = None
        self.session_instances = set()
        self.attached = set()
        self._build()
        self.after(100, self._pump)
        self.after(800, self._startup)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- small helpers ----------
    def var(self, name, default):
        saved = self.saved.get(name, default)
        v = tk.BooleanVar(value=bool(saved)) if isinstance(default, bool) else tk.StringVar(value=str(saved))
        self.vars[name] = v
        if self._preset_section:
            self.preset_keys.add(name)
        return v

    def field(self, parent, label, name, default, kind="entry", values=(), width=26, show=None, tip="", buttons=()):
        r = getattr(parent, "_frow", 0)
        parent._frow = r + 1
        v = self.var(name, default)
        if kind == "check":
            w = ttk.Checkbutton(parent, text=label, variable=v)
            w.grid(row=r, column=0, columnspan=2, sticky="w", padx=6, pady=2)
        else:
            ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=(6, 2), pady=2)
            if kind in ("combo", "readonly"):
                w = ttk.Combobox(parent, textvariable=v, values=list(values), width=width,
                                 state="readonly" if kind == "readonly" else "normal")
            else:
                w = ttk.Entry(parent, textvariable=v, width=width, show=show or "")
            w.grid(row=r, column=1, sticky="we", padx=2, pady=2)
        col = 2
        for text, cmd in buttons:
            ttk.Button(parent, text=text, command=cmd).grid(row=r, column=col, sticky="w", padx=2)
            col += 1
        if tip:
            ttk.Label(parent, text=tip, foreground="#777").grid(row=r, column=col, sticky="w", padx=4)
        return w

    def note(self, parent, text, wrap=420):
        r = getattr(parent, "_frow", 0)
        parent._frow = r + 1
        ttk.Label(parent, text=text, foreground="#777", justify="left", wraplength=wrap).grid(
            row=r, column=0, columnspan=4, sticky="w", padx=6, pady=(2, 4))

    def text(self, parent, name, default, height=6):
        t = ScrolledText(parent, height=height, font=("Consolas", 9), wrap="none", undo=True)
        t.insert("1.0", self.saved.get(name, default))
        self.texts[name] = t
        return t

    def log(self, msg):
        self.q.put(("log", f"[{time.strftime('%H:%M:%S')}] {msg}"))

    def status(self, msg, pct=None):
        self.q.put(("status", (msg, pct)))

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
                    msg, pct = val
                    self.status_lbl.config(text=msg)
                    if pct is not None:
                        self.pbar["value"] = pct
                elif kind == "call":
                    try:
                        val()
                    except Exception as e:
                        self.logbox.insert("end", f"UI error: {e}\n")
        except queue.Empty:
            pass
        self.after(100, self._pump)

    def collect(self):
        c = {k: v.get() for k, v in self.vars.items()}
        for k, t in self.texts.items():
            c[k] = t.get("1.0", "end-1c")
        c["files"] = list(self.files.get(0, "end"))
        c["chain"] = [dict(s) for s in self.chain]
        return c

    def _save(self):
        c = self.collect()
        if not c.get("remember_key"):
            c["api_key"] = ""
        try:
            CONFIG_PATH.write_text(json.dumps(c, indent=2), encoding="utf-8")
            self.status(f"Settings saved to {CONFIG_PATH}")
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
            except StopJob:
                self.log("Stopped.")
            except RuntimeError as e:
                self.log(f"ERROR: {e}")
            except Exception as e:
                self.log(f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            finally:
                if job:
                    self.status("Idle.", 0)

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

    def _show_text(self, title, content):
        w = tk.Toplevel(self)
        w.title(title)
        w.geometry("1000x650")
        t = ScrolledText(w, font=("Consolas", 9), wrap="none")
        t.pack(fill="both", expand=True)
        t.insert("1.0", content)

    @staticmethod
    def _sortable(tree):
        def sort(col, rev):
            def key(item):
                s = item[0].replace("$", "")
                try:
                    return (0, float(s))
                except ValueError:
                    return (1, s.lower())
            rows = sorted(((tree.set(k, col), k) for k in tree.get_children("")), key=key, reverse=rev)
            for i, (_, k) in enumerate(rows):
                tree.move(k, "", i)
            tree.heading(col, command=lambda: sort(col, not rev))
        for col in tree["columns"]:
            tree.heading(col, command=lambda c=col: sort(c, False))

    def _tree(self, parent, cols, sortable=True, height=10):
        frame = ttk.Frame(parent)
        tree = ttk.Treeview(frame, columns=[c[0] for c in cols], show="headings", selectmode="browse", height=height)
        for cid, title, width in cols:
            tree.heading(cid, text=title)
            tree.column(cid, width=width, anchor="w", stretch=cid in ("loc", "gpu", "model", "args"))
        sb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        if sortable:
            self._sortable(tree)
        return frame, tree

    # ---------- layout ----------
    def _build(self):
        paned = ttk.PanedWindow(self, orient="vertical")
        paned.pack(fill="both", expand=True)
        top, bottom = ttk.Frame(paned), ttk.Frame(paned)
        paned.add(top, weight=4)
        paned.add(bottom, weight=1)

        pb = ttk.Frame(top)
        pb.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Label(pb, text="Preset:").pack(side="left")
        self.preset_box = ttk.Combobox(pb, width=58, state="readonly")
        self.preset_box.pack(side="left", padx=4)
        self._refresh_presets()
        for text, cmd in (("Load", self.on_preset_load), ("Save as…", self.on_preset_save),
                          ("Delete", self.on_preset_delete), ("Preview pipeline", self.on_preview)):
            ttk.Button(pb, text=text, command=cmd).pack(side="left", padx=2)
        ttk.Label(pb, text="built-in ★ presets only change the settings they name", foreground="#777"
                  ).pack(side="left", padx=8)

        self.nb = ttk.Notebook(top)
        self.nb.pack(fill="both", expand=True, padx=6, pady=6)
        self._tab_account()
        self._preset_section = True
        self._tab_models()
        self._tab_params()
        self._tab_chain()
        self._preset_section = False
        self._tab_run()

        bar = ttk.Frame(bottom)
        bar.pack(fill="x", padx=6)
        self.pbar = ttk.Progressbar(bar, length=260, maximum=100)
        self.pbar.pack(side="right", padx=4)
        ttk.Button(bar, text="Clear log", command=lambda: self.logbox.delete("1.0", "end")).pack(side="right")
        ttk.Button(bar, text="Save settings", command=self._save).pack(side="right", padx=4)
        self.status_lbl = ttk.Label(bar, text="Ready.", anchor="w")
        self.status_lbl.pack(side="left", fill="x", expand=True)
        self.logbox = ScrolledText(bottom, height=12, font=("Consolas", 9))
        self.logbox.pack(fill="both", expand=True, padx=6, pady=(2, 6))

    def _tab_account(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text=" 1 · Vast account & GPU ")

        acc = ttk.LabelFrame(tab, text="Account")
        acc.pack(fill="x", padx=6, pady=6)
        acc.columnconfigure(1, weight=1)
        self.key_entry = self.field(acc, "Vast.ai API key", "api_key", "", width=60, show="•", buttons=[
            ("Show/Hide", lambda: self.key_entry.config(show="" if self.key_entry.cget("show") else "•")),
            ("Test key", self.on_test_key)])
        self.acct_lbl = ttk.Label(acc, text="(get a key at cloud.vast.ai → Account → API Keys)", foreground="#777")
        self.acct_lbl.grid(row=0, column=4, sticky="w", padx=6)
        self.field(acc, f"Remember API key (saved in {CONFIG_PATH})", "remember_key", True, kind="check")
        self.field(acc, "SSH private key", "ssh_key_path", str(Path.home() / ".ssh" / "uvr_vast_rsa"), width=60,
                   buttons=[("Browse", self.on_browse_key), ("Create/check key", self.on_check_key),
                            ("Add key to Vast account", self.on_add_key)],
                   tip="auto-created + uploaded when you rent")

        filt = ttk.LabelFrame(tab, text="GPU offer filters")
        filt.pack(fill="x", padx=6, pady=(0, 6))
        cols = [ttk.Frame(filt) for _ in range(3)]
        for i, f in enumerate(cols):
            f.grid(row=0, column=i, sticky="nw", padx=8, pady=4)
        f1, f2, f3 = cols
        self.field(f1, "GPU model(s), comma sep.", "gpu_names", "RTX 4090, RTX 3090", "combo", GPUS, tip="blank = any")
        self.field(f1, "# GPUs", "num_gpus", "1", width=8)
        self.field(f1, "Min VRAM (GB)", "min_vram_gb", "12", width=8)
        self.field(f1, "Max $/hr", "max_dph", "0.60", width=8)
        self.field(f1, "Offer type", "offer_type", "on-demand", "readonly", ["on-demand", "interruptible"], width=14)
        self.field(f1, "Bid $/hr (interruptible)", "bid_price", "", width=8, tip="blank = min bid +5%")
        self.field(f2, "Min reliability (0-1)", "min_reliability", "0.95", width=8)
        self.field(f2, "Disk GB (filter + rent)", "disk_gb", "40", width=8)
        self.field(f2, "Min download Mb/s", "min_inet_down", "200", width=8)
        self.field(f2, "Min CUDA version", "min_cuda", "12.8", width=8, tip="image needs 12.8")
        self.field(f2, "Min system RAM (GB)", "min_cpu_ram_gb", "16", width=8)
        self.field(f2, "Verified machines only", "verified_only", True, kind="check")
        self.field(f3, "Order by", "order_by", "dph_total asc", "combo", ORDERS, width=24)
        self.field(f3, "Max results", "limit", "64", width=8)
        self.field(f3, "Location contains", "location_filter", "", width=14, tip="e.g. US, Germany")
        btns = ttk.Frame(f3)
        btns.grid(row=f3._frow, column=0, columnspan=3, sticky="w", pady=6)
        ttk.Button(btns, text="🔍 Search offers", command=self.on_search).pack(side="left", padx=4)
        ttk.Button(btns, text="💳 Rent selected offer", command=self.on_rent).pack(side="left", padx=4)

        frame, self.offer_tree = self._tree(tab, [
            ("id", "Offer ID", 80), ("gpu", "GPU", 150), ("n", "#", 30), ("vram", "VRAM GB", 70),
            ("dph", "$/hr", 70), ("rel", "Reliab.", 60), ("cuda", "CUDA", 50), ("down", "Down Mb/s", 80),
            ("up", "Up Mb/s", 70), ("disk", "Disk GB", 65), ("ram", "RAM GB", 60), ("dlperf", "DLPerf", 60),
            ("loc", "Location", 170), ("ver", "Verified", 60)])
        frame.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self.offer_tree.bind("<<TreeviewSelect>>", lambda e: self._offer_selected())

    def _tab_models(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text=" 2 · Models & ensemble ")
        tab.columnconfigure(0, weight=3)
        tab.columnconfigure(1, weight=2)
        tab.rowconfigure(0, weight=1)

        left = ttk.LabelFrame(tab, text="Primary models - one per line (# = comment), each runs on every input")
        left.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.text(left, "models", ROFORMER + "\n", height=14).pack(fill="both", expand=True, padx=4, pady=4)
        add = ttk.Frame(left)
        add.pack(fill="x", padx=4, pady=(0, 4))
        self.model_pick = ttk.Combobox(add, values=MODEL_PRESETS, width=50)
        self.model_pick.pack(side="left", fill="x", expand=True)
        ttk.Button(add, text="Add", command=self.on_add_model).pack(side="left", padx=3)
        ttk.Button(left, text="List every model available on the server (needs a running instance)",
                   command=self.on_list_models).pack(fill="x", padx=4, pady=(0, 4))
        ttk.Label(left, foreground="#777", justify="left", text=(
            ".ckpt = Roformer / MDX23C    .onnx = MDX-Net    .pth = VR Arch    .yaml = Demucs\n"
            "Models download once per instance and are cached in <remote dir>/models.")
        ).pack(anchor="w", padx=4, pady=(0, 6))

        right = ttk.Frame(tab)
        right.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        p = ttk.LabelFrame(right, text="Primary pass")
        p.pack(fill="x", pady=(0, 6))
        self.field(p, "Mode", "primary_mode", "separate", "readonly", ["separate", "ensemble"], width=12,
                   tip="ensemble needs 2+ models")
        self.field(p, "Single stem (primary pass)", "single_stem", "", "combo", STEMS, width=16, tip="blank = all")
        self.field(p, "Download primary results", "dl_primary", True, kind="check")

        e = ttk.LabelFrame(right, text="Ensemble - combines the same stem across all primary models")
        e.pack(fill="x", pady=(0, 6))
        self.field(e, "Algorithm", "ens_algo", "avg_wave", "readonly", ENS_ALGOS, width=14)
        self.field(e, "Model weights (avg only)", "ens_weights", "", width=14, tip="e.g. 2, 1, 1")
        self.field(e, "STFT size (spec algos)", "ens_nfft", "2048", "combo", ["1024", "2048", "4096", "8192"], width=8)
        self.field(e, "STFT hop", "ens_hop", "512", "combo", ["256", "512", "1024", "2048"], width=8)
        self.field(e, "Also download each model's own stems", "ens_keep_individual", False, kind="check")
        self.note(e, "avg = balanced · max_spec = fullest, most bleed (UVR 'Max Spec') · min_spec = cleanest, "
                     "thinnest (UVR 'Min Spec') · median = drops the outlier model (3+ models) · "
                     "_wave = time domain, _spec = per frequency bin")

        iv = ttk.LabelFrame(right, text="Inversion - rebuild the counterpart as  mix − stem")
        iv.pack(fill="x", pady=(0, 6))
        self.field(iv, "Invert stems (comma, wildcards)", "invert_stems", "", width=14, tip='Vocals → "No Vocals"')
        self.field(iv, "Method", "invert_method", "wave", "readonly", ["wave", "spec"], width=8,
                   tip="spec = magnitude subtraction")

        s = ttk.LabelFrame(right, text="Input slicing - try settings on a short piece first")
        s.pack(fill="x")
        self.field(s, "Start at (seconds)", "sample_start", "", width=8)
        self.field(s, "Duration (seconds)", "sample_duration", "", width=8, tip="blank = whole file")

    def _tab_params(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text=" 3 · Model parameters ")
        for i in range(3):
            tab.columnconfigure(i, weight=1)

        g = ttk.LabelFrame(tab, text="All models")
        g.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.field(g, "Processing sample rate", "sample_rate", "44100", "combo", ["44100", "48000", "96000"], width=10)
        self.field(g, "Normalization (max peak)", "normalization", "0.9", width=10)
        self.field(g, "Amplification (min peak)", "amplification", "0.0", width=10)
        self.field(g, "Log level", "log_level", "info", "readonly", ["debug", "info", "warning", "error"], width=10)
        self.field(g, "Invert spectrogram for secondary stem", "invert_spect", False, kind="check")
        self.field(g, "Write stems with soundfile", "use_soundfile", False, kind="check")
        self.field(g, "Use autocast (faster fp16 on GPU)", "use_autocast", False, kind="check")

        mdx = ttk.LabelFrame(tab, text="MDX-Net (.onnx)")
        mdx.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        self.field(mdx, "Segment size", "mdx_segment_size", "256", "combo", ["32", "64", "128", "256", "512", "1024"], width=8)
        self.field(mdx, "Overlap (0.001-0.999)", "mdx_overlap", "0.25", width=8)
        self.field(mdx, "Batch size", "mdx_batch_size", "1", width=8)
        self.field(mdx, "Hop length", "mdx_hop_length", "1024", width=8)
        self.field(mdx, "Enable denoise", "mdx_enable_denoise", False, kind="check")

        mx = ttk.LabelFrame(tab, text="MDXC / Roformer / MDX23C (.ckpt)")
        mx.grid(row=0, column=2, sticky="nsew", padx=6, pady=6)
        self.field(mx, "Segment size", "mdxc_segment_size", "256", width=8)
        self.field(mx, "Overlap (2-50)", "mdxc_overlap", "8", width=8, tip="higher = better, slower")
        self.field(mx, "Batch size", "mdxc_batch_size", "1", width=8)
        self.field(mx, "Pitch shift (semitones)", "mdxc_pitch_shift", "0", width=8)
        self.field(mx, "Override model segment size", "mdxc_override_model_segment_size", False, kind="check")

        vr = ttk.LabelFrame(tab, text="VR Arch (.pth)")
        vr.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        self.field(vr, "Batch size", "vr_batch_size", "1", width=8)
        self.field(vr, "Window size", "vr_window_size", "512", "combo", ["320", "512", "1024"], width=8,
                   tip="lower = better, slower")
        self.field(vr, "Aggression (-100..100)", "vr_aggression", "5", width=8)
        self.field(vr, "Post-process threshold", "vr_post_process_threshold", "0.2", width=8, tip="0.1-0.3")
        self.field(vr, "TTA (better, slower)", "vr_enable_tta", False, kind="check")
        self.field(vr, "High-end process (mirror missing freqs)", "vr_high_end_process", False, kind="check")
        self.field(vr, "Post-process (remove vocal artifacts)", "vr_enable_post_process", False, kind="check")

        dm = ttk.LabelFrame(tab, text="Demucs (.yaml)")
        dm.grid(row=1, column=1, sticky="nsew", padx=6, pady=6)
        self.field(dm, "Segment size", "demucs_segment_size", "Default", "combo",
                   ["Default", "1", "5", "10", "15", "20", "25", "30", "35", "40", "45", "50", "60", "70", "80", "90", "100"],
                   width=8)
        self.field(dm, "Shifts", "demucs_shifts", "2", width=8, tip="more = better, slower")
        self.field(dm, "Overlap (0.001-0.999)", "demucs_overlap", "0.25", width=8)
        self.field(dm, "Segment-wise processing", "demucs_segments_enabled", True, kind="check")

        ex = ttk.LabelFrame(tab, text="Anything else")
        ex.grid(row=1, column=2, sticky="nsew", padx=6, pady=6)
        ex.columnconfigure(1, weight=1)
        self.field(ex, "Extra audio-separator args", "extra_args", "", width=30)
        self.note(ex, "Appended to every audio-separator call (chain steps can add their own). "
                      "Later flags win, so this overrides the fields above.", wrap=360)

    def _tab_chain(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text=" 4 · Chain & output ")

        ch = ttk.LabelFrame(tab, text="Post-processing chain - each step runs one more model on stems from an earlier stage")
        ch.pack(fill="both", expand=True, padx=6, pady=6)
        frame, self.chain_tree = self._tree(ch, [
            ("n", "#", 30), ("src", "From", 80), ("stems", "Stems", 150), ("model", "Model", 330),
            ("single", "Single stem", 90), ("args", "Extra args", 200), ("dl", "Download", 70)], sortable=False, height=6)
        frame.pack(fill="both", expand=True, padx=4, pady=4)
        self.chain_tree.bind("<<TreeviewSelect>>", lambda e: self._chain_selected())

        ed = ttk.Frame(ch)
        ed.pack(fill="x", padx=4)
        self.ce = {"src": tk.StringVar(value="primary"), "stems": tk.StringVar(value="Vocals"),
                   "model": tk.StringVar(value="UVR-DeEcho-DeReverb.pth"), "single": tk.StringVar(),
                   "args": tk.StringVar(), "dl": tk.BooleanVar(value=True)}
        self.chain_model = ttk.Combobox(ed, textvariable=self.ce["model"], values=MODEL_PRESETS, width=46)
        widgets = [
            ("From", ttk.Combobox(ed, textvariable=self.ce["src"], width=9,
                                  values=["primary", "previous"] + [f"step {i}" for i in range(1, 10)])),
            ("Stems (comma, wildcards)", ttk.Combobox(ed, textvariable=self.ce["stems"], values=STEM_FILTERS, width=18)),
            ("Model", self.chain_model),
            ("Single stem", ttk.Combobox(ed, textvariable=self.ce["single"], values=STEMS, width=12)),
            ("Extra args (this step)", ttk.Entry(ed, textvariable=self.ce["args"], width=28))]
        for i, (label, w) in enumerate(widgets):
            ttk.Label(ed, text=label).grid(row=0, column=i, sticky="w", padx=3)
            w.grid(row=1, column=i, sticky="we", padx=3)
        ttk.Checkbutton(ed, text="Download", variable=self.ce["dl"]).grid(row=1, column=len(widgets), padx=6)
        bb = ttk.Frame(ch)
        bb.pack(fill="x", padx=4, pady=4)
        for text, cmd in (("+ Add step", self.on_chain_add), ("Update selected", self.on_chain_update),
                          ("Remove", self.on_chain_remove), ("▲", lambda: self.on_chain_move(-1)),
                          ("▼", lambda: self.on_chain_move(1)), ("Clear chain", self.on_chain_clear)):
            ttk.Button(bb, text=text, command=cmd).pack(side="left", padx=2)
        ttk.Label(bb, foreground="#777", text="From: primary = primary/ensemble results · previous = the step above · "
                                              "step N = that step's outputs").pack(side="left", padx=8)
        self._refresh_chain()

        out = ttk.Frame(tab)
        out.pack(fill="x", padx=6, pady=(0, 6))
        for i in range(3):
            out.columnconfigure(i, weight=1)
        enc = ttk.LabelFrame(out, text="Final encode (ffmpeg on the GPU box)")
        enc.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.field(enc, "Format", "final_format", "flac", "readonly", list(FORMATS), width=10)
        self.field(enc, "Bit depth (lossless)", "bit_depth", "24", "readonly", ["16", "24", "32 float"], width=10,
                   tip="FLAC/ALAC max 24")
        self.field(enc, "Bitrate (lossy)", "lossy_bitrate", "320k", "combo",
                   ["320k", "256k", "192k", "160k", "128k", "96k"], width=10)
        self.field(enc, "VBR quality (mp3/ogg)", "lossy_vbr", "", width=10, tip="mp3 0-9, ogg 0-10")
        self.field(enc, "FLAC compression (0-12)", "flac_level", "5", width=10)
        self.field(enc, "Sample rate", "final_sr", "keep", "combo", ["keep", "44100", "48000", "88200", "96000"], width=10)
        self.field(enc, "Channels", "final_channels", "keep", "readonly", ["keep", "stereo", "mono"], width=10)
        self.field(enc, "Dither to 16-bit", "dither", "triangular", "readonly",
                   ["none", "triangular", "triangular_hp", "lipshitz", "shibata", "low_shibata", "high_shibata"], width=12)
        self.field(enc, "High-quality resampler (soxr)", "hq_resample", True, kind="check")

        lv = ttk.LabelFrame(out, text="Level & filters")
        lv.grid(row=0, column=1, sticky="nsew", padx=6)
        self.field(lv, "Gain (dB)", "gain_db", "", width=10)
        self.field(lv, "Loudness target (LUFS)", "loudnorm_lufs", "", width=10, tip="blank = off, e.g. -14")
        self.field(lv, "True peak (dBTP)", "loudnorm_tp", "-1.0", width=10)
        self.field(lv, "Loudness range (LRA)", "loudnorm_lra", "11", width=10)
        self.field(lv, "Extra ffmpeg filters (-af)", "extra_af", "", width=24)
        self.note(lv, "e.g. highpass=f=30,afade=t=in:d=0.3 - applied after gain/loudness, before resampling.",
                  wrap=320)

        fs = ttk.LabelFrame(out, text="Files")
        fs.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        self.field(fs, "Name template", "name_template", DEFAULT_TEMPLATE, width=28)
        self.note(fs, "{base} song name · {stem} last stem · {lineage} e.g. Vocals - No Reverb · {first} "
                      "primary stem · {model} · {step} primary / ensemble / inverted / stepN", wrap=330)
        self.field(fs, "Only download stems", "stem_filter", "", width=20, tip="blank = all")
        self.field(fs, "If file exists", "if_exists", "overwrite", "readonly", ["overwrite", "keep both"], width=12)
        self.field(fs, "Subfolder per song", "subfolder_per_song", False, kind="check")

    def _tab_run(self):
        tab = ttk.Frame(self.nb)
        self.tab_run = tab
        self.nb.add(tab, text=" 5 · Run ")
        top = ttk.LabelFrame(tab, text="Machine & output")
        top.pack(fill="x", padx=6, pady=6)
        top.columnconfigure(1, weight=1)
        self.field(top, "Instance ID", "instance_id", "", width=14, buttons=[
            ("Find my instances", lambda: self.on_find_instances(False)), ("Setup log", self.on_setup_log),
            ("Copy SSH cmd", self.on_copy_ssh), ("🗑 Destroy now", self.on_destroy_current)])
        self.inst_lbl = ttk.Label(top, text="", foreground="#777")
        self.inst_lbl.grid(row=0, column=6, sticky="w", padx=6)
        self.vars["instance_id"].trace_add("write", lambda *a: self._update_inst_label())
        self.field(top, "Local output folder", "local_out", str(Path.home() / "Music" / "UVR_output"), width=60,
                   buttons=[("Browse", self.on_browse_out), ("Open", self.on_open_out)])
        self.field(top, "Remote work dir", "remote_dir", "/root/uvr", width=30)
        self.field(top, "Timeout for boot + setup (min)", "ready_timeout_min", "30", width=8)
        self.field(top, "Destroy instance when the job finishes (also on error; not on Stop)", "auto_destroy",
                   True, kind="check")
        self.field(top, "Keep job files on the instance", "keep_remote", False, kind="check")

        fl = ttk.LabelFrame(tab, text="Input audio files")
        fl.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        side = ttk.Frame(fl)
        side.pack(side="right", fill="y", padx=4, pady=4)
        for text, cmd in (("Add files", self.on_add_files), ("Add folder", self.on_add_folder),
                          ("Remove selected", self.on_remove_files), ("Clear", lambda: self.files.delete(0, "end"))):
            ttk.Button(side, text=text, command=cmd).pack(fill="x", pady=2)
        self.files = tk.Listbox(fl, selectmode="extended", font=("Consolas", 9))
        self.files.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        for f in self.saved.get("files", []):
            self.files.insert("end", f)

        act = ttk.Frame(tab)
        act.pack(fill="x", padx=6, pady=(0, 8))
        ttk.Button(act, text="⚡ ONE CLICK: rent (selected or cheapest) → separate → download → destroy",
                   command=self.on_oneclick).pack(side="left", padx=3)
        ttk.Button(act, text="▶ Process on instance", command=self.on_process).pack(side="left", padx=3)
        ttk.Button(act, text="Wait until ready", command=self.on_wait_ready).pack(side="left", padx=3)
        ttk.Button(act, text="■ Stop", command=self.on_stop).pack(side="left", padx=3)

    # ---------- presets ----------
    def _refresh_presets(self, select=None):
        names = list(BUILTIN_PRESETS) + sorted(self.user_presets)
        self.preset_box["values"] = names
        if select:
            self.preset_box.set(select)

    def on_preset_load(self):
        name = self.preset_box.get()
        preset = BUILTIN_PRESETS.get(name) or self.user_presets.get(name)
        if not preset:
            return
        for k, v in preset.items():
            if k == "chain":
                self.chain = [dict(s) for s in v]
                self._refresh_chain()
            elif k in self.texts:
                self.texts[k].delete("1.0", "end")
                self.texts[k].insert("1.0", v)
            elif k in self.vars:
                self.vars[k].set(bool(v) if isinstance(self.vars[k], tk.BooleanVar) else str(v))
        self.log(f"Loaded preset: {name}")

    def on_preset_save(self):
        name = simpledialog.askstring(APP, "Preset name:", parent=self)
        if not name or not name.strip():
            return
        name = name.strip().lstrip("★ ")
        c = self.collect()
        self.user_presets[name] = {k: c[k] for k in sorted(self.preset_keys) if k in c}
        self._write_presets()
        self._refresh_presets(name)
        self.log(f"Saved preset: {name}")

    def on_preset_delete(self):
        name = self.preset_box.get()
        if name in self.user_presets and messagebox.askyesno(APP, f"Delete preset '{name}'?"):
            del self.user_presets[name]
            self._write_presets()
            self._refresh_presets("")

    def _write_presets(self):
        try:
            PRESETS_PATH.write_text(json.dumps(self.user_presets, indent=2), encoding="utf-8")
        except OSError as e:
            messagebox.showerror(APP, f"Could not save presets: {e}")

    def on_preview(self):
        try:
            c = self.collect()
            models = validate(c) or ["<model>"]
            rd = c["remote_dir"].strip().rstrip("/") or "/root/uvr"
            sr = num(c, "sample_rate", INT) or 44100
            start, dur = num(c, "sample_start", float), num(c, "sample_duration", float)
            lines = [f"inputs: decode → {sr} Hz stereo float WAV"
                     + (f", slice {start or 0}s + {dur}s" if dur else f", from {start}s" if start else "")]
            for m in models:
                lines.append("primary: audio-separator " + " ".join(
                    q(x) for x in uvr_args(c, m, f"{rd}/job/p", f"{rd}/models", c["single_stem"])) + " <inputs>")
            if c["primary_mode"] == "ensemble" and len(models) > 1:
                w = c["ens_weights"].strip()
                lines.append(f"ensemble: {c['ens_algo']} over {len(models)} models per stem"
                             + (f", weights {w}" if w else "")
                             + (f", STFT {c['ens_nfft']}/{c['ens_hop']}" if "spec" in c["ens_algo"] else ""))
            if split_list(c["invert_stems"]):
                lines.append(f"invert ({c['invert_method']}): mix − [{c['invert_stems']}] → 'No <stem>'")
            for i, st in enumerate(c["chain"], 1):
                lines.append(f"step {i} [{st.get('stems') or '*'} from {st.get('src')}"
                             f"{', download' if st.get('dl') else ''}]: audio-separator " + " ".join(
                                 q(x) for x in uvr_args(c, st["model"], f"{rd}/job/c{i}", f"{rd}/models",
                                                        st.get("single", ""), st.get("args", ""))))
            enc, ext = encode_args(c)
            lines.append("encode: ffmpeg -i <stem.wav> " + " ".join(q(x) for x in enc) + f" <name>.{ext}")
            lines.append("download: " + ("primary results" if c["dl_primary"] else "no primary results")
                         + (", individual model stems" if c["ens_keep_individual"] else "")
                         + (f", only stems [{c['stem_filter']}]" if c["stem_filter"].strip() else ""))
            lines.append("example file: " + out_name(c, {"song": "My Song", "lineage": ["Vocals", "No Reverb"],
                                                           "model": "UVR-DeNoise", "step": "step2"}) + f".{ext}")
            self.log("Pipeline preview:\n  " + "\n  ".join(lines))
        except RuntimeError as e:
            messagebox.showerror(APP, str(e))

    # ---------- chain editor ----------
    def _refresh_chain(self, select=None):
        t = self.chain_tree
        t.delete(*t.get_children())
        for i, st in enumerate(self.chain, 1):
            t.insert("", "end", iid=str(i - 1), values=(
                i, st.get("src", ""), st.get("stems", "") or "*", st.get("model", ""), st.get("single", ""),
                st.get("args", ""), "✓" if st.get("dl") else ""))
        if select is not None and 0 <= select < len(self.chain):
            t.selection_set(str(select))

    def _chain_entry(self):
        st = {k: (v.get().strip() if isinstance(v, tk.StringVar) else v.get()) for k, v in self.ce.items()}
        if not st["model"]:
            messagebox.showinfo(APP, "Pick a model for the step.")
            return None
        return st

    def _chain_index(self):
        sel = self.chain_tree.selection()
        return int(sel[0]) if sel else None

    def _chain_selected(self):
        i = self._chain_index()
        if i is not None:
            for k, v in self.chain[i].items():
                if k in self.ce:
                    self.ce[k].set(v)

    def on_chain_add(self):
        st = self._chain_entry()
        if st:
            self.chain.append(st)
            self._refresh_chain(len(self.chain) - 1)

    def on_chain_update(self):
        i, st = self._chain_index(), self._chain_entry()
        if i is not None and st:
            self.chain[i] = st
            self._refresh_chain(i)

    def on_chain_remove(self):
        i = self._chain_index()
        if i is not None:
            del self.chain[i]
            self._refresh_chain(min(i, len(self.chain) - 1))

    def on_chain_move(self, d):
        i = self._chain_index()
        if i is not None and 0 <= i + d < len(self.chain):
            self.chain[i], self.chain[i + d] = self.chain[i + d], self.chain[i]
            self._refresh_chain(i + d)

    def on_chain_clear(self):
        if self.chain and messagebox.askyesno(APP, "Remove all chain steps?"):
            self.chain = []
            self._refresh_chain()

    # ---------- account / keys ----------
    def on_test_key(self):
        c = self.collect()

        def work():
            me = Vast(c["api_key"]).me()
            who = me.get("username") or me.get("email") or me.get("id")
            credit = me.get("credit", me.get("balance", "?"))
            txt = f"✔ {who} - credit ${float(credit):.2f}" if isinstance(credit, (int, float)) else f"✔ {who}"
            self.ui(self.acct_lbl.config, {"text": txt, "foreground": "green"})
            self.log(f"API key OK: {txt}")
        self.bg(work)

    def on_browse_key(self):
        p = filedialog.askopenfilename(title="SSH private key", initialdir=str(Path.home() / ".ssh"))
        if p:
            self.vars["ssh_key_path"].set(p)

    def on_check_key(self):
        c = self.collect()
        self.bg(lambda: self.log("SSH public key: " + load_or_create_key(c["ssh_key_path"], self.log)))

    def on_add_key(self):
        c = self.collect()
        self.bg(lambda: self._ensure_account_key(Vast(c["api_key"]), load_or_create_key(c["ssh_key_path"], self.log), True))

    def _ensure_account_key(self, vast, pub, verbose=False):
        body = pub.split()[1]
        try:
            if body in json.dumps(vast.req("GET", "/ssh/")):
                if verbose:
                    self.log("SSH key is already on your Vast account.")
                return
        except Exception as e:
            self.log(f"(could not list account SSH keys: {e})")
        try:
            vast.req("POST", "/ssh/", json={"ssh_key": pub})
            self.log("Added SSH public key to your Vast account.")
        except Exception as e:
            self.log(f"(could not add SSH key to account: {e})")

    # ---------- offers ----------
    def on_search(self):
        c = self.collect()

        def work():
            self.status("Searching offers...")
            offers = self._search(Vast(c["api_key"]), c)
            self.ui(self._fill_offers, offers)
            self.log(f"Found {len(offers)} offers.")
            self.status(f"{len(offers)} offers.")
        self.bg(work)

    def _search(self, vast, c):
        offers = vast.search(offer_query(c))
        loc = c["location_filter"].strip().lower()
        if loc:
            offers = [o for o in offers if loc in str(o.get("geolocation") or "").lower()]
        return offers

    def _fill_offers(self, offers):
        self.offer_tree.delete(*self.offer_tree.get_children())
        self.offers = {}
        for o in offers:
            oid = str(o["id"])
            self.offers[oid] = o
            rel = o.get("reliability2", o.get("reliability")) or 0
            self.offer_tree.insert("", "end", iid=oid, values=(
                oid, o.get("gpu_name", "?"), o.get("num_gpus", ""), f"{(o.get('gpu_ram') or 0) / 1024:.0f}",
                f"${float(o.get('dph_total') or 0):.3f}", f"{float(rel):.3f}", o.get("cuda_max_good", ""),
                f"{float(o.get('inet_down') or 0):.0f}", f"{float(o.get('inet_up') or 0):.0f}",
                f"{float(o.get('disk_space') or 0):.0f}", f"{(o.get('cpu_ram') or 0) / 1024:.0f}",
                f"{float(o.get('dlperf') or 0):.0f}", o.get("geolocation") or "",
                "✓" if o.get("verification") == "verified" or o.get("verified") else ""))

    def _selected_offer(self):
        sel = self.offer_tree.selection()
        return self.offers.get(sel[0]) if sel else None

    def _drop_offer(self, oid):
        """A rented offer is no longer available - keep it from being picked again."""
        self.offers.pop(oid, None)
        if self.offer_tree.exists(oid):
            self.offer_tree.delete(oid)

    def _rent_any(self, vast, c, offer=None):
        """Rent `offer`, or the cheapest match; if an offer was taken meanwhile, move on to the next one."""
        tried, pool = set(), [offer] if offer else []
        for _ in range(6):
            if not pool:
                pool = [o for o in self._search(vast, c) if o["id"] not in tried]
                if not pool:
                    raise RuntimeError("No offers match your filters - loosen them in tab 1.")
            o = pool.pop(0)
            tried.add(o["id"])
            try:
                return self._rent(vast, c, o)
            except RuntimeError as e:
                if not _offer_gone(e):
                    raise
                self.log(f"Offer #{o['id']} is no longer available - trying the next cheapest match...")
                self.ui(self._drop_offer, str(o["id"]))
        raise RuntimeError("Could not rent any of the matching offers - search again.")

    def _offer_selected(self):
        o = self._selected_offer()
        if o:
            self.status("Selected " + offer_desc(o))

    def on_rent(self):
        o = self._selected_offer()
        if not o:
            messagebox.showinfo(APP, "Search and select an offer first.")
            return
        if not messagebox.askyesno(APP, f"Rent this machine?\n\n{offer_desc(o)}\n\nBilling runs until you destroy it."):
            return
        self.bg(self._rent_and_prepare, self.collect(), o, job=True)

    def _rent_and_prepare(self, c, offer):
        vast = Vast(c["api_key"])
        pub = load_or_create_key(c["ssh_key_path"], self.log)
        self._ensure_account_key(vast, pub)
        try:
            iid = self._rent(vast, c, offer)
        except RuntimeError as e:
            if not _offer_gone(e):
                raise
            self.ui(self._fill_offers, self._search(vast, c))
            raise RuntimeError(f"Offer #{offer['id']} was just taken - the list is refreshed, pick another one.")
        r, _ = self._connect_ready(vast, c, iid, pub)
        r.close()
        self.log(f"Instance #{iid} is ready. Add files in the Run tab and press Process.")
        self.ui(self.nb.select, self.tab_run)
        self.ui(self.bell)

    def _rent(self, vast, c, offer):
        body = {"client_id": "me", "image": IMAGE, "disk": num(c, "disk_gb", float, True), "label": UVR_LABEL,
                "runtype": RUNTYPE, "onstart": ONSTART, "env": {}}
        if c["offer_type"] == "interruptible":
            body["price"] = num(c, "bid_price", float) or round(
                float(offer.get("min_bid") or offer.get("dph_total") or 0) * 1.05, 4)
        self.log("Renting " + offer_desc(offer) + " ...")
        res = vast.create(offer["id"], body)
        iid = res.get("new_contract")
        if not iid or res.get("success") is False:
            raise RuntimeError(f"Rent failed: {res}")
        iid = int(iid)
        self.session_instances.add(iid)
        self.ui(self._drop_offer, str(offer["id"]))
        self.ui(self.vars["instance_id"].set, str(iid))
        self.ui(self.on_find_instances, True)
        self.log(f"Rented instance #{iid}. Billing has started.")
        return iid

    # ---------- instances ----------
    def _startup(self):
        if self.vars["api_key"].get().strip():
            self.on_find_instances(True)

    def on_find_instances(self, quiet=False):
        c = self.collect()
        if not c["api_key"].strip():
            if not quiet:
                messagebox.showinfo(APP, "Enter your API key in tab 1 first.")
            return
        self.bg(lambda: self.ui(self._got_instances, Vast(c["api_key"]).instances(), quiet))

    def _uvr_ids(self):
        return {int(k) for k, i in self.inst_cache.items() if i.get("label") == UVR_LABEL}

    def _got_instances(self, insts, quiet):
        self.inst_cache = {str(i.get("id")): i for i in insts}
        self.session_instances &= {int(k) for k in self.inst_cache}
        mine = [i for i in insts if i.get("label") == UVR_LABEL]
        others = [i for i in insts if i.get("label") != UVR_LABEL]
        cur = self.vars["instance_id"].get().strip()
        for i in mine:
            self.log(f"UVR instance #{i.get('id')}: {i.get('actual_status') or '?'} · {i.get('num_gpus', 1)}x "
                     f"{i.get('gpu_name', '?')} · ${float(i.get('dph_total') or 0):.3f}/hr - billing until destroyed")
        if others:
            self.log("Also billing on your account, not managed here: " + ", ".join(
                f"#{i.get('id')} '{i.get('label') or '-'}' ${float(i.get('dph_total') or 0):.3f}/hr" for i in others))
        if mine and cur not in self.inst_cache:
            pick = next((i for i in mine if i.get("actual_status") == "running"), mine[0])
            self.vars["instance_id"].set(str(pick.get("id")))
        elif not mine and cur not in self.inst_cache:
            self.vars["instance_id"].set("")
        if not insts and not quiet:
            self.log("No instances on your account.")
        self._update_inst_label()

    def _update_inst_label(self):
        i = self.inst_cache.get(self.vars["instance_id"].get().strip())
        self.inst_lbl.config(text=f"'{i.get('label') or '-'}' · {i.get('actual_status') or '?'} · {i.get('num_gpus', 1)}x "
                                  f"{i.get('gpu_name', '?')} · ${float(i.get('dph_total') or 0):.3f}/hr" if i else "")

    def on_destroy_current(self):
        c = self.collect()
        try:
            iid = self._need_iid(c)
        except RuntimeError as e:
            messagebox.showinfo(APP, str(e))
            return
        label = (self.inst_cache.get(str(iid)) or {}).get("label")
        warn = f"\n\n⚠ It is labelled '{label}' - it was NOT rented by this app." if label and label != UVR_LABEL else ""
        if not messagebox.askyesno(APP, f"Destroy instance #{iid}? Everything on it is deleted and billing stops.{warn}"):
            return

        def work():
            self._destroy(Vast(c["api_key"]), iid)
            self.ui(self.vars["instance_id"].set, "")
            self.ui(self.on_find_instances, True)
        self.bg(work)

    def _destroy(self, vast, iid):
        vast.destroy(iid)
        self.session_instances.discard(int(iid))
        self.log(f"Destroyed instance #{iid}. Billing stopped.")

    def on_setup_log(self):
        c = self.collect()

        def work():
            iid = self._need_iid(c)
            inst = Vast(c["api_key"]).instance(iid)
            if not inst:
                raise RuntimeError(f"Instance #{iid} not found.")
            r = self._ssh(inst, c, time.time() + 60)
            try:
                _, out = r.run(f"tail -n 80 {SETUP_LOG}; echo; ls -l {READY_MARKER} 2>&1", timeout=30)
            finally:
                r.close()
            self.ui(self._show_text, f"Setup log - instance #{iid}", out)
        self.bg(work)

    def on_copy_ssh(self):
        c = self.collect()

        def work():
            iid = self._need_iid(c)
            tg = ssh_targets(Vast(c["api_key"]).instance(iid) or {})
            if not tg:
                raise RuntimeError("No SSH endpoint yet - wait until the instance is running.")
            cmd = f'ssh -i "{Path(c["ssh_key_path"]).expanduser()}" -p {tg[0][1]} root@{tg[0][0]}'
            self.ui(self.clipboard_clear)
            self.ui(self.clipboard_append, cmd)
            self.log("Copied: " + cmd)
        self.bg(work)

    # ---------- SSH / readiness ----------
    def _need_iid(self, c):
        try:
            return int(str(c["instance_id"]).strip())
        except ValueError:
            raise RuntimeError("No instance - rent one in tab 1 or press 'Find my instances'.")

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

    def _ssh(self, inst, c, deadline):
        targets = ssh_targets(inst)
        if not targets:
            raise RuntimeError("Instance has no SSH endpoint yet.")
        key = str(Path(c["ssh_key_path"]).expanduser())
        last = None
        while True:
            for host, port in targets:
                try:
                    self.status(f"Connecting SSH {host}:{port} ...")
                    r = Remote(host, port, key)
                    self.log(f"SSH connected: root@{host} -p {port}")
                    return r
                except Exception as e:
                    last = e
            if time.time() > deadline:
                raise RuntimeError(f"SSH failed: {type(last).__name__}: {last}")
            self.status(f"SSH not up yet ({type(last).__name__}); retrying...")
            self.nap(10)

    def _connect_ready(self, vast, c, iid, pub):
        deadline = time.time() + (num(c, "ready_timeout_min", float) or 30) * 60
        inst = self._wait_running(vast, iid, deadline)
        if iid not in self.attached:
            try:
                vast.req("POST", f"/instances/{iid}/ssh/", json={"ssh_key": pub})
            except Exception as e:
                self.log(f"(attach SSH key to instance: {e})")
            self.attached.add(iid)
        r = self._ssh(inst, c, deadline)
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
                        raise RuntimeError("Remote setup failed again - see the errors above (Run tab → Setup log).")
                    self._repair_setup(r)
                    repaired = True
                    continue
                if tail.strip() and tail != last:
                    self.log("setup: " + tail.strip().replace("\n", " | ")[-300:])
                    last = tail
                if time.time() > deadline:
                    raise RuntimeError(f"Setup not finished before the timeout. Check {SETUP_LOG} (Run tab → Setup log).")
                self.status("Waiting for remote setup (installing ffmpeg, compiler + audio-separator)...")
                self.nap(8)
        except BaseException:
            r.close()
            raise
        return r, float(inst.get("dph_total") or 0)

    def _repair_setup(self, r):
        """Boot-time setup failed or never ran (e.g. machines rented by an older version): run it over SSH."""
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

    # ---------- models ----------
    def on_add_model(self):
        m = self.model_pick.get().strip()
        if m:
            t = self.texts["models"]
            cur = t.get("1.0", "end-1c")
            t.insert("end", ("" if not cur or cur.endswith("\n") else "\n") + m + "\n")

    def on_list_models(self):
        c = self.collect()

        def work():
            vast = Vast(c["api_key"])
            r, _ = self._connect_ready(vast, c, self._need_iid(c), load_or_create_key(c["ssh_key_path"], self.log))
            try:
                self.status("Listing models on server...")
                _, out = r.run(PATH_PREFIX + "audio-separator -l", timeout=300)
            finally:
                r.close()
            names = sorted(set(re.findall(r"[\w.\-]+\.(?:ckpt|onnx|pth|yaml)\b", out))) or MODEL_PRESETS
            self.ui(self.model_pick.config, {"values": names})
            self.ui(self.chain_model.config, {"values": names})
            self.ui(self._show_text, "audio-separator -l", out)
            self.log(f"{len(names)} model files available - pick them from the model dropdowns.")
        self.bg(work, job=True)

    # ---------- Run tab ----------
    def on_browse_out(self):
        p = filedialog.askdirectory(title="Output folder")
        if p:
            self.vars["local_out"].set(p)

    def on_open_out(self):
        p = Path(self.vars["local_out"].get()).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(p)
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(p)])

    def _add_paths(self, paths):
        have = set(self.files.get(0, "end"))
        for p in paths:
            if p not in have:
                self.files.insert("end", p)
                have.add(p)

    def on_add_files(self):
        exts = " ".join("*" + e for e in sorted(AUDIO_EXT))
        self._add_paths(filedialog.askopenfilenames(title="Audio files", filetypes=[("Audio", exts), ("All", "*.*")]))

    def on_add_folder(self):
        d = filedialog.askdirectory(title="Folder with audio")
        if d:
            self._add_paths(str(p) for p in sorted(Path(d).rglob("*")) if p.suffix.lower() in AUDIO_EXT)

    def on_remove_files(self):
        for i in reversed(self.files.curselection()):
            self.files.delete(i)

    def on_stop(self):
        self.stop_event.set()
        self.status("Stopping...")

    def on_wait_ready(self):
        c = self.collect()

        def work():
            vast = Vast(c["api_key"])
            r, dph = self._connect_ready(vast, c, self._need_iid(c), load_or_create_key(c["ssh_key_path"], self.log))
            r.close()
            self.log(f"Instance ready (${dph:.3f}/hr).")
        self.bg(work, job=True)

    def on_process(self):
        c = self.collect()
        try:
            iid = self._need_iid(c)
        except RuntimeError as e:
            messagebox.showerror(APP, str(e))
            return
        self.bg(self._pipeline, c, iid, None, job=True)

    def on_oneclick(self):
        c = self.collect()
        o = self._selected_offer()
        what = offer_desc(o) if o else f"the cheapest offer matching your filters (max ${c['max_dph'] or '∞'}/hr)"
        n_files, n_models, n_steps = len(c["files"]), len(parse_models(c)), len(c["chain"])
        steps = f" + {n_steps} chain step(s)" if n_steps else ""
        if not messagebox.askyesno(APP, f"Rent {what},\nrun {n_files} file(s) through {n_models} model(s){steps}, "
                                        f"download the stems{' and destroy the machine' if c['auto_destroy'] else ''}?"):
            return
        self.bg(self._pipeline, c, None, o, job=True)

    def _pipeline(self, c, iid=None, offer=None):
        models, files = validate(c), c["files"]
        if not models:
            raise RuntimeError("Add at least one primary model (tab 2).")
        if not files:
            raise RuntimeError("Add input audio files in the Run tab.")
        gone = [f for f in files if not os.path.isfile(f)]
        if gone:
            raise RuntimeError("Missing input files: " + ", ".join(gone[:5]))
        out_local = Path(c["local_out"]).expanduser()
        out_local.mkdir(parents=True, exist_ok=True)
        vast = Vast(c["api_key"])
        pub = load_or_create_key(c["ssh_key_path"], self.log)
        t0, dph = time.time(), 0.0
        try:
            if iid is None:
                self._ensure_account_key(vast, pub)
                iid = self._rent_any(vast, c, offer)
            r, dph = self._connect_ready(vast, c, iid, pub)
            try:
                errors = self._run_job(r, c, models, files, out_local)
            finally:
                r.close()
            self.log(f"DONE{' (with problems, see above)' if errors else ''} - stems are in {out_local}")
            self.ui(self.bell)
        finally:
            mins = (time.time() - t0) / 60
            if dph:
                self.log(f"Job time {mins:.1f} min ≈ ${dph * mins / 60:.3f} at ${dph:.3f}/hr")
            if iid and c["auto_destroy"] and not self.stop_event.is_set():
                try:
                    self._destroy(vast, iid)
                    self.ui(self.vars["instance_id"].set, "")
                except Exception as e:
                    self.log(f"!! Could not destroy #{iid}: {e} - DESTROY IT on cloud.vast.ai!")
            elif iid:
                self.log(f"Instance #{iid} is still running (billing). Use 'Destroy now' in the Run tab when done.")
            self.ui(self.on_find_instances, True)

    def _progress(self, label):
        last = [0.0]

        def cb(done, total):
            now = time.time()
            if now - last[0] > 0.3 or done == total:
                last[0] = now
                self.status(f"{label}: {done / 1e6:.1f}/{total / 1e6:.1f} MB", int(done * 100 / total) if total else None)
            if self.stop_event.is_set():
                raise StopJob()
        return cb

    def _remote_line(self, line):
        if "%|" in line:  # tqdm progress bar -> status bar instead of log
            m = re.search(r"(\d+)%\|", line)
            self.status(line.strip()[-140:], int(m.group(1)) if m else None)
        else:
            self.log("  " + line)

    def _separate(self, r, sftp, c, model, in_paths, out_dir, model_dir, single, extra, meta, new, step, mi=0):
        """Run audio-separator on remote WAVs named f<key>.wav and register every stem it writes."""
        args = uvr_args(c, model, out_dir, model_dir, single, extra)
        self.log("$ audio-separator " + " ".join(q(x) for x in args) + f"  <{len(in_paths)} file(s)>")
        code = r.stream(PATH_PREFIX + f"mkdir -p {q(out_dir)} && audio-separator "
                        + " ".join(q(x) for x in args + in_paths), self._remote_line, self.stop_event)
        self.check_stop()
        try:
            names = sorted(sftp.listdir(out_dir))
        except IOError:
            names = []
        keys = []
        for name in names:
            m = OUT_RE.match(name)
            if m:
                src_key, stem = int(m.group(1)), m.group(2)
            else:
                m = LOOSE_RE.match(name)
                if not m:
                    continue
                src_key = int(m.group(1))
                paren = re.search(r"\(([^()]+)\)", m.group(2))
                stem = paren.group(1) if paren else (m.group(2) or "stem")
            src = meta.get(src_key)
            if src:
                keys.append(new(path=f"{out_dir}/{name}", song=src["song"], lineage=src["lineage"] + [stem],
                                model=Path(model).stem, step=step, input=src["input"], mi=mi))
        if code != 0 or not keys:
            self.log(f"!! {model}: exit code {code}, {len(keys)} stem file(s) produced")
        return keys, code == 0 and bool(keys)

    def _run_job(self, r, c, models, files, out_local):
        root = c["remote_dir"].strip().rstrip("/") or "/root/uvr"
        base = f"{root}/job_{time.strftime('%Y%m%d_%H%M%S')}"
        model_dir, fx_dir = f"{root}/models", f"{base}/fx"
        r.run(f"mkdir -p {q(base)}/src {q(base)}/in {q(fx_dir)} {q(base)}/final {q(model_dir)}")
        sftp = r.client.open_sftp()
        meta, counter, errors = {}, itertools.count(1), 0

        def new(**m):
            k = next(counter)
            m.setdefault("path", f"{fx_dir}/f{k}.wav")
            meta[k] = m
            return k

        try:
            sftp.putfo(io.BytesIO(FX_PY.encode()), f"{base}/uvr_fx.py")

            # 1) upload, then decode every input to stereo float WAV (optionally sliced)
            sr = num(c, "sample_rate", INT) or 44100
            start, dur = num(c, "sample_start", float), num(c, "sample_duration", float)
            cmds, inputs = [], []
            for i, f in enumerate(files, 1):
                self.check_stop()
                src = f"{base}/src/{i}{Path(f).suffix.lower()}"
                self.log(f"Upload {i}/{len(files)}: {Path(f).name} ({os.path.getsize(f) / 1e6:.1f} MB)")
                sftp.put(f, src, callback=self._progress(f"Uploading {Path(f).name}"))
                k = new(song=Path(f).stem, lineage=[], model="", step="input")
                meta[k].update(path=f"{base}/in/f{k}.wav", input=k)
                inputs.append(k)
                ff = (["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
                      + (["-ss", str(start)] if start else []) + ["-i", src] + (["-t", str(dur)] if dur else [])
                      + ["-ar", str(sr), "-ac", "2", "-c:a", "pcm_f32le", meta[k]["path"]])
                cmds.append(" ".join(q(x) for x in ff) + " || echo " + q(f"!! could not decode {Path(f).name}"))
            self.log(f"Decoding {len(inputs)} input(s) → {sr} Hz stereo"
                     + (f", slice {start or 0}s + {dur}s" if dur else f", from {start}s" if start else ""))
            r.stream(" ; ".join(cmds), self._remote_line, self.stop_event)
            self.check_stop()
            sizes = {a.filename: a.st_size or 0 for a in sftp.listdir_attr(f"{base}/in")}
            empty = [k for k in inputs if sizes.get(f"f{k}.wav", 0) < 4096]  # missing, or header only
            for k in empty:
                self.log(f"!! {meta[k]['song']}: no audio after decoding (slice start past the end?) - skipped")
            errors += len(empty)
            inputs = [k for k in inputs if k not in empty]
            if not inputs:
                raise RuntimeError("No input produced any audio - check the files and the slice settings.")

            # 2) primary pass
            prim = []
            for mi, model in enumerate(models, 1):
                self.check_stop()
                self.log(f"Primary model {mi}/{len(models)}: {model}")
                ks, ok = self._separate(r, sftp, c, model, [meta[k]["path"] for k in inputs], f"{base}/p{mi}",
                                        model_dir, c["single_stem"], "", meta, new, "primary", mi)
                errors += not ok
                prim += ks

            # 3) ensemble + inversion (uvr_fx.py)
            ops, individual = [], []
            nfft, hop = num(c, "ens_nfft", INT) or 2048, num(c, "ens_hop", INT) or 512
            if c["primary_mode"] == "ensemble" and len(models) > 1:
                weights = [float(w) for w in split_list(c["ens_weights"])]
                groups, results = {}, []
                for k in prim:
                    groups.setdefault((meta[k]["input"], meta[k]["lineage"][-1].lower()), []).append(k)
                for ks in groups.values():
                    if len(ks) == 1:
                        results.append(ks[0])
                        continue
                    m0 = meta[ks[0]]
                    k = new(song=m0["song"], lineage=list(m0["lineage"]), model=f"ensemble_{c['ens_algo']}",
                            step="ensemble", input=m0["input"])
                    ops.append({"op": "ensemble", "algo": c["ens_algo"], "n_fft": nfft, "hop": hop,
                                "inputs": [meta[x]["path"] for x in ks], "out": meta[k]["path"],
                                "weights": [weights[meta[x]["mi"] - 1] if meta[x]["mi"] <= len(weights) else 1.0
                                            for x in ks]})
                    results.append(k)
                individual = prim
                self.log(f"Ensembling {len(ops)} stem group(s) with {c['ens_algo']}")
            else:
                results = list(prim)
            inv = split_list(c["invert_stems"])
            for k in list(results) if inv else []:
                m0 = meta[k]
                if stem_match(m0["lineage"][-1], inv):
                    k2 = new(song=m0["song"], lineage=[f"No {m0['lineage'][-1]}"], model=m0["model"],
                             step="inverted", input=m0["input"])
                    ops.append({"op": "invert", "method": c["invert_method"], "n_fft": nfft, "hop": hop,
                                "stem": m0["path"], "mix": meta[m0["input"]]["path"], "out": meta[k2]["path"]})
                    results.append(k2)
            if ops:
                self.check_stop()
                sftp.putfo(io.BytesIO(json.dumps(ops).encode()), f"{base}/ops.json")
                r.stream(PATH_PREFIX + f"cd {q(base)} && python3 uvr_fx.py ops.json", self._remote_line, self.stop_event)
                self.check_stop()
                made = set(sftp.listdir(fx_dir))
                failed = {k for k in results if meta[k]["path"].startswith(fx_dir + "/")
                          and posixpath.basename(meta[k]["path"]) not in made}
                errors += len(failed)
                results = [k for k in results if k not in failed]

            # 4) post-processing chain
            step_out, prev = {}, results
            for si, st in enumerate(c["chain"], 1):
                self.check_stop()
                src = (st.get("src") or "previous").strip()
                if src == "primary":
                    pool = results
                elif src == "previous":
                    pool = prev
                else:
                    n = src.split()[-1]
                    pool = step_out.get(int(n), []) if n.isdigit() else []
                pats = split_list(st.get("stems", ""))
                sel = [k for k in pool if stem_match(meta[k]["lineage"][-1], pats)]
                self.log(f"Chain step {si}: {st['model']} on {len(sel)} file(s) [{st.get('stems') or '*'} from {src}]")
                if not sel:
                    self.log(f"!! step {si}: no stems matched - skipped")
                    errors += 1
                    step_out[si] = prev = []
                    continue
                in_dir = f"{base}/c{si}in"
                r.run(f"mkdir -p {q(in_dir)} && " + " && ".join(
                    f"ln -sf {q(meta[k]['path'])} {q(f'{in_dir}/f{k}.wav')}" for k in sel), timeout=120)
                ks, ok = self._separate(r, sftp, c, st["model"], [f"{in_dir}/f{k}.wav" for k in sel], f"{base}/c{si}",
                                        model_dir, st.get("single", ""), st.get("args", ""), meta, new, f"step{si}")
                errors += not ok
                step_out[si] = prev = ks

            # 5) pick, encode and download
            dl = (results if c["dl_primary"] else []) + (individual if c["ens_keep_individual"] else [])
            for si, st in enumerate(c["chain"], 1):
                if st.get("dl"):
                    dl += step_out.get(si, [])
            pats = split_list(c["stem_filter"])
            dl = [k for k in dict.fromkeys(dl) if stem_match(meta[k]["lineage"][-1], pats)]
            if not dl:
                raise RuntimeError("Nothing to download - check the download ticks and 'Only download stems'.")
            errors += self._deliver(r, sftp, c, base, meta, dl, out_local)
        finally:
            sftp.close()
            if not c["keep_remote"]:
                try:
                    r.run(f"rm -rf {q(base)}", timeout=60)
                except Exception:
                    pass
        return errors

    def _deliver(self, r, sftp, c, base, meta, keys, out_local):
        enc, ext = encode_args(c)
        jobs, cmds, used = [], [], set()
        for i, k in enumerate(keys, 1):
            m = meta[k]
            folder = out_local / safe_name(m["song"]) if c["subfolder_per_song"] else out_local
            dst = unique_path(folder, out_name(c, m), ext, used, c["if_exists"])
            remote = f"{base}/final/{i}.{ext}"
            ff = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", m["path"], *enc, remote]
            cmds.append(" ".join(q(x) for x in ff) + " || echo " + q(f"!! encode failed: {dst.name}"))
            jobs.append((remote, dst))
        self.log(f"Encoding {len(keys)} file(s) → {c['final_format']} ...")
        r.stream(" ; ".join(cmds), self._remote_line, self.stop_event)
        self.check_stop()
        errors = 0
        for remote, dst in jobs:
            try:
                sftp.stat(remote)
                dst.parent.mkdir(parents=True, exist_ok=True)
                sftp.get(remote, str(dst), callback=self._progress(f"Downloading {dst.name}"))
                self.log(f"  ↓ {dst}")
            except IOError as e:
                errors += 1
                self.log(f"!! could not download {dst.name}: {e}")
        return errors

    # ---------- shutdown ----------
    def _on_close(self):
        self._save()
        alive = sorted(self.session_instances | self._uvr_ids())
        if alive:
            ans = messagebox.askyesnocancel(APP, "These UVR instances may still be running (and billing): "
                                            + ", ".join(f"#{i}" for i in alive) + "\n\nDestroy them now?")
            if ans is None:
                return
            if ans:
                try:
                    vast = Vast(self.vars["api_key"].get())
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
