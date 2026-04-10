import os
import subprocess
import re
import json
import tempfile
import shutil
import threading
import uuid
from pathlib import Path

import gradio as gr

try:
    from huggingface_hub import HfApi, list_models
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# ══════════════════════════════════════════════════════════════
#  RAILWAY CONFIG — leer variables de entorno
# ══════════════════════════════════════════════════════════════
PORT        = int(os.environ.get("PORT", 7860))
HF_TOKEN    = os.environ.get("HF_TOKEN", "")          # token por defecto
HF_REPO     = os.environ.get("HF_REPO", "")           # repo por defecto
NUM_THREADS = int(os.environ.get("NUM_THREADS", os.cpu_count() or 4))

# Detectar GPU disponible
def detect_gpu():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                            "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return None

GPU_INFO = detect_gpu()
HAS_GPU  = GPU_INFO is not None

# Encoder óptimo según hardware disponible
def best_h264():
    if HAS_GPU:
        r = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
        if "h264_nvenc" in r.stdout:
            return "h264_nvenc"
    return "libx264"

def best_h265():
    if HAS_GPU:
        r = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
        if "hevc_nvenc" in r.stdout:
            return "hevc_nvenc"
    return "libx265"

H264_ENC = best_h264()
H265_ENC = best_h265()

# ══════════════════════════════════════════════════════════════
#  ESTADO GLOBAL
# ══════════════════════════════════════════════════════════════
JOBS: dict = {}
LOCK = threading.Lock()
LOGS_DIR = Path("/tmp/vcpro_logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Semáforo para limitar jobs concurrentes
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT_JOBS", 3))
JOB_SEMAPHORE  = threading.Semaphore(MAX_CONCURRENT)


# ══════════════════════════════════════════════════════════════
#  UTILS
# ══════════════════════════════════════════════════════════════
def safe_name(s: str, maxlen=120) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', s).strip()[:maxlen]

def fmt_dur(secs: float) -> str:
    s = int(secs)
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"

def jlog(jid: str, msg: str):
    with LOCK:
        if jid in JOBS:
            JOBS[jid].setdefault("log_lines", []).append(msg)
    with open(LOGS_DIR / f"{jid}.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def jset(jid: str, **kw):
    with LOCK:
        if jid in JOBS:
            JOBS[jid].update(kw)

def jget(jid: str) -> dict:
    with LOCK:
        return dict(JOBS.get(jid, {}))

def read_log(jid: str) -> str:
    p = LOGS_DIR / f"{jid}.log"
    return p.read_text("utf-8") if p.exists() else ""

def _parse_idx(s) -> int:
    try:
        return int(str(s).split("]")[0].strip("[").split(":")[0]) if s else 0
    except:
        return 0


# ══════════════════════════════════════════════════════════════
#  FFPROBE
# ══════════════════════════════════════════════════════════════
def probe(source: str) -> dict:
    info = {"audio": [], "subs": [], "duration": 0.0, "title": ""}
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", source],
            capture_output=True, text=True, timeout=90
        )
        if r.returncode != 0:
            return info
        data = json.loads(r.stdout)
        tags = data.get("format", {}).get("tags", {})
        info["title"] = (tags.get("title") or tags.get("TITLE") or "").strip()
        info["duration"] = float(data.get("format", {}).get("duration", 0) or 0)
        ai = si = 0
        for s in data.get("streams", []):
            t = s.get("tags", {})
            ct = s.get("codec_type", "")
            if ct == "audio":
                info["audio"].append({
                    "idx": ai, "codec": s.get("codec_name", "?"),
                    "lang": t.get("language", "und"), "ch": s.get("channels", 2),
                    "title": t.get("title", ""),
                })
                ai += 1
            elif ct == "subtitle":
                info["subs"].append({
                    "idx": si, "codec": s.get("codec_name", "?"),
                    "lang": t.get("language", "und"), "title": t.get("title", ""),
                    "forced": s.get("disposition", {}).get("forced", 0) == 1,
                })
                si += 1
    except Exception as e:
        print(f"probe err: {e}")
    return info

def audio_label(t):
    lang = t["lang"] if t["lang"] != "und" else "?"
    return f"[{t['idx']}] {lang} · {t['codec']} · {t['ch']}ch {t['title']}".rstrip(" ·")

def sub_label(t):
    lang = t["lang"] if t["lang"] != "und" else "?"
    return f"[{t['idx']}] {lang} · {t['codec']}" + (" (Forzado)" if t["forced"] else "")


# ══════════════════════════════════════════════════════════════
#  FFMPEG — corre en Railway, sin conexión del usuario
# ══════════════════════════════════════════════════════════════
NET_ARGS = [
    "-user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "-headers", "Referer: https://google.com\r\n",
    "-timeout", "180000000",
    "-reconnect", "1",
    "-reconnect_streamed", "1",
    "-reconnect_at_eof", "1",
    "-reconnect_delay_max", "30",
    "-rw_timeout", "180000000",
    "-multiple_requests", "1",
]

def _gpu_extra_args(encoder: str) -> list:
    """Flags adicionales para encoders NVENC."""
    if "nvenc" in encoder:
        return ["-gpu", "0", "-rc:v", "vbr", "-b:v", "0"]
    return []

def run_ffmpeg(cmd: list, jid: str, total: float, label: str) -> bool:
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            universal_newlines=True, bufsize=1
        )
        jset(jid, _proc=proc)
        pat = re.compile(r"time=(\d+):(\d+):(\d+)\.(\d+)")
        for line in proc.stderr:
            m = pat.search(line)
            if m:
                h, mi, s, cs = map(int, m.groups())
                cur = h * 3600 + mi * 60 + s + cs / 100
                pct = min(99, int(cur / max(total, 1) * 100))
                jset(jid, progress=pct, progress_label=label,
                     progress_time=fmt_dur(cur), total_time=fmt_dur(total))
        proc.wait()
        return proc.returncode == 0
    except Exception as e:
        jlog(jid, f"  ✗ ffmpeg exc: {e}")
        return False


# ══════════════════════════════════════════════════════════════
#  HF UPLOAD
# ══════════════════════════════════════════════════════════════
def hf_upload_file(api: HfApi, local: str, repo_id: str,
                   path_in_repo: str, jid: str) -> bool:
    try:
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
        )
        return True
    except Exception as e:
        jlog(jid, f"  ✗ upload error: {e}")
        return False


# ══════════════════════════════════════════════════════════════
#  PROCESAR UN ARCHIVO
# ══════════════════════════════════════════════════════════════
def process_one(jid: str, token: str, repo_id: str,
                source: str, is_url: bool, mode: str,
                audio_idx: int, gen_single: bool,
                extract_sub: bool, sub_idx: int,
                file_name: str,
                hf_folder: str,
                ) -> bool:

    extra = NET_ARGS if is_url else []

    jset(jid, step=1, step_name="Analizando")
    jlog(jid, f"  ⟳ probe {source[:60]}…")
    info = probe(source)
    dur = info["duration"] if info["duration"] > 0 else 1
    jlog(jid, f"  ℹ {fmt_dur(dur)} | {len(info['audio'])} aud | {len(info['subs'])} sub")
    jlog(jid, f"  🖥  GPU: {GPU_INFO or 'No detectada — usando CPU'}")
    jlog(jid, f"  🔧 Threads CPU: {NUM_THREADS} | Enc H264: {H264_ENC} | H265: {H265_ENC}")

    # Usar /tmp local de Railway (Railway tiene discos efímeros rápidos)
    tmp = tempfile.mkdtemp(prefix="vcpro_", dir="/tmp")
    out_mp4 = os.path.join(tmp, f"{file_name}.mp4")

    cmd = ["ffmpeg", "-y"] + extra + ["-i", source, "-map", "0:v:0"]
    for i in range(len(info["audio"])):
        cmd.extend(["-map", f"0:a:{i}"])

    # ── LÓGICA DE VIDEO con GPU o CPU según disponibilidad ──
    if mode == "Copy (Video Original)":
        cmd += ["-c:v", "copy"]

    elif mode == "H.264 4K":
        if "nvenc" in H264_ENC:
            cmd += [
                "-c:v", H264_ENC,
                "-vf", "scale=-2:2160",
                "-preset", "p5",           # NVENC preset balanceado
                "-cq", "18",
                "-b:v", "0",
                "-maxrate", "50M", "-bufsize", "100M",
            ]
        else:
            cmd += [
                "-c:v", H264_ENC,
                "-vf", "scale=-2:2160",
                "-preset", "slow",
                "-crf", "18",
                "-threads", str(NUM_THREADS),
                "-maxrate", "50M", "-bufsize", "100M",
            ]

    elif mode == "H.265 4K":
        if "nvenc" in H265_ENC:
            cmd += [
                "-c:v", H265_ENC,
                "-vf", "scale=-2:2160",
                "-preset", "p5",
                "-cq", "20",
                "-b:v", "0",
                "-maxrate", "30M", "-bufsize", "60M",
            ]
        else:
            cmd += [
                "-c:v", H265_ENC,
                "-vf", "scale=-2:2160",
                "-preset", "slow",
                "-crf", "20",
                "-threads", str(NUM_THREADS),
                "-maxrate", "30M", "-bufsize", "60M",
            ]

    elif mode == "H.264 1080p":
        enc = H264_ENC
        if "nvenc" in enc:
            cmd += ["-c:v", enc, "-vf", "scale=-2:1080", "-preset", "p5",
                    "-cq", "20", "-b:v", "0", "-maxrate", "20M", "-bufsize", "40M"]
        else:
            cmd += ["-c:v", enc, "-vf", "scale=-2:1080", "-preset", "fast",
                    "-crf", "20", "-threads", str(NUM_THREADS),
                    "-maxrate", "20M", "-bufsize", "40M"]

    elif mode == "H.265 1080p":
        enc = H265_ENC
        if "nvenc" in enc:
            cmd += ["-c:v", enc, "-vf", "scale=-2:1080", "-preset", "p5",
                    "-cq", "22", "-b:v", "0", "-maxrate", "12M", "-bufsize", "24M"]
        else:
            cmd += ["-c:v", enc, "-vf", "scale=-2:1080", "-preset", "fast",
                    "-crf", "22", "-threads", str(NUM_THREADS),
                    "-maxrate", "12M", "-bufsize", "24M"]

    # ── AUDIO: siempre FLAC ──
    for i in range(len(info["audio"])):
        cmd += [f"-c:a:{i}", "flac", f"-compression_level:a:{i}", "5"]

    cmd += ["-map_metadata", "0", out_mp4]

    jset(jid, step=2, step_name="Convirtiendo", progress=0)
    jlog(jid, f"  ⚙ Convirtiendo ({mode} · {'GPU' if HAS_GPU else 'CPU'} + FLAC)…")
    ok = run_ffmpeg(cmd, jid, dur, "Convirtiendo")
    if not ok:
        jlog(jid, "  ✗ conversión falló")
        shutil.rmtree(tmp, ignore_errors=True)
        return False
    jset(jid, progress=100)
    jlog(jid, "  ✓ conversión OK")

    uploads: list[tuple[str, str]] = [
        (out_mp4, f"{hf_folder}/{file_name}.mp4")
    ]

    if gen_single and info["audio"] and audio_idx < len(info["audio"]):
        sp = os.path.join(tmp, f"{file_name}_aud{audio_idx}.mp4")
        sc = ["ffmpeg", "-y", "-i", out_mp4,
              "-map", "0:v:0", "-map", f"0:a:{audio_idx}", "-c", "copy", sp]
        run_ffmpeg(sc, jid, dur, "Audio individual")
        if os.path.exists(sp):
            uploads.append((sp, f"{hf_folder}/{file_name}_aud{audio_idx}.mp4"))

    if extract_sub and info["subs"] and sub_idx < len(info["subs"]):
        vtt = os.path.join(tmp, f"{file_name}_sub{sub_idx}.vtt")
        sc = ["ffmpeg", "-y"] + extra + [
            "-i", source, "-map", f"0:s:{sub_idx}", "-c:s", "webvtt", vtt]
        try:
            subprocess.run(sc, capture_output=True, check=True, timeout=120)
            if os.path.exists(vtt):
                uploads.append((vtt, f"{hf_folder}/{file_name}_sub{sub_idx}.vtt"))
            jlog(jid, "  ✓ subtítulo extraído")
        except Exception as e:
            jlog(jid, f"  ⚠ sub error: {e}")

    jset(jid, step=3, step_name="Subiendo", progress=0)
    jlog(jid, f"  ☁ subiendo {len(uploads)} archivo(s) → {hf_folder}/")

    if not HF_AVAILABLE:
        jlog(jid, "  ✗ huggingface_hub no disponible")
        shutil.rmtree(tmp, ignore_errors=True)
        return False

    try:
        api = HfApi(token=token)
        try:
            api.create_repo(repo_id=repo_id, repo_type="model",
                            private=True, exist_ok=True)
        except Exception:
            pass

        for fi, (local_path, repo_path) in enumerate(uploads):
            jlog(jid, f"  ↑ {Path(local_path).name}")
            jset(jid,
                 progress=int(fi / len(uploads) * 100),
                 progress_label=f"Subiendo {fi+1}/{len(uploads)}")
            if not hf_upload_file(api, local_path, repo_id, repo_path, jid):
                shutil.rmtree(tmp, ignore_errors=True)
                return False

        jset(jid, progress=100, progress_label="Subida completa")
        jlog(jid, f"  ✓ subido → {repo_id}/{hf_folder}/")
    except Exception as e:
        jlog(jid, f"  ✗ upload exc: {e}")
        shutil.rmtree(tmp, ignore_errors=True)
        return False

    shutil.rmtree(tmp, ignore_errors=True)
    return True


# ══════════════════════════════════════════════════════════════
#  JOB THREADS con semáforo para concurrencia controlada
# ══════════════════════════════════════════════════════════════
def _thread_single(jid, token, repo, source, is_url, mode,
                   audio_idx, gen_single, extract_sub, sub_idx,
                   cname, ctype, serie, season, ep):
    with JOB_SEMAPHORE:
        jset(jid, status="running", total_items=1, done_items=0)
        jlog(jid, f"▸ {jid} · individual · Railway")

        if cname.strip():
            fname = safe_name(cname.strip())
        elif ctype == "serie" and serie.strip():
            fname = safe_name(f"{serie.strip()}_T{int(season)}_Ep{int(ep)}")
        else:
            fname = safe_name(Path(source).stem or "video")

        if ctype == "serie" and serie.strip():
            hf_folder = safe_name(f"{serie.strip()}_T{int(season)}")
        else:
            hf_folder = "videos"

        jset(jid, current_name=fname, hf_folder=hf_folder)
        jlog(jid, f"  📁 {hf_folder}/{fname}.mp4")

        ok = process_one(jid, token, repo, source, is_url, mode,
                         audio_idx, gen_single, extract_sub, sub_idx, fname, hf_folder)

        jset(jid, done_items=1,
             status="done" if ok else "error",
             step=4 if ok else 0,
             step_name="Completado" if ok else "Error",
             progress=100 if ok else 0)
        jlog(jid, "✓ Completado" if ok else "✗ Falló")


def _thread_bulk(jid, token, repo, entries, mode, gen_single, extract_sub):
    with JOB_SEMAPHORE:
        total = len(entries)
        jset(jid, status="running", total_items=total, done_items=0)
        jlog(jid, f"▸ {jid} · bulk · {total} urls · Railway")

        success = 0
        for i, e in enumerate(entries):
            with LOCK:
                if JOBS.get(jid, {}).get("cancelled"):
                    jlog(jid, "⛔ Cancelado")
                    break

            jset(jid, current_item=i+1, current_name=e["fname"],
                 hf_folder=e["hf_folder"],
                 step=1, step_name=f"[{i+1}/{total}] {e['fname']}")
            jlog(jid, f"\n[{i+1}/{total}] {e['fname']}")

            ok = process_one(
                jid, token, repo, e["url"], True, mode,
                e.get("audio_idx", 0), gen_single, extract_sub,
                e.get("sub_idx", 0), e["fname"], e["hf_folder"]
            )
            if ok:
                success += 1
            jset(jid, done_items=i+1)
            jlog(jid, f"  {'✓' if ok else '✗'} [{i+1}/{total}]")

        jset(jid, status="done", step=4, step_name="Completado",
             progress=100, result=f"{success}/{total} ok")
        jlog(jid, f"\n✓ Bulk: {success}/{total}")


# ══════════════════════════════════════════════════════════════
#  HELPERS UI
# ══════════════════════════════════════════════════════════════
def _new_job() -> str:
    jid = uuid.uuid4().hex[:8]
    with LOCK:
        JOBS[jid] = {
            "status": "queued", "log_lines": [], "result": "",
            "progress": 0, "progress_label": "", "progress_time": "00:00:00",
            "total_time": "00:00:00", "step": 0, "step_name": "En cola…",
            "total_items": 1, "done_items": 0, "current_item": 0,
            "current_name": "", "hf_folder": "", "_proc": None,
        }
    return jid

def _get_repo(token, repo):
    if repo and "/" in repo:
        return repo
    if HF_REPO and "/" in HF_REPO:
        return HF_REPO
    try:
        api = HfApi(token=token)
        uname = api.whoami(token=token)["name"]
        return f"{uname}/media-storage"
    except:
        return "user/media-storage"


# ══════════════════════════════════════════════════════════════
#  GRADIO HANDLERS
# ══════════════════════════════════════════════════════════════
def do_analyze(src_file, src_url):
    source = None
    if src_file:
        fl = src_file if isinstance(src_file, list) else [src_file]
        source = fl[0].name
    elif src_url and src_url.strip():
        source = src_url.strip()
    if not source:
        return gr.update(choices=[]), gr.update(choices=[]), "⚠ Sin fuente"
    info = probe(source)
    ac = [audio_label(t) for t in info["audio"]]
    sc = [sub_label(t) for t in info["subs"]]
    txt = f"✓ {info['title'] or '—'} · {fmt_dur(info['duration'])} · {len(ac)} aud · {len(sc)} sub"
    return (gr.update(choices=ac, value=ac[0] if ac else None),
            gr.update(choices=sc, value=sc[0] if sc else None), txt)

def do_load_repos(token):
    t = token or HF_TOKEN
    if not t or not HF_AVAILABLE:
        return gr.update(choices=[])
    try:
        api = HfApi(token=t)
        uname = api.whoami(token=t)["name"]
        ch = [m.modelId for m in list_models(token=t, author=uname, limit=200)]
        return gr.update(choices=ch, value=ch[0] if ch else None)
    except Exception:
        return gr.update(choices=[])

def do_single(token, repo, src_file, src_url, mode,
              aud_s, sub_s, gen_single, extract_sub,
              cname, ctype, serie, season, ep):
    token = token or HF_TOKEN
    if not token:
        return None, gr.update(active=False)
    source, is_url = None, False
    if src_file:
        fl = src_file if isinstance(src_file, list) else [src_file]
        source = fl[0].name
    elif src_url and src_url.strip():
        source, is_url = src_url.strip(), True
    if not source:
        return None, gr.update(active=False)

    ai = _parse_idx(aud_s)
    si = _parse_idx(sub_s)
    try: season = int(season or 1)
    except: season = 1
    try: ep = int(ep or 1)
    except: ep = 1

    repo = _get_repo(token, repo)
    jid  = _new_job()

    threading.Thread(
        target=_thread_single,
        args=(jid, token, repo, source, is_url, mode,
              ai, gen_single, extract_sub, si,
              cname or "", ctype, serie or "", season, ep),
        daemon=True
    ).start()
    return jid, gr.update(active=True)

def do_bulk(token, repo, urls_text, mode, gen_single, extract_sub,
            btype, bserie, bseason, bep_start):
    token = token or HF_TOKEN
    if not token or not urls_text.strip():
        return None, gr.update(active=False)
    try: bseason = int(bseason or 1)
    except: bseason = 1
    try: bep_start = int(bep_start or 1)
    except: bep_start = 1

    lines = [l.strip() for l in urls_text.strip().split("\n") if l.strip()]
    if not lines:
        return None, gr.update(active=False)

    repo = _get_repo(token, repo)

    entries = []
    for i, url in enumerate(lines):
        if btype == "serie":
            ep = bep_start + i
            sname = safe_name(bserie or "Serie")
            entries.append({
                "url": url,
                "fname": f"{sname}_T{bseason}_Ep{ep}",
                "hf_folder": f"{sname}_T{bseason}",
                "audio_idx": 0, "sub_idx": 0,
            })
        else:
            entries.append({
                "url": url,
                "fname": safe_name(Path(url).stem or f"video_{i+1}"),
                "hf_folder": "videos",
                "audio_idx": 0, "sub_idx": 0,
            })

    jid = _new_job()
    jset(jid, total_items=len(entries))
    threading.Thread(
        target=_thread_bulk,
        args=(jid, token, repo, entries, mode, gen_single, extract_sub),
        daemon=True
    ).start()
    return jid, gr.update(active=True)

def do_cancel(jid):
    if not jid: return
    with LOCK:
        j = JOBS.get(jid)
        if j:
            j["cancelled"] = True
            j["status"]    = "cancelled"
            p = j.get("_proc")
            if p:
                try: p.terminate()
                except: pass

def do_recover(jid_input):
    jid = (jid_input or "").strip()
    if not jid or not read_log(jid):
        return None, gr.update(active=False)
    with LOCK:
        j = dict(JOBS.get(jid, {}))
    active = j.get("status") in ("queued", "running")
    return jid, gr.update(active=active)


# ══════════════════════════════════════════════════════════════
#  RENDER PANEL
# ══════════════════════════════════════════════════════════════
def render_panel(jid):
    if not jid:
        return _idle_html(), gr.update(active=False)
    j = jget(jid)
    log_txt = read_log(jid)
    active = j.get("status") in ("queued", "running")
    return _job_html(j, log_txt, jid), gr.update(active=active)


def _hw_badge():
    if HAS_GPU:
        return f'<span style="background:#00e5a015;border:1px solid #00e5a040;color:#00e5a0;padding:2px 8px;border-radius:4px;font-size:9px;font-weight:700;">⚡ GPU · {GPU_INFO}</span>'
    return f'<span style="background:#4da8ff15;border:1px solid #4da8ff40;color:#4da8ff;padding:2px 8px;border-radius:4px;font-size:9px;font-weight:700;">🖥 CPU · {NUM_THREADS} threads</span>'


def _idle_html():
    return f"""
<div style="font-family:'Space Grotesk',sans-serif;display:flex;flex-direction:column;
            align-items:center;justify-content:center;min-height:380px;gap:14px;padding:20px;">
  <svg width="48" height="48" viewBox="0 0 48 48" fill="none" style="opacity:.12">
    <polygon points="24,4 44,14 44,34 24,44 4,34 4,14" stroke="#4da8ff" stroke-width="1.5" fill="none"/>
    <polygon points="24,12 36,18 36,30 24,36 12,30 12,18" stroke="#4da8ff" stroke-width="1" fill="none"/>
  </svg>
  <div style="color:#0d2040;font-size:11px;letter-spacing:.16em;text-transform:uppercase;font-weight:700;">
    sin job activo
  </div>
  <div style="margin-top:4px;">{_hw_badge()}</div>
  <div style="color:#071020;font-size:11px;max-width:240px;text-align:center;line-height:1.8;font-family:'JetBrains Mono',monospace;">
    Railway activo · Max {MAX_CONCURRENT} jobs concurrentes.<br>
    Podés cerrar la ventana —<br>
    todo sigue corriendo en el servidor.
  </div>
</div>"""


def _job_html(j: dict, log_txt: str, jid: str) -> str:
    status = j.get("status", "?")
    step   = j.get("step", 0)
    sname  = j.get("step_name", "")
    prog   = j.get("progress", 0)
    pl     = j.get("progress_label", "")
    pt     = j.get("progress_time", "00:00:00")
    tt     = j.get("total_time", "00:00:00")
    total  = j.get("total_items", 1)
    done   = j.get("done_items", 0)
    cname  = j.get("current_name", "")
    hff    = j.get("hf_folder", "")
    result = j.get("result", "")

    if status == "done":
        sc, slabel, bdr = "#00e5a0", "COMPLETADO", "#00e5a020"
        glow = "#00e5a044"
    elif status in ("error", "cancelled"):
        sc, slabel, bdr = "#ff4d5e", status.upper(), "#ff4d5e20"
        glow = "#ff4d5e44"
    else:
        sc, slabel, bdr = "#4da8ff", "PROCESANDO", "#4da8ff18"
        glow = "#4da8ff44"

    STEPS = [("01","Análisis"),("02","Conversión"),("03","Upload"),("04","Listo")]
    steps = ""
    for i,(num,lbl) in enumerate(STEPS):
        n = i+1
        if n < step:
            bg,tc,dot = "#00e5a0","#000","✓"
        elif n == step:
            bg,tc,dot = "#1d4ed8","#fff",num
        else:
            bg,tc,dot = "#0a1828","#1e3a5a",num
        steps += f"""
        <div style="display:flex;align-items:center;gap:9px;">
          <div style="width:26px;height:26px;border-radius:7px;background:{bg};
                      display:flex;align-items:center;justify-content:center;
                      font-size:10px;font-weight:800;color:{tc};flex-shrink:0;
                      box-shadow:0 0 10px {bg}55;">{dot}</div>
          <span style="font-size:11px;color:{'#a0c0e0' if n==step else '#1e3a5a'};
                       font-weight:{'700' if n==step else '500'};">{lbl}</span>
        </div>
        {'<div style="width:1px;height:6px;background:#0a1828;margin-left:12px;"></div>' if i<3 else ''}"""

    if status == "done":
        bc, pd = "#00e5a0", 100
    elif status in ("error","cancelled"):
        bc, pd = "#ff4d5e", prog
    else:
        bc, pd = "#4da8ff", prog

    bulk = ""
    if total > 1:
        bpct = int(done/total*100) if total else 0
        bulk = f"""
        <div style="margin-top:12px;padding:11px 13px;background:#040c18;
                    border:1px solid #0a1828;border-radius:8px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:7px;">
            <span style="color:#0d2040;font-size:9px;font-weight:700;
                         text-transform:uppercase;letter-spacing:.12em;">Bulk</span>
            <span style="color:#4da8ff;font-size:12px;font-weight:700;">{done} / {total}</span>
          </div>
          <div style="background:#08121e;border-radius:4px;height:4px;overflow:hidden;">
            <div style="height:100%;width:{bpct}%;border-radius:4px;
                        background:linear-gradient(90deg,#1d4ed8,#4da8ff);
                        transition:width .6s ease;"></div>
          </div>
          {f'<div style="color:#0d2040;font-size:9px;font-family:monospace;margin-top:5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">↳ {cname}</div>' if cname else ''}
        </div>"""

    dest = ""
    if hff:
        dest = f"""
        <div style="margin-top:8px;padding:6px 10px;background:#02080e;
                    border:1px solid #08121e;border-radius:6px;
                    font-family:'JetBrains Mono',monospace;font-size:9px;
                    color:#0d2040;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
          ↗ {hff}/
        </div>"""

    lines = [l for l in log_txt.strip().split("\n") if l.strip()][-16:]
    def lc(l):
        if l.startswith("  ✓") or l.startswith("✓"): return "#00e5a0"
        if l.startswith("  ✗") or l.startswith("✗"): return "#ff4d5e"
        if l.startswith("  ⚠"): return "#ffb347"
        if l.startswith("  ↑") or l.startswith("  ☁"): return "#4da8ff"
        if l.startswith("▸") or l.startswith("[") : return "#2a5080"
        if l.startswith("  ⚙") or l.startswith("  ⟳"): return "#1e6090"
        if "GPU" in l: return "#00e5a0"
        if "CPU" in l or "Thread" in l: return "#4da8ff"
        return "#0d2040"
    logs_h = "".join(
        f'<div style="color:{lc(l)};padding:1px 0;white-space:pre;overflow:hidden;text-overflow:ellipsis;">'
        f'{l.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")}</div>'
        for l in lines
    ) or '<span style="color:#08121e;">—</span>'

    pulse = "animation:hb 1.4s ease-in-out infinite;" if status in ("queued","running") else ""
    hw_badge = _hw_badge()

    return f"""
<div style="font-family:'Space Grotesk',sans-serif;user-select:none;">
  <div style="display:flex;align-items:center;justify-content:space-between;
              padding:10px 14px;background:{bdr};
              border:1px solid {sc}33;border-radius:10px;margin-bottom:14px;
              box-shadow:0 0 30px {glow};">
    <div style="display:flex;align-items:center;gap:10px;">
      <div style="width:7px;height:7px;border-radius:50%;background:{sc};{pulse}
                  box-shadow:0 0 8px {sc};"></div>
      <span style="color:{sc};font-size:10px;font-weight:800;letter-spacing:.14em;">{slabel}</span>
    </div>
    <div style="display:flex;gap:8px;align-items:center;">
      {hw_badge}
      <span style="color:#0d2040;font-size:9px;font-family:'JetBrains Mono',monospace;">{jid}</span>
    </div>
    {f'<span style="color:{sc};font-size:11px;font-weight:700;">{result}</span>' if result else ''}
  </div>

  <div style="display:grid;grid-template-columns:110px 1fr;gap:14px;align-items:start;">
    <div style="padding:12px 10px;background:#040c18;border:1px solid #08121e;border-radius:10px;">
      {steps}
    </div>
    <div>
      <div style="margin-bottom:12px;">
        <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:5px;">
          <span style="color:#0d2040;font-size:9px;font-weight:700;
                       text-transform:uppercase;letter-spacing:.1em;">{pl or sname}</span>
          <span style="color:{bc};font-size:13px;font-weight:800;">{pd}%</span>
        </div>
        <div style="background:#040c18;border-radius:6px;height:8px;overflow:hidden;border:1px solid #08121e;">
          <div style="height:100%;width:{pd}%;border-radius:6px;
                      background:linear-gradient(90deg,#1d3a8a,{bc});
                      transition:width .5s ease;
                      box-shadow:0 0 14px {bc}55;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;margin-top:3px;">
          <span style="color:#08121e;font-size:9px;font-family:monospace;">{pt}</span>
          <span style="color:#08121e;font-size:9px;font-family:monospace;">{tt}</span>
        </div>
      </div>
      {bulk}
      {dest}
      <div style="margin-top:10px;background:#02080e;border:1px solid #08121e;border-radius:8px;overflow:hidden;">
        <div style="padding:4px 10px;border-bottom:1px solid #08121e;background:#040c18;">
          <span style="color:#08182e;font-size:8px;font-weight:800;letter-spacing:.16em;text-transform:uppercase;">log</span>
        </div>
        <div style="padding:8px 10px;min-height:110px;max-height:190px;overflow-y:auto;
                    font-family:'JetBrains Mono',monospace;font-size:10px;line-height:1.8;">
          {logs_h}
        </div>
      </div>
    </div>
  </div>
</div>
<style>@keyframes hb{{0%,100%{{opacity:.4;transform:scale(.8)}}50%{{opacity:1;transform:scale(1.2)}}}}</style>"""


# ══════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
*,*::before,*::after{box-sizing:border-box;}
body,.gradio-container{background:#020912!important;font-family:'Space Grotesk',sans-serif!important;color:#1e3a5a!important;}
.gradio-container{max-width:1300px!important;margin:0 auto!important;padding:14px!important;}
.gr-group,.gr-form,.gr-box{background:#040c18!important;border:1px solid #08121e!important;border-radius:12px!important;box-shadow:none!important;}
label span,.label-wrap span{color:#0d2040!important;font-size:9px!important;font-weight:800!important;text-transform:uppercase!important;letter-spacing:.12em!important;}
input[type=text],input[type=password],textarea,.gr-input input{background:#02080e!important;border:1px solid #08121e!important;border-radius:8px!important;color:#2a5080!important;font-family:'JetBrains Mono',monospace!important;font-size:12px!important;transition:border-color .2s!important;}
input:focus,textarea:focus{border-color:#1d4ed8!important;box-shadow:0 0 0 3px rgba(29,78,216,.1)!important;outline:none!important;}
button.primary{background:linear-gradient(135deg,#0f2a6a,#1d4ed8)!important;border:none!important;border-radius:10px!important;color:#7ab8ff!important;font-family:'Space Grotesk',sans-serif!important;font-weight:700!important;font-size:13px!important;letter-spacing:.04em!important;padding:11px 20px!important;box-shadow:0 2px 24px rgba(29,78,216,.3)!important;transition:all .2s!important;}
button.primary:hover{background:linear-gradient(135deg,#1d4ed8,#2563ff)!important;transform:translateY(-1px)!important;box-shadow:0 6px 30px rgba(29,78,216,.45)!important;}
button.secondary{background:#040c18!important;border:1px solid #08121e!important;color:#0d2040!important;border-radius:8px!important;font-family:'Space Grotesk',sans-serif!important;font-size:12px!important;transition:all .2s!important;}
button.secondary:hover{border-color:#1d4ed8!important;color:#4da8ff!important;}
button.stop{background:#0a0408!important;border:1px solid #1a0510!important;color:#ff4d5e!important;border-radius:8px!important;font-size:12px!important;transition:all .2s!important;}
button.stop:hover{background:#120608!important;}
.gr-tab-nav{border-bottom:1px solid #08121e!important;}
.gr-tab-nav button{background:transparent!important;border:none!important;border-bottom:2px solid transparent!important;color:#0d2040!important;font-family:'Space Grotesk',sans-serif!important;font-size:12px!important;font-weight:600!important;padding:8px 18px!important;letter-spacing:.04em!important;border-radius:0!important;transition:all .2s!important;}
.gr-tab-nav button.selected,.gr-tab-nav button[aria-selected=true]{color:#4da8ff!important;border-bottom-color:#1d4ed8!important;}
.gr-radio-group label,.gr-checkbox label{color:#0d2040!important;font-size:12px!important;}
::-webkit-scrollbar{width:3px;height:3px;}
::-webkit-scrollbar-track{background:#02080e;}
::-webkit-scrollbar-thumb{background:#08182e;border-radius:2px;}
::-webkit-scrollbar-thumb:hover{background:#1d4ed8;}
.divider{height:1px;background:#08121e;margin:10px 0;}
"""


# ══════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════
with gr.Blocks(css=CSS, title="VCPro · Railway") as demo:

    _jid = gr.State(None)

    gr.HTML(f"""
    <div style="padding:20px 2px 14px;display:flex;align-items:baseline;gap:16px;flex-wrap:wrap;">
      <span style="font-family:'Space Grotesk',sans-serif;font-size:24px;font-weight:700;
                   color:#1d4ed8;letter-spacing:-0.5px;">VCPro</span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#08182e;
                   letter-spacing:.18em;text-transform:uppercase;">
        Railway · cerrá la ventana cuando quieras</span>
      {_hw_badge()}
    </div>
    <div style="height:1px;background:linear-gradient(90deg,#1d4ed8,#1d4ed820,transparent);margin-bottom:18px;"></div>
    """)

    with gr.Row(equal_height=False):

        with gr.Column(scale=5, min_width=310):

            with gr.Group():
                gr.HTML('<div style="color:#08182e;font-size:8px;font-weight:800;letter-spacing:.18em;text-transform:uppercase;padding:12px 14px 2px;">HuggingFace</div>')
                with gr.Row():
                    token_in = gr.Textbox(
                        label="Token",
                        type="password",
                        placeholder="hf_… (o dejá vacío si usás HF_TOKEN en env)",
                        value=HF_TOKEN,
                        scale=3
                    )
                    btn_repos = gr.Button("↺", variant="secondary", scale=1)
                repo_dd = gr.Dropdown(
                    label="Repositorio destino",
                    choices=[HF_REPO] if HF_REPO else [],
                    value=HF_REPO or None,
                    interactive=True
                )
                gr.HTML('<div class="divider"></div>')

                mode_r = gr.Radio(
                    choices=["Copy (Video Original)", "H.264 4K", "H.265 4K",
                             "H.264 1080p", "H.265 1080p"],
                    value="H.264 4K",
                    label="Modo de Video (Todo el audio pasará a FLAC)"
                )
                with gr.Row():
                    chk_single = gr.Checkbox(label="Audio individual", value=False)
                    chk_sub    = gr.Checkbox(label="Extraer sub .vtt", value=False)

            with gr.Group():
                with gr.Tabs():
                    with gr.Tab("◈ Individual"):
                        src_file = gr.File(label="Archivo", file_count="single")
                        src_url  = gr.Textbox(label="URL", placeholder="https://…")
                        btn_analyze = gr.Button("⟳ Analizar", variant="secondary")
                        info_txt = gr.Textbox(label="Info", interactive=False, lines=1)
                        with gr.Row():
                            aud_dd = gr.Dropdown(label="Audio", choices=[], interactive=True)
                            sub_dd = gr.Dropdown(label="Sub",   choices=[], interactive=True)
                        gr.HTML('<div class="divider"></div>')
                        sin_cname  = gr.Textbox(label="Nombre personalizado (opcional)")
                        sin_ctype  = gr.Radio(choices=["película","serie"], value="película", label="Tipo")
                        with gr.Column(visible=False) as sin_sc:
                            sin_serie  = gr.Textbox(label="Nombre serie", placeholder="Breaking Bad")
                            with gr.Row():
                                sin_season = gr.Number(label="Temporada", value=1, precision=0, minimum=1)
                                sin_ep     = gr.Number(label="Episodio",  value=1, precision=0, minimum=1)
                        sin_ctype.change(fn=lambda c: gr.update(visible=c=="serie"),
                                         inputs=sin_ctype, outputs=sin_sc)
                        btn_single = gr.Button("⬆  PROCESAR Y SUBIR", variant="primary", size="lg")

                    with gr.Tab("◈ Bulk"):
                        bulk_ui = gr.Textbox(
                            label="URLs — una por línea", lines=8,
                            placeholder="https://cdn.ejemplo.com/ep01.mkv\nhttps://cdn.ejemplo.com/ep02.mkv\n…"
                        )
                        bulk_type = gr.Radio(choices=["película","serie"], value="serie", label="Tipo")
                        with gr.Column(visible=True) as bulk_sc:
                            bulk_serie  = gr.Textbox(label="Serie", placeholder="Breaking Bad")
                            with gr.Row():
                                bulk_season   = gr.Number(label="Temporada", value=1, precision=0, minimum=1)
                                bulk_ep_start = gr.Number(label="Ep. inicial", value=1, precision=0, minimum=1)
                            gr.HTML('<div style="color:#08182e;font-size:9px;font-family:monospace;margin:4px 0 6px;">↳ Serie_T1/Ep1.mp4 · Ep2.mp4 · Ep3.mp4 …</div>')
                        bulk_type.change(fn=lambda c: gr.update(visible=c=="serie"),
                                         inputs=bulk_type, outputs=bulk_sc)
                        btn_bulk = gr.Button("⬆  INICIAR BULK JOB", variant="primary", size="lg")

            with gr.Group():
                gr.HTML('<div style="color:#08182e;font-size:8px;font-weight:800;letter-spacing:.18em;text-transform:uppercase;padding:10px 14px 4px;">↺ Recuperar job</div>')
                with gr.Row():
                    rec_in  = gr.Textbox(placeholder="job id  e.g. a1b2c3d4",
                                         show_label=False, scale=3)
                    btn_rec = gr.Button("Recuperar", variant="secondary", scale=1)

        with gr.Column(scale=7):
            with gr.Group():
                gr.HTML('<div style="color:#08182e;font-size:8px;font-weight:800;letter-spacing:.18em;text-transform:uppercase;padding:12px 14px 4px;">Estado</div>')
                panel_out = gr.HTML(value=_idle_html())
                with gr.Row():
                    jid_display = gr.Textbox(
                        label="Job ID — copialo para recuperar si cerrás la ventana",
                        interactive=False, placeholder="—"
                    )
                    btn_cancel = gr.Button("✕ Cancelar", variant="stop")

            timer = gr.Timer(value=2, active=False)

    btn_repos.click(fn=do_load_repos, inputs=[token_in], outputs=[repo_dd])
    btn_analyze.click(fn=do_analyze, inputs=[src_file, src_url],
                      outputs=[aud_dd, sub_dd, info_txt])

    btn_single.click(
        fn=do_single,
        inputs=[token_in, repo_dd, src_file, src_url, mode_r,
                aud_dd, sub_dd, chk_single, chk_sub,
                sin_cname, sin_ctype, sin_serie, sin_season, sin_ep],
        outputs=[_jid, timer],
    ).then(fn=lambda j: j or "—", inputs=[_jid], outputs=[jid_display])

    btn_bulk.click(
        fn=do_bulk,
        inputs=[token_in, repo_dd, bulk_ui, mode_r, chk_single, chk_sub,
                bulk_type, bulk_serie, bulk_season, bulk_ep_start],
        outputs=[_jid, timer],
    ).then(fn=lambda j: j or "—", inputs=[_jid], outputs=[jid_display])

    timer.tick(fn=render_panel, inputs=[_jid], outputs=[panel_out, timer])
    btn_cancel.click(fn=do_cancel, inputs=[_jid], outputs=[])

    def _recover_full(jid_in):
        jid, timer_upd = do_recover(jid_in)
        p, _ = render_panel(jid)
        return jid, p, jid or "—", timer_upd

    btn_rec.click(
        fn=_recover_full,
        inputs=[rec_in],
        outputs=[_jid, panel_out, jid_display, timer]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   # Railway necesita escuchar en todas las interfaces
        server_port=PORT,         # Railway inyecta $PORT
        show_error=True,
    )
