#!/usr/bin/env python3
"""Cut gaming-setup clips into a beat-synced 9:16 TikTok.

Shot plan (in beats): hook shot (4 beats) -> 2-beat shots, shortened to 1 beat
where the song hits hardest -> closing shot (4 beats) that reuses the hook clip
so the video loops. Every cut lands exactly on a detected beat. Inside middle shots,
the footage also jump-cuts ahead (--jump, default 0.25 s) on beats, like the teeqzy references.

Usage:
  make_edit.py --song song.mp3 --clips wide.mp4 b.mp4@2.5 c.mp4 --out edit.mp4
               [--start 42.5] [--seconds 16] [--grade clean] [--pace fast]
               [--hook-text "New desk?"] [--stabilize light]
               [--track b.mp4=0.40,0.55,0.20,0.15 [--track-zoom 1.4]]

The first clip is the hook (open + close). Order the rest the way they should
appear; they are cycled through if there are more shots than clips.
--track keeps a subject centered: the box is x,y,w,h as fractions of the 9:16
frame at that clip's start point (get it with pick_box.py).
Writes edit.mp4 (with song), edit_nosound.mp4 (to add the TikTok sound in-app),
edit_plan.json, and edit_preview.jpg (a frame grid for checking the result).
"""
import argparse, json, math, os, subprocess, sys, tempfile, warnings

warnings.filterwarnings("ignore")
import numpy as np
from concurrent.futures import ThreadPoolExecutor

try:
    import librosa
except ImportError:
    sys.exit("librosa is missing: run `pip install librosa` first")

# Output format. By default it matches the clips (set_format): 4K clips -> 2160x3840,
# 60 fps clips -> 60 fps. These defaults are only used by pick_box.py.
FPS = 30
W, H = 1080, 1920
PW, PH = int(W * 1.5), int(H * 1.5)  # working size; extra room so push-ins stay sharp

RESOLUTIONS = {"1080": (1080, 1920), "4k": (2160, 3840)}


def set_format(res, fps):
    global FPS, W, H, PW, PH
    W, H = RESOLUTIONS[res]
    FPS = fps
    # At 1080p we work at 1.5x for sharp zooms; 4K sources have no extra pixels, so work at 4K.
    PW, PH = (int(W * 1.5), int(H * 1.5)) if res == "1080" else (W, H)


def probe_format(path):
    """('4k' or '1080', fps) of a clip, using the short side so rotated phone video works."""
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=width,height,avg_frame_rate", "-of", "csv=p=0", path]).strip().split(",")
    w, h, rate = int(out[0]), int(out[1]), out[2]
    num, _, den = rate.partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 30
    return ("4k" if min(w, h) >= 2160 else "1080"), (60 if fps >= 50 else 30)
# Bold font for --hook-text. On Windows the drive colon must be escaped inside ffmpeg filters.
FONT = ("C\\:/Windows/Fonts/arialbd.ttf" if os.name == "nt"
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

# Video encoder: "cpu" (libx264) or "gpu" (NVIDIA NVENC + CUDA decoding). Set in main().
ENCODER = "cpu"


def codec(kind):
    """ffmpeg codec args. 'inter' = temporary near-lossless files, 'final' = each finished shot."""
    if ENCODER == "gpu":
        if kind == "inter":
            return ["-c:v", "h264_nvenc", "-preset", "p1", "-rc", "constqp", "-qp", "14"]
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0",
                "-spatial_aq", "1"]
    if kind == "inter":
        return ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "12"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]


def gpu_works():
    """True if this ffmpeg can encode with an NVIDIA GPU (tested with a tiny 4K encode)."""
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i",
                        "color=black:s=2160x3840:d=0.2", "-c:v", "h264_nvenc", "-f", "null", "-"],
                       capture_output=True, text=True)
    return r.returncode == 0

# Color grades seen across the reference videos.
GRADES = {
    "none": "",
    "clean": "eq=contrast=1.06:saturation=1.08:brightness=0.01",
    "moody": "eq=contrast=1.12:saturation=0.85:brightness=-0.04:gamma=0.95",
    "faded": "eq=contrast=0.94:saturation=0.7:brightness=0.02,curves=all='0/0.05 1/0.97'",
    "bw": "hue=s=0,eq=contrast=1.1",
    # clean, plus a gentle shift toward warm (golden-hour / wood-desk vibe)
    "warm": "eq=contrast=1.06:saturation=1.1:brightness=0.01,colortemperature=temperature=5400:mix=0.6",
}

# Pace presets: (beats per normal shot, percentile of beat strength that gets a 1-beat shot).
# fast = the user's preferred style; chill = the slower "aesthetic tour" references (~4 beats).
PACES = {"hyper": (1, 100), "fast": (2, 85), "medium": (2, 100), "chill": (4, 100)}

# Stabilization: vidstab smoothing window in frames (higher = steadier, more like a gimbal).
STABILIZE = {"off": 0, "light": 15, "strong": 45}

# HDR -> SDR (bt709) tone mapping, applied after downscaling so 4K clips stay fast.
TONEMAP = ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
           "tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")


def run(cmd, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}\n{r.stderr[-2000:]}")
    return r.stdout


def duration(path):
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                      "-of", "csv=p=0", path]).strip())


def is_hdr(path):
    """iPhones record HLG HDR by default; it looks washed out unless tone-mapped to SDR."""
    t = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=color_transfer", "-of", "csv=p=0", path]).strip()
    return t in ("arib-std-b67", "smpte2084")


def base_filters(hdr):
    """fps, HDR tone-map, fill + center-crop to the 9:16 working size."""
    vf = [f"fps={FPS}"]
    if hdr:
        vf += [f"scale={PW}:{PH}:force_original_aspect_ratio=increase", TONEMAP]
    vf += [f"scale={PW}:{PH}:force_original_aspect_ratio=increase", f"crop={PW}:{PH}", "setsar=1"]
    return vf


def encode(args, out, cwd=None):
    # Intermediate files: fast to write, near-lossless; the final encode happens per shot.
    run(["ffmpeg", "-loglevel", "error", "-y", *args, "-an", "-pix_fmt", "yuv420p",
         *codec("inter"), out], cwd=cwd)


def read_frames(src, w, h):
    """Yield BGR frames decoded by ffmpeg with an explicit bt709 conversion.
    (OpenCV's own decoder uses a different color matrix, which shifts colors.)"""
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", src, "-vf",
                          f"scale={w}:{h}:in_color_matrix=bt709:in_range=tv,format=bgr24",
                          "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    n = w * h * 3
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    p.stdout.close()
    p.wait()


def track_and_crop(src, out, box, zoom):
    """Follow the subject in `box` (x,y,w,h fractions of the first frame) and crop around it."""
    import cv2
    # Pass 1: track on a small copy (fast, and plenty accurate for framing).
    TW, TH = 360, 640
    x, y, w, h = box
    tracker = None
    centers, lost = [], 0  # centers as fractions of the frame
    for frame in read_frames(src, TW, TH):
        if tracker is None:
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, (int(x * TW), int(y * TH), max(8, int(w * TW)), max(8, int(h * TH))))
            centers.append((x + w / 2, y + h / 2))
            continue
        found, (bx, by, bw, bh) = tracker.update(frame)
        if found:
            centers.append(((bx + bw / 2) / TW, (by + bh / 2) / TH))
        else:  # subject lost (left frame / covered): hold the last position
            centers.append(centers[-1])
            lost += 1
    if not centers:
        sys.exit(f"could not read {src} for tracking")
    fw, fh = PW, PH
    c = np.array(centers, dtype=float) * [fw, fh]
    # Smooth over ~0.6 s so the virtual camera glides instead of jittering with the tracker.
    k = int(0.6 * FPS) | 1
    pad = np.pad(c, ((k // 2, k // 2), (0, 0)), mode="edge")
    kern = np.hanning(k) / np.hanning(k).sum()
    c = np.stack([np.convolve(pad[:, i], kern, mode="valid") for i in range(2)], axis=1)
    cw, ch = fw / zoom, fh / zoom
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                          "-s", f"{PW}x{PH}", "-r", str(FPS), "-i", "-",
                          "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
                          "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                          "-color_range", "tv",
                          *codec("inter"), out],
                         stdin=subprocess.PIPE)
    # Pass 2: re-read at full size and crop around the smoothed path.
    for (cx, cy), f in zip(c, read_frames(src, PW, PH)):
        x0 = int(min(max(cx - cw / 2, 0), fw - cw))
        y0 = int(min(max(cy - ch / 2, 0), fh - ch))
        crop = f[y0:y0 + int(ch), x0:x0 + int(cw)]
        p.stdin.write(cv2.resize(crop, (PW, PH), interpolation=cv2.INTER_CUBIC).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        sys.exit("encoding the tracked clip failed")
    return lost / max(1, len(c))


def prep_clip(clip, t_from, t_len, hdr, stab, box, track_zoom, tmp, tag):
    """Make a 30 fps, SDR, 9:16 working copy of `clip` (t_from .. t_from+t_len),
    stabilized/tracked if asked. Only the part the edit uses is processed, to keep it fast."""
    cur = os.path.join(tmp, f"{tag}_a.mp4")
    hw = ["-hwaccel", "cuda"] if ENCODER == "gpu" else []  # GPU decoding of 4K HEVC phone video
    encode([*hw, "-ss", f"{t_from:.3f}", "-t", f"{t_len:.3f}", "-i", clip,
            "-vf", ",".join(base_filters(hdr))], cur)
    info = {}
    if stab:
        # Relative .trf name + cwd=tmp: a full Windows path (C:\...) would break the filter syntax.
        trf = f"{tag}.trf"
        run(["ffmpeg", "-loglevel", "error", "-y", "-i", cur, "-vf",
             f"vidstabdetect=shakiness=6:accuracy=15:result={trf}", "-f", "null", "-"], cwd=tmp)
        nxt = os.path.join(tmp, f"{tag}_b.mp4")
        # optzoom=1 zooms just enough to hide the moving borders stabilization creates.
        encode(["-i", cur, "-vf", f"vidstabtransform=input={trf}:smoothing={stab * FPS // 30}:optzoom=1"
                f":interpol=bicubic,unsharp=5:5:0.5,scale={PW}:{PH},setsar=1"], nxt, cwd=tmp)
        cur = nxt
    if box:
        nxt = os.path.join(tmp, f"{tag}_c.mp4")
        info["track_lost_fraction"] = round(track_and_crop(cur, nxt, box, track_zoom), 3)
        cur = nxt
    return cur, info


def fill_beat_grid(beats, y, sr):
    """The beat tracker often locks on a few seconds late (soft intros) or stops early.
    Extend its grid at the same spacing back to the start of the audio and forward to the
    last audible moment, so short clips of a song (e.g. a TikTok's audio) can be used whole."""
    if len(beats) < 4:
        return beats
    step = float(np.median(np.diff(beats)))
    rms = librosa.feature.rms(y=y)[0]
    rms_t = librosa.times_like(rms, sr=sr)
    audible = rms_t[rms > 0.1 * np.median(rms[rms > 0])]
    last_sound = float(audible[-1]) if len(audible) else len(y) / sr
    head = []
    b = beats[0] - step
    while b >= -0.02:
        head.append(max(0.0, b))
        b -= step
    tail = []
    b = beats[-1] + step
    while b <= last_sound - 0.1:
        tail.append(b)
        b += step
    # Snap the guessed beats onto real drum hits when one is within 80 ms.
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    snap = lambda t: (onsets[np.abs(onsets - t).argmin()]
                      if len(onsets) and np.abs(onsets - t).min() < 0.08 else t)
    head = [snap(t) for t in head[::-1]]
    tail = [snap(t) for t in tail]
    return np.concatenate([head, beats, tail])


def analyze_song(song, tmp):
    wav = os.path.join(tmp, "song.wav")
    run(["ffmpeg", "-loglevel", "error", "-y", "-i", song, "-vn", "-ac", "1", "-ar", "22050", wav])
    y, sr = librosa.load(wav, sr=22050)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    beats = fill_beat_grid(np.asarray(beats), y, sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(onset_env, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    rms_t = librosa.times_like(rms, sr=sr)
    # Per-beat strength: onset peak right at the beat, and loudness over the beat.
    hit = np.array([onset_env[np.abs(times - b).argmin()] for b in beats])
    loud = np.array([rms[(rms_t >= b) & (rms_t < b + 60 / float(np.atleast_1d(tempo)[0]))].mean()
                     if np.any((rms_t >= b)) else 0 for b in beats])
    return float(np.atleast_1d(tempo)[0]), np.asarray(beats), hit, np.nan_to_num(loud), len(y) / sr


def pick_window(beats, loud, n_beats, start):
    """Index of the first beat. Uses --start if given, else the loudest bar-aligned stretch."""
    if start is not None:
        return int(np.abs(beats - start).argmin())
    best, best_i = -1, 0
    for i in range(0, len(beats) - n_beats, 4):
        e = loud[i:i + n_beats].mean()
        if e > best:
            best, best_i = e, i
    return best_i


def plan_shots(hit, i0, n_beats, shot_beats=2, quick_pct=85, hook_beats=4, end_beats=4):
    """Return list of (first_beat_idx, n_beats) covering the window."""
    mid = hit[i0 + hook_beats:i0 + n_beats - end_beats]
    thresh = np.percentile(mid, quick_pct) if len(mid) else 0
    shots = [(i0, hook_beats)]
    b = i0 + hook_beats
    end = i0 + n_beats - end_beats
    while b < end:
        left = end - b
        # A beat the song hits hard gets a quick 1-beat shot.
        if quick_pct < 100 and hit[b] >= thresh:
            shots.append((b, 1))
            b += 1
        else:
            n = min(shot_beats, left)
            shots.append((b, n))
            b += n
    shots.append((end, end_beats))
    return shots


def render_shot(prep, pieces, out, grade, zoom, text=None):
    """Cut one shot from a prepped clip: push-in zoom, grade, optional text.
    `pieces` is [(src_start, n_frames), ...]; more than one piece = jump cuts inside the shot
    (the footage skips ahead on a beat while the shot, and its push-in, carry on)."""
    n_frames = sum(n for _, n in pieces)
    s0 = pieces[0][0]
    if len(pieces) == 1:
        head = ""
    else:
        segs = []
        for i, (ps, n) in enumerate(pieces):
            f0 = int(round((ps - s0) * FPS))
            segs.append(f"[s{i}]trim=start_frame={f0}:end_frame={f0 + n},setpts=PTS-STARTPTS[p{i}]")
        head = (f"[0:v]split={len(pieces)}" + "".join(f"[s{i}]" for i in range(len(pieces))) + ";"
                + ";".join(segs) + ";" + "".join(f"[p{i}]" for i in range(len(pieces)))
                + f"concat=n={len(pieces)}:v=1:a=0,")
    # If the clip runs out early, hold its last frame so the shot still lasts exactly n_frames
    # (a short shot would shift every later cut off the beat).
    vf = ["tpad=stop_mode=clone:stop=-1"]
    if zoom:
        vf += [f"zoompan=z='1+{zoom}*on/{n_frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
              f":d=1:s={W}x{H}:fps={FPS}"]
    else:
        vf += [f"scale={W}:{H}"]
    if GRADES[grade]:
        vf.append(GRADES[grade])
    if text:
        safe = text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’")
        vf.append(f"drawtext=fontfile={FONT}:text='{safe}':fontsize=72:fontcolor=white"
                  f":borderw=4:bordercolor=black@0.7:x=(w-tw)/2:y=h*0.18")
    vf.append("format=yuv420p")
    span = (pieces[-1][0] - s0) + pieces[-1][1] / FPS
    graph = (head + ",".join(vf)) if head else ("[0:v]" + ",".join(vf))
    run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{s0:.3f}", "-i", prep,
         "-t", f"{span + 0.5:.3f}", "-filter_complex", graph + "[v]", "-map", "[v]",
         "-frames:v", str(n_frames),
         "-an", "-pix_fmt", "yuv420p", *codec("final"), out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", required=True)
    ap.add_argument("--clips", nargs="+", required=True,
                    help="first clip = hook / ending; append @SECONDS to choose where a clip starts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=float, help="song time (s) to start at; default = loudest section")
    ap.add_argument("--seconds", type=float, default=16, help="target length (snapped to whole bars)")
    ap.add_argument("--grade", choices=GRADES, default="clean")
    ap.add_argument("--pace", choices=PACES, default="fast")
    ap.add_argument("--zoom", type=float, default=0.05, help="push-in amount per shot, 0 = off")
    ap.add_argument("--jump", type=float, default=0.25,
                    help="jump cuts: seconds of footage skipped on beats inside a shot (0 = off)")
    ap.add_argument("--hook-jump", type=float, default=0,
                    help="jump cuts in the opening shot too (seconds per jump; 0 = clean opener). "
                         "Use it to fit a long action, e.g. a box being cut open, into the hook.")
    ap.add_argument("--jump-every", type=int, default=0,
                    help="beats between jump cuts inside a shot; 0 = auto (half a shot)")
    ap.add_argument("--hook-text", help="optional text over the first shot")
    ap.add_argument("--end", metavar="CLIP[@SECONDS]",
                    help="clip for the closing shot; default = the hook clip (loops). Use it when "
                         "the opener can't double as the ending, e.g. an unboxing.")
    ap.add_argument("--skip", type=float, default=0.3, help="seconds to skip at the start of each clip")
    ap.add_argument("--stabilize", choices=STABILIZE, default="off",
                    help="smooth out camera shake on every clip")
    ap.add_argument("--track", action="append", default=[], metavar="CLIP=x,y,w,h",
                    help="keep a subject centered in CLIP; box as fractions of the 9:16 frame "
                         "at the clip's start point (see pick_box.py). Repeatable.")
    ap.add_argument("--track-zoom", type=float, default=1.4,
                    help="how tight the tracked framing is (1.2 loose .. 2.0 tight)")
    ap.add_argument("--res", choices=["auto", *RESOLUTIONS], default="auto",
                    help="output resolution; auto = match the clips (best clip wins)")
    ap.add_argument("--fps", choices=["auto", "30", "60"], default="auto",
                    help="output frame rate; auto = match the clips")
    ap.add_argument("--encoder", choices=["auto", "cpu", "gpu"], default="auto",
                    help="gpu = NVIDIA NVENC (much faster for 4K); auto = gpu if it works, else cpu")
    a = ap.parse_args()
    global ENCODER
    ENCODER = "gpu" if a.encoder == "gpu" or (a.encoder == "auto" and gpu_works()) else "cpu"

    fmts = [probe_format(c.partition("@")[0]) for c in a.clips]
    res = a.res if a.res != "auto" else ("4k" if any(r == "4k" for r, _ in fmts) else "1080")
    fps = int(a.fps) if a.fps != "auto" else max(f for _, f in fmts)
    set_format(res, fps)

    tmp = tempfile.mkdtemp(prefix="edit_")
    tempo, beats, hit, loud, song_len = analyze_song(a.song, tmp)
    beat_len = 60 / tempo
    n_beats = max(12, int(round(a.seconds / beat_len / 4)) * 4)
    i0 = pick_window(beats, loud, n_beats, a.start)
    if i0 + n_beats >= len(beats):
        n_beats = len(beats) - 1 - i0  # song ends first: use every beat that's left
        if n_beats < 12:
            sys.exit("not enough song after --start for an edit; pick an earlier start")
    shots = plan_shots(hit, i0, n_beats, *PACES[a.pace])
    t0 = beats[i0]
    # Cut points on the frame grid, measured from the first beat, so nothing drifts.
    frame_at = lambda bi: int(round((beats[bi] - t0) * FPS))

    # "clip.mp4@3.5" starts that clip's first shot 3.5s in (pick the best moment).
    clips, cursor = [], {}
    for c in a.clips:
        path, _, at = c.partition("@")
        path = os.path.abspath(path)
        clips.append(path)
        cursor[path] = float(at) if at else a.skip
    first_start = dict(cursor)
    tracks = {}
    for t in a.track:
        name, _, nums = t.partition("=")
        try:
            box = [float(v) for v in nums.split(",")]
        except ValueError:
            box = []
        match = [c for c in clips if os.path.basename(c) == os.path.basename(name)]
        if len(box) != 4 or not match:
            sys.exit(f"bad --track {t!r}: use CLIPNAME=x,y,w,h with a clip from --clips")
        tracks[match[0]] = box

    hook, rest = clips[0], clips[1:] or clips[:1]
    end = hook  # by default the video ends on the opening shot, so it loops
    if a.end:
        path, _, at = a.end.partition("@")
        end = os.path.abspath(path)
        if end not in clips:
            clips.append(end)
        first_start[end] = cursor[end] = float(at) if at else a.skip
    durs = {c: duration(c) for c in clips}
    # Decide which part of which clip every shot uses.
    plan, assign = [], []
    for k, (bi, nb) in enumerate(shots):
        nf = frame_at(bi + nb) - frame_at(bi)
        # Jump cuts (teeqzy style): inside middle shots, on every `every`-th beat the footage
        # skips ahead a.jump seconds instead of cutting to a new clip. Hook and ending stay clean.
        every = a.jump_every or max(1, PACES[a.pace][0] // 2)
        middle = 0 < k < len(shots) - 1
        jump = a.jump if middle else a.hook_jump if k == 0 else 0
        jb = [j for j in range(every, nb, every)] if jump > 0 else []
        bounds = [0] + [frame_at(bi + j) - frame_at(bi) for j in jb] + [nf]
        need = nf / FPS + len(jb) * jump  # footage used, including the skipped bits
        last = k == len(shots) - 1
        clip = hook if k == 0 else end if last else rest[(k - 1) % len(rest)]
        start = cursor[clip]
        if last:
            start = first_start[clip]  # same framing as the opener, so the loop is seamless
        if start + need > durs[clip]:
            start = first_start[clip]  # clip used up: reuse from its start point
        cursor[clip] = start + need
        pieces = [(start + bounds[m] / FPS + m * jump, bounds[m + 1] - bounds[m])
                  for m in range(len(bounds) - 1)]
        assign.append((clip, pieces))
        plan.append({"shot": k, "clip": os.path.basename(clip), "clip_start": round(start, 2),
                     "beats": nb, "seconds": round(nf / FPS, 3), "jump_cuts": len(jb),
                     "song_time": round(float(beats[bi]), 3)})

    # Prepare each clip once (tone-map, crop, stabilize, track) over just the span it's used.
    def prep(i_c):
        i, c = i_c
        span = max(pc[-1][0] + pc[-1][1] / FPS for cc, pc in assign if cc == c) - first_start[c] + 0.5
        return c, prep_clip(c, first_start[c], span, is_hdr(c), STABILIZE[a.stabilize],
                            tracks.get(c), a.track_zoom, tmp, f"clip{i}")
    preps, prep_info = {}, {}
    with ThreadPoolExecutor(2) as pool:  # two clips at a time; ffmpeg is already multi-threaded
        for c, (path, info) in pool.map(prep, enumerate(dict.fromkeys(x[0] for x in assign))):
            preps[c] = path
            if info:
                prep_info[os.path.basename(c)] = info

    parts = []
    for k, (clip, pieces) in enumerate(assign):
        out = os.path.join(tmp, f"shot{k:03d}.mp4")
        render_shot(preps[clip], [(ps - first_start[clip], n) for ps, n in pieces], out,
                    a.grade, a.zoom, a.hook_text if k == 0 else None)
        parts.append(out)

    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{p.replace(os.sep, '/')}'\n" for p in parts)
    base = os.path.splitext(a.out)[0]
    nosound = base + "_nosound.mp4"
    run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
         "-c", "copy", nosound])
    total = frame_at(i0 + n_beats) / FPS
    run(["ffmpeg", "-loglevel", "error", "-y", "-i", nosound, "-ss", f"{t0:.3f}", "-i", a.song,
         "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-af", f"afade=t=out:st={max(0, total - 0.4):.3f}:d=0.4", "-t", f"{total:.3f}",
         "-movflags", "+faststart", a.out])
    # Preview grid: 4 frames per second of the result.
    run(["ffmpeg", "-loglevel", "error", "-y", "-i", nosound, "-vf",
         f"fps=4,scale=180:320,tile=8x{math.ceil(total * 4 / 8)}", "-frames:v", "1",
         base + "_preview.jpg"])
    mmss = lambda t: f"{int(t // 60)}:{t % 60:05.2f}"
    summary = {"tempo_bpm": round(tempo, 1), "song_start": round(float(t0), 3),
               "song_end": round(float(t0) + total, 3),
               "song_section": f"{mmss(t0)} - {mmss(t0 + total)}",
               "length_s": round(total, 2), "format": f"{W}x{H} @ {FPS} fps", "encoder": ENCODER,
               "stabilize": a.stabilize, "shots": plan,
               "tiktok_sound_tip": f"In TikTok, start the sound at {int(t0 // 60)}:{t0 % 60:04.1f}"}
    if prep_info:
        summary["tracking"] = prep_info
    with open(base + "_plan.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "shots"} | {"n_shots": len(plan)}, indent=2))


if __name__ == "__main__":
    main()
