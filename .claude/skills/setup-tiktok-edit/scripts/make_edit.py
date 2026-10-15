#!/usr/bin/env python3
"""Cut gaming-setup clips into a beat-synced 9:16 TikTok.

Shot plan (in beats): hook shot (4 beats) -> 2-beat shots, shortened to 1 beat
where the song hits hardest -> closing shot (4 beats) that reuses the hook clip
so the video loops. Every cut lands exactly on a detected beat.

Usage:
  make_edit.py --song song.mp3 --clips wide.mp4 b.mp4@2.5 c.mp4 --out edit.mp4
               [--start 42.5] [--seconds 16] [--grade clean] [--hook-text "New desk?"]

The first clip is the hook (open + close). Order the rest the way they should
appear; they are cycled through if there are more shots than clips.
Writes edit.mp4 (with song), edit_nosound.mp4 (to add the TikTok sound in-app),
edit_plan.json, and edit_preview.jpg (a frame grid for checking the result).
"""
import argparse, json, math, os, subprocess, sys, tempfile, warnings

warnings.filterwarnings("ignore")
import numpy as np

try:
    import librosa
except ImportError:
    sys.exit("librosa is missing: run `pip install librosa` first")

FPS = 30
W, H = 1080, 1920
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Color grades seen across the reference videos.
GRADES = {
    "none": "",
    "clean": "eq=contrast=1.06:saturation=1.08:brightness=0.01",
    "moody": "eq=contrast=1.12:saturation=0.85:brightness=-0.04:gamma=0.95",
    "faded": "eq=contrast=0.94:saturation=0.7:brightness=0.02,curves=all='0/0.05 1/0.97'",
    "bw": "hue=s=0,eq=contrast=1.1",
}


# Pace presets: (beats per normal shot, percentile of beat strength that gets a 1-beat shot).
# fast = the user's preferred style; chill = the slower "aesthetic tour" references (~4 beats).
PACES = {"hyper": (1, 100), "fast": (2, 85), "medium": (2, 100), "chill": (4, 100)}


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}\n{r.stderr[-2000:]}")
    return r.stdout


def is_hdr(path):
    """iPhones record HLG HDR by default; it looks washed out unless tone-mapped to SDR."""
    t = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=color_transfer", "-of", "csv=p=0", path]).strip()
    return t in ("arib-std-b67", "smpte2084")


# HDR -> SDR (bt709) tone mapping, applied after downscaling so 4K clips stay fast.
TONEMAP = ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
           "tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")


def duration(path):
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                      "-of", "csv=p=0", path]).strip())


def analyze_song(song, tmp):
    wav = os.path.join(tmp, "song.wav")
    run(["ffmpeg", "-loglevel", "error", "-y", "-i", song, "-vn", "-ac", "1", "-ar", "22050", wav])
    y, sr = librosa.load(wav, sr=22050)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
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


def render_shot(clip, src_start, n_frames, out, grade, zoom, text=None, hdr=False):
    secs = n_frames / FPS
    vf = [f"fps={FPS}"]
    if hdr:
        vf += [f"scale={int(W*1.5)}:{int(H*1.5)}:force_original_aspect_ratio=increase", TONEMAP]
    vf += [
          # Fill 9:16, center crop. Upscale 1.5x first so the zoom is smooth.
          f"scale={int(W*1.5)}:{int(H*1.5)}:force_original_aspect_ratio=increase",
          f"crop={int(W*1.5)}:{int(H*1.5)}",
          "setsar=1"]
    if zoom:
        vf.append(f"zoompan=z='1+{zoom}*on/{n_frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                  f":d=1:s={W}x{H}:fps={FPS}")
    else:
        vf.append(f"scale={W}:{H}")
    if GRADES[grade]:
        vf.append(GRADES[grade])
    if text:
        safe = text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’")
        vf.append(f"drawtext=fontfile={FONT}:text='{safe}':fontsize=72:fontcolor=white"
                  f":borderw=4:bordercolor=black@0.7:x=(w-tw)/2:y=h*0.18")
    vf.append("format=yuv420p")
    run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{src_start:.3f}", "-i", clip,
         "-t", f"{secs + 0.5:.3f}", "-vf", ",".join(vf), "-frames:v", str(n_frames),
         "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", out])


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
    ap.add_argument("--hook-text", help="optional text over the first shot")
    ap.add_argument("--skip", type=float, default=0.3, help="seconds to skip at the start of each clip")
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="edit_")
    tempo, beats, hit, loud, song_len = analyze_song(a.song, tmp)
    beat_len = 60 / tempo
    n_beats = max(12, int(round(a.seconds / beat_len / 4)) * 4)
    i0 = pick_window(beats, loud, n_beats, a.start)
    if i0 + n_beats >= len(beats):
        n_beats = (len(beats) - 1 - i0) // 4 * 4
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
    durs = {c: duration(c) for c in clips}
    hdr = {c: is_hdr(c) for c in clips}
    hook, rest = clips[0], clips[1:] or clips[:1]
    plan, parts = [], []
    for k, (bi, nb) in enumerate(shots):
        nf = frame_at(bi + nb) - frame_at(bi)
        need = nf / FPS
        if k == 0:
            clip = hook
        elif k == len(shots) - 1:
            clip = hook  # loop back to the opening shot
        else:
            clip = rest[(k - 1) % len(rest)]
        start = cursor[clip]
        if k == len(shots) - 1:
            start = first_start[clip]  # same framing as the opener, so the loop is seamless
        if start + need > durs[clip]:
            start = max(0.0, min(first_start[clip], durs[clip] - need))  # clip used up: reuse
        cursor[clip] = start + need
        out = os.path.join(tmp, f"shot{k:03d}.mp4")
        render_shot(clip, start, nf, out, a.grade, a.zoom, a.hook_text if k == 0 else None,
                    hdr[clip])
        parts.append(out)
        plan.append({"shot": k, "clip": os.path.basename(clip), "clip_start": round(start, 2),
                     "beats": nb, "seconds": round(nf / FPS, 3),
                     "song_time": round(float(beats[bi]), 3)})

    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{p}'\n" for p in parts)
    base = os.path.splitext(a.out)[0]
    nosound = base + "_nosound.mp4"
    run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
         "-c", "copy", nosound])
    total = frame_at(i0 + n_beats) / FPS
    run(["ffmpeg", "-loglevel", "error", "-y", "-i", nosound, "-ss", f"{t0:.3f}", "-i", a.song,
         "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-af", f"afade=t=out:st={max(0, total - 0.4):.3f}:d=0.4", "-t", f"{total:.3f}",
         "-movflags", "+faststart", a.out])
    # Preview grid: one frame from the middle of every shot.
    run(["ffmpeg", "-loglevel", "error", "-y", "-i", nosound, "-vf",
         f"fps=4,scale=180:320,tile=8x{math.ceil(total * 4 / 8)}", "-frames:v", "1",
         base + "_preview.jpg"])
    summary = {"tempo_bpm": round(tempo, 1), "song_start": round(float(t0), 3),
               "length_s": round(total, 2), "shots": plan,
               "tiktok_sound_tip": f"In TikTok, start the sound at {int(t0 // 60)}:{t0 % 60:04.1f}"}
    with open(base + "_plan.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "shots"} | {"n_shots": len(plan)}, indent=2))


if __name__ == "__main__":
    main()
