---
name: setup-tiktok-edit
description: Edit the user's raw clips of a gaming setup / desk setup / PC build into a beat-synced vertical TikTok, Reel or Short, cut to a song they provide, in their established style (wide hook shot, fast detail cuts on the beat, looping wide ending). Use this whenever the user sends or mentions clips plus a song and wants a TikTok/Reel/Short made, says "make one like usual", "edit my setup video", "cut these to the beat", or anything about turning setup footage into a short-form video, even if they don't say "skill" or name the style.
---

# Gaming setup TikTok edit

The user makes short vertical videos showing off gaming setups. They give you **raw clips + a song**; you return a finished 9:16 MP4 whose cuts land on the beat. The style comes from 8 reference TikToks they picked (details in `references/style.md`): open wide, show details in quick 1–2 beat shots with slow camera movement, close on the opening shot so it loops.

## 1. Get the inputs

You need clips and a song. Optional: where in the song to start (e.g. "the drop at 0:42"), target length (default ~16 s), hook text (default none), and a look (default `clean`).

- **Uploaded files**: use them directly.
- **Google Drive folder/link**: list the folder (`parentId = '<folder id>'`; follow `nextPageToken`, results come in pages). The Drive connector only downloads files **under 10 MB**; those come back saved to a JSON file with a base64 `content` field, which you decode with Python. Phone clips are usually far bigger. For those, try `curl -L -o FILE "https://drive.usercontent.google.com/download?id=FILE_ID&export=download&confirm=t"`; if the result is HTML instead of video, the folder is private, so ask the user to set it to "Anyone with the link can view" (Share → General access) and try again. Don't change sharing yourself.
- **TikTok/YouTube links for the song**: these are usually blocked; ask for the audio file (an MP3, or any video with the song in it works, since only its audio is used).

If something essential is missing, ask once, briefly. Don't ask about optional settings; use defaults and mention them at the end.

## 2. Check tools

`ffmpeg` and `ffprobe` must exist. Run `python3 -c "import librosa"`; if it fails, `pip install librosa`. Only if you'll use subject tracking: `python3 -c "import cv2; cv2.TrackerCSRT_create"`; if it fails, `pip install opencv-contrib-python-headless`. (librosa 1.0 can't read MP4/M4A directly; the script converts through ffmpeg, so that's handled.)

## 3. Look at the clips before cutting

The script handles timing, but choosing *which footage goes where* is your job, and it's what makes the edit look intentional. For each clip, make a frame grid and view it:

```bash
ffmpeg -loglevel error -i CLIP -vf "fps=2,scale=144:256,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='%{pts\:hms}':x=4:y=4:fontsize=14:fontcolor=yellow:box=1:boxcolor=black@0.6,tile=10x4" -frames:v 1 grid.jpg
```

Then decide:
- **Hook clip** (goes first, and returns as the last shot for the loop): the widest, most impressive view of the whole setup. Pass it first.
- **Order of the rest**: details that flow — e.g. monitor → keyboard → mouse → PC case → lighting. Alternate close and medium shots so consecutive cuts don't look alike.
- **Start points**: if a clip's best moment isn't at the beginning (shaky start, hand in frame, boring first seconds), append `@SECONDS` to its path, e.g. `case.mp4@3.2`.
- **Grade**: `clean` (bright, white/wood setups), `moody` (dark/RGB setups), `faded` (soft, desaturated aesthetic), `bw`, or `none`. Pick what fits the footage; all shots get the same grade so the video feels like one piece.
- Drop clips that are blurry, badly lit, or off-topic.

## 4. Run the edit

```bash
python3 <skill-dir>/scripts/make_edit.py \
  --song SONG --clips HOOK.mp4 a.mp4 b.mp4@2.5 c.mp4 \
  --out OUTDIR/edit.mp4 [--start 42.5] [--seconds 16] [--grade clean] [--pace fast] \
  [--hook-text "Rate my setup"] [--zoom 0.05] \
  [--stabilize off|light|strong] [--track a.mp4=x,y,w,h [--track-zoom 1.4]]
```

### Stabilization (`--stabilize`)

Off by default. Turn it on when the user asks, or when the frame grids show handheld shake or jitter. `light` removes jitter but keeps the natural camera move; `strong` makes it glide like a gimbal. It crops in slightly to hide the moving edges, and it adds roughly 15–20 s per clip.

### Subject tracking (`--track`)

Keeps one subject (e.g. the mouse, the PC case, a figure on the desk) framed while the camera moves, by panning a tighter virtual crop that follows it. Use it when the user names something to follow, or when a clip's camera move lets the subject drift toward the edge.

1. Pick the frame the shot starts on (the clip's `@` start, or 0.3 s) and render it with a coordinate grid:
   ```bash
   python3 <skill-dir>/scripts/pick_box.py CLIP --at START --out box.jpg
   ```
2. View it and read off a box around the subject as fractions of the frame: `x,y,w,h` (left, top, width, height; the grid lines are every 0.1). Make it snug around the subject but complete.
3. Check it: re-run with `--box x,y,w,h` and look for the green rectangle on the subject.
4. Pass `--track CLIPNAME=x,y,w,h` (repeat for more clips). `--track-zoom` sets how tight the follow framing is: 1.2 loose, 1.4 default, 2.0 tight.

Afterwards check `tracking` in the script output: `track_lost_fraction` above ~0.2 means the tracker lost the subject for part of the shot (it holds the last position), so look at that shot in the preview, and try a better box or a different start point. Tracking adds about 30 s per tracked clip.

iPhone clips are usually 4K HDR (HLG); the script detects HDR and tone-maps it to normal SDR so it doesn't look washed out on TikTok.

Output matches the clips by default (`--res auto --fps auto`): 4K clips give a 2160×3840 video, 60 fps clips give 60 fps. Force a format with `--res 4k|1080` and `--fps 60|30`. A 16 s edit from 4K60 clips takes ~2–3 min to render at 4K60, ~7 min with stabilization and tracking on (about 1 and 2 min at 1080p30). Run long renders in the background.

What it does: finds the beats, picks the loudest bar-aligned section of the song (or starts at `--start`), plans shots (hook 4 beats → 2-beat shots, 1-beat shots on the hardest hits → 4-beat closing shot on the hook clip), cuts each clip on the frame grid so nothing drifts, center-crops to 9:16, adds a slow push-in to every shot (`--zoom 0` turns it off if the footage already moves a lot), and grades.

Outputs next to `--out`:
- `edit.mp4`: final video with the song
- `edit_nosound.mp4`: same video, silent, for adding the official TikTok sound in-app
- `edit_plan.json`: every shot (clip, start, beats, song time) and the song start time
- `edit_preview.jpg`: a frame grid of the result

## 5. Check your work

View `edit_preview.jpg`. Look for: the same framing twice in a row, a shot that's mostly blur or a hand, a bad crop (landscape clips lose their sides), the hook text covering something important. Fix by reordering clips, changing `@` start points or dropping a clip, then re-run. To iterate faster on a 4K project, test with `--res 1080 --fps 30`, then render the final version at full quality.

## 6. Deliver

Send `edit.mp4` (and mention `edit_nosound.mp4`). Tell the user briefly:
- **always** where in the song the video starts and stops, as exact timestamps (`song_section` in the script output, e.g. "1:04.25 – 1:20.22"); the user always wants this
- length, number of cuts, and the BPM
- the TikTok tip: to use TikTok's licensed version of the sound, upload `edit_nosound.mp4`, pick the sound, and set it to start at the time in `tiktok_sound_tip` so the cuts line up
- the defaults you chose (grade, hook text, length) so they can ask for changes

Then take feedback. Common requests map to flags:
- pace → `--pace`: `fast` (default, the user's preference: 2-beat shots with 1-beat shots on hard hits), `hyper` (every beat), `medium` (steady 2 beats), `chill` (4 beats, like the slower aesthetic references)
- different part of the song → `--start`; longer/shorter → `--seconds`
- different look → `--grade`; too much or too little push-in → `--zoom`
- shaky footage → `--stabilize light` (or `strong`); "keep the X in frame / follow the X" → `--track`

If the user gives feedback that should apply to every future video (not just this one), offer to update this skill so it sticks.
