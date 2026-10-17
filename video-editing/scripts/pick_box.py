#!/usr/bin/env python3
"""Save a frame of CLIP at --at seconds, framed exactly as the edit sees it (9:16 crop,
HDR tone-mapped), with a labeled 10x10 grid so you can read off a subject's box for
make_edit.py --track CLIP=x,y,w,h (fractions of width/height).

Usage: pick_box.py CLIP --at 2.5 --out frame.jpg [--box 0.3,0.4,0.2,0.15]
--box draws a candidate box in green so you can check it before tracking.
"""
import argparse, os, subprocess, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_edit import base_filters, is_hdr  # same framing as the real edit

import cv2

ap = argparse.ArgumentParser()
ap.add_argument("clip")
ap.add_argument("--at", type=float, default=0.3, help="seconds into the clip (use the clip's @ start)")
ap.add_argument("--out", required=True)
ap.add_argument("--box", help="x,y,w,h to draw for checking")
a = ap.parse_args()

raw = os.path.join(tempfile.mkdtemp(), "f.png")
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{a.at:.3f}", "-i", a.clip,
                "-vf", ",".join(base_filters(is_hdr(a.clip))), "-frames:v", "1", raw], check=True)
img = cv2.resize(cv2.imread(raw), (720, 1280))
h, w = img.shape[:2]
for i in range(1, 10):
    x, y = int(w * i / 10), int(h * i / 10)
    cv2.line(img, (x, 0), (x, h), (0, 255, 255), 1)
    cv2.line(img, (0, y), (w, y), (0, 255, 255), 1)
    cv2.putText(img, f".{i}", (x + 3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(img, f".{i}", (x + 3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(img, f".{i}", (3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(img, f".{i}", (3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
if a.box:
    bx, by, bw, bh = (float(v) for v in a.box.split(","))
    cv2.rectangle(img, (int(bx * w), int(by * h)), (int((bx + bw) * w), int((by + bh) * h)),
                  (0, 255, 0), 3)
cv2.imwrite(a.out, img)
print(a.out)
