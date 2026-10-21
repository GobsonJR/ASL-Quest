"""Analyse REAL captured webcam predictions (see backend/services/live_diagnostics.py).

Start the backend with ASL_QUEST_DEV_DIAGNOSTICS=1, run the frontend with `npm run dev`, hold the sign in
Practice for a few seconds, then run this script. For every saved bundle it performs the four-way comparison:

  A  the ORIGINAL frame through the actual live pipeline   (POST /predict?require_hand=true)
  B  the exact crop the classifier received                (POST /predict, no detection)
  C  that same crop MIRRORED left-right
  D  known, correctly labelled dataset images of the target letter through the same server

and prints the top-5 classes and scores for each, whether the replay of A matches what was recorded when the
frame was captured, and per-bundle montages (frame + box | crop | model input | mirrored crop | dataset
example) saved next to the bundle for looking at the handshape.

    venv311\\Scripts\\python.exe scripts\\analyze_live_requests.py --target C --last 12

Needs only the running backend (no torch here). The classification calls use lossless PNG so the server sees
exactly the saved pixels.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATASET_DIR, OUTPUTS_DIR  # noqa: E402

CRLF = bytes([13, 10])


def classify(base: str, image: Image.Image, require_hand: bool) -> dict:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "PNG")
    boundary = uuid.uuid4().hex
    head = (
        f"--{boundary}".encode() + CRLF
        + b'Content-Disposition: form-data; name="file"; filename="x.png"' + CRLF
        + b"Content-Type: image/png" + CRLF + CRLF
    )
    body = head + buffer.getvalue() + CRLF + f"--{boundary}--".encode() + CRLF
    url = f"{base}/predict?require_hand={'true' if require_hand else 'false'}&top_k=5"
    request = urllib.request.Request(url, body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.load(urllib.request.urlopen(request, timeout=60))


def top5(reply: dict) -> list[tuple[str, float]]:
    return [(t["label"], t["confidence"]) for t in reply.get("top_predictions", [])]


def fmt(reply: dict) -> str:
    if reply.get("status") not in (None, "ok"):
        return f"<{reply['status']}>"
    return "  ".join(f"{label}:{conf:.2f}" for label, conf in top5(reply))


def dataset_examples(letter: str, count: int = 3) -> list[Path]:
    paths = [DATASET_DIR / letter / f"{letter}{n}.jpg" for n in range(2551, 3001, 30)]  # held-out numbers
    return [p for p in paths if p.exists()][:count]


def montage(frame_overlay: Image.Image, crop: Image.Image, model_input: Image.Image, ref: Image.Image, out: Path) -> None:
    size = 260
    tiles = [
        ("frame + box (red) / window (green)", frame_overlay),
        ("classifier crop", crop),
        ("model input 224 (un-normalised)", model_input),
        ("crop MIRRORED", ImageOps.mirror(crop)),
        ("dataset example", ref),
    ]
    canvas = Image.new("RGB", (size * len(tiles), size + 22), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (title, img) in enumerate(tiles):
        canvas.paste(ImageOps.contain(img.convert("RGB"), (size, size)), (i * size, 22))
        draw.text((i * size + 4, 4), title, fill=(0, 0, 0))
    canvas.save(out, quality=92)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--dir", type=Path, default=OUTPUTS_DIR / "diagnostics" / "live_requests")
    ap.add_argument("--target", default=None, help="only bundles captured while practicing this letter")
    ap.add_argument("--last", type=int, default=12, help="analyse the newest N matching bundles")
    args = ap.parse_args()

    bundles = sorted(p for p in args.dir.glob("*") if (p / "result.json").exists())
    records = []
    for folder in bundles:
        record = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        if args.target and (record.get("client", {}).get("target") or "").upper() != args.target.upper():
            continue
        records.append((folder, record))
    records = records[-args.last :]
    if not records:
        sys.exit(f"No captured bundles in {args.dir}. Start the backend with ASL_QUEST_DEV_DIAGNOSTICS=1 and practice in the app.")

    target = (args.target or records[-1][1]["client"].get("target") or "C").upper()
    health = json.load(urllib.request.urlopen(f"{args.server}/health", timeout=20))
    print(f"server model: {health['model_path']}")
    print(f"bundles analysed: {len(records)}  (target letter: {target})\n")

    first = records[-1][1]
    print(f"checkpoint recorded with the capture: {first['checkpoint']}\n  sha256 {first['checkpoint_sha256'][:16]}...  class order A-Z: {first['class_order_is_a_to_z']}  device: {first['device']}")
    print(f"client context: {first['client']}\n")

    # ---- D: known correctly-labelled dataset images of the target letter ----
    print(f"D. dataset {target} images (held-out) through the same server:")
    refs = dataset_examples(target)
    for path in refs:
        print(f"   {path.name:10s} {fmt(classify(args.server, Image.open(path), False))}")

    tallies = {"A": Counter(), "B": Counter(), "C": Counter()}
    hits = {"A": 0, "B": 0, "C": 0}
    replay_match = 0
    for index, (folder, record) in enumerate(records):
        frame = Image.open(next(folder.glob("1_original_frame.*")))
        crop = Image.open(folder / "3_classifier_crop.png")
        overlay = Image.open(folder / "2_hand_box_overlay.png")
        model_in = Image.open(folder / "4_model_input_224.png")
        a = classify(args.server, frame, True)
        b = classify(args.server, crop, False)
        c = classify(args.server, ImageOps.mirror(crop), False)
        for key, reply in (("A", a), ("B", b), ("C", c)):
            if reply.get("prediction"):
                tallies[key][reply["prediction"]] += 1
                hits[key] += reply["prediction"] == target
        replay_match += bool(a.get("prediction")) and a["prediction"] == record["prediction"]
        print(f"\n--- {folder.name}  frame {record['frame_size']}  crop {record['classifier_crop_size']}  "
              f"hand box {record['landmark_box']}  recorded: {record['prediction']} {record['confidence']:.2f}")
        print(f"A original frame (live pipeline) {fmt(a)}")
        print(f"B classifier crop                {fmt(b)}")
        print(f"C crop MIRRORED                  {fmt(c)}")
        if refs:
            montage(overlay, crop, model_in, Image.open(refs[index % len(refs)]), folder / "montage.jpg")

    n = len(records)
    print("\n================ summary over the captured requests ================")
    print(f"A replay reproduces what the app was told when captured: {replay_match}/{n}")
    for key, name in (("A", "original frame (live pipeline)"), ("B", "classifier crop"), ("C", "crop mirrored")):
        print(f"{key} {name:32s} top-1 = {target}: {hits[key]}/{n}   top-1 counts: {dict(tallies[key].most_common(5))}")
    crops = [np.array(Image.open(f / "3_classifier_crop.png").convert("L")).mean() for f, _ in records]
    refs_mean = [np.array(Image.open(p).convert("L")).mean() for p in refs] or [float("nan")]
    print(f"mean brightness  live crops {np.mean(crops):.0f}   dataset {target} {np.mean(refs_mean):.0f}")
    print(f"montages (frame/crop/model input/mirrored/dataset example) saved as montage.jpg inside each bundle folder")


if __name__ == "__main__":
    main()
