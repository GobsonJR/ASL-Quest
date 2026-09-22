"""Extended app validation: upload parity, live API behavior, smoothing, timing."""

from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from src.config import CONFIDENCE_THRESHOLD, SMOOTHING_WINDOW
from src.inference.hands import HandDetector
from src.utils.smoothing import TemporalSmoother

BASE = "http://127.0.0.1:8000"
PROXY = "http://localhost:5173/api"
BOUNDARY = "----AslValidationBoundary2"


def multipart_body(filename: str, payload: bytes) -> bytes:
    return (
        f"--{BOUNDARY}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8") + payload + f"\r\n--{BOUNDARY}--\r\n".encode("utf-8")


def post_predict(payload: bytes, filename: str, require_hand: bool = False, base: str = BASE) -> tuple[int, dict]:
    url = f"{base}/predict" + ("?require_hand=true" if require_hand else "")
    request = urllib.request.Request(url, data=multipart_body(filename, payload), method="POST")
    request.add_header("Content-Type", f"multipart/form-data; boundary={BOUNDARY}")
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def main() -> None:
    out_dir = ROOT / "outputs" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"checks": []}

    def record(name: str, ok: bool, detail: str | dict) -> None:
        report["checks"].append({"name": name, "ok": ok, "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    # Frontend proxy
    try:
        with urllib.request.urlopen(f"{PROXY}/health", timeout=15) as response:
            proxy = json.loads(response.read().decode("utf-8"))
        record("frontend_proxy_health", proxy.get("model_loaded") is True, {"device": proxy.get("device")})
    except Exception as exc:
        record("frontend_proxy_health", False, str(exc))

    # Upload parity across letters
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    upload_rows = []
    for letter in letters:
        img = sorted((ROOT / "dataset" / "asl_alphabet_train" / letter).glob("*.jpg"))[0]
        direct_code, direct = post_predict(img.read_bytes(), img.name, require_hand=False, base=BASE)
        proxy_code, proxy = post_predict(img.read_bytes(), img.name, require_hand=False, base=PROXY)
        upload_rows.append(
            {
                "letter": letter,
                "file": img.name,
                "direct_pred": direct.get("prediction"),
                "proxy_pred": proxy.get("prediction"),
                "direct_conf": direct.get("confidence"),
                "proxy_conf": proxy.get("confidence"),
                "match": direct.get("prediction") == letter,
                "parity": direct.get("prediction") == proxy.get("prediction"),
            }
        )
    parity_failures = [row for row in upload_rows if not row["parity"]]
    wrong = [row for row in upload_rows if not row["match"]]
    record("upload_direct_vs_proxy_parity", len(parity_failures) == 0, {"failures": parity_failures[:5]})
    record(
        "upload_letter_accuracy_first_image",
        len(wrong) <= 2,
        {"wrong": wrong, "note": "J/B are known weak classes; low-confidence still counts as wrong label"},
    )

    # require_hand on cropped dataset image (expected: may fail detection)
    sample = (ROOT / "dataset" / "asl_alphabet_train" / "A" / "A1.jpg").read_bytes()
    _, hand_resp = post_predict(sample, "A1.jpg", require_hand=True, base=BASE)
    record(
        "require_hand_on_cropped_dataset_image",
        hand_resp.get("status") in {"ok", "no_hand_detected"},
        {
            "status": hand_resp.get("status"),
            "prediction": hand_resp.get("prediction"),
            "message": hand_resp.get("message"),
        },
    )

    # Hand detector availability
    detector = HandDetector(static_image_mode=True)
    record("hand_landmarker_available", detector.available, detector.error or "available")
    blank = Image.new("RGB", (640, 480), (255, 255, 255))
    blank_crop = detector.crop_pil(blank)
    record("blank_no_hand", not blank_crop.detected, {"box": blank_crop.box})
    detector.close()

    # Smoothing behavior
    smoother = TemporalSmoother(window=SMOOTHING_WINDOW, threshold=CONFIDENCE_THRESHOLD)
    votes = []
    for label, conf in [("A", 0.95), ("A", 0.92), ("A", 0.91), ("B", 0.95), ("B", 0.94), ("B", 0.93), ("B", 0.96)]:
        if conf < CONFIDENCE_THRESHOLD:
            smoother.reset()
            continue
        stable, _, _ = smoother.update(label, conf)
        if stable:
            votes.append(stable)
    record("temporal_smoothing_majority", len(votes) >= 1, {"stable_samples": votes[:5]})

    # Sequential request timing (simulates live loop waiting for previous response)
    frame = (ROOT / "dataset" / "asl_alphabet_train" / "M" / "M1.jpg").read_bytes()
    intervals = []
    t0 = time.perf_counter()
    for _ in range(5):
        start = time.perf_counter()
        post_predict(frame, "live.jpg", require_hand=True, base=BASE)
        intervals.append(time.perf_counter() - start)
    record(
        "live_request_sequential_timing",
        max(intervals) < 2.5 and statistics.mean(intervals) > 0.05,
        {
            "intervals_sec": [round(v, 3) for v in intervals],
            "mean_sec": round(statistics.mean(intervals), 3),
            "note": "No fixed 350ms delay; each request waits for backend round-trip",
        },
    )

    # Threshold constant
    record(
        "confidence_threshold_constant",
        CONFIDENCE_THRESHOLD == 0.70,
        {"threshold": CONFIDENCE_THRESHOLD},
    )

    report["upload_rows"] = upload_rows
    report["wrong_predictions"] = wrong
    (out_dir / "app_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out_dir / 'app_validation_report.json'}")


if __name__ == "__main__":
    main()
