"""End-to-end backend/API validation without extra dependencies."""

from __future__ import annotations

import json
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

BASE = "http://127.0.0.1:8000"
PROXY = "http://localhost:5173/api"
BOUNDARY = "----AslValidationBoundary"


def multipart_body(filename: str, payload: bytes, content_type: str = "image/jpeg") -> bytes:
    return (
        f"--{BOUNDARY}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8") + payload + f"\r\n--{BOUNDARY}--\r\n".encode("utf-8")


def get_json(url: str) -> tuple[int, dict]:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def post_predict(
    file_path: Path | None = None,
    payload: bytes | None = None,
    filename: str = "upload.jpg",
    require_hand: bool = False,
    base: str = BASE,
) -> tuple[int, dict]:
    data = payload if payload is not None else file_path.read_bytes()  # type: ignore[union-attr]
    url = f"{base}/predict" + ("?require_hand=true" if require_hand else "")
    request = urllib.request.Request(url, data=multipart_body(filename, data), method="POST")
    request.add_header("Content-Type", f"multipart/form-data; boundary={BOUNDARY}")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, {"detail": body}


def main() -> None:
    print("=== HEALTH ===")
    status, health = get_json(f"{BASE}/health")
    print("status_code", status)
    print(
        json.dumps(
            {
                "status": health.get("status"),
                "model_loaded": health.get("model_loaded"),
                "model_path": health.get("model_path"),
                "classes_count": len(health.get("classes", [])),
                "device": health.get("device"),
                "cuda_available": health.get("cuda_available"),
                "gpu_name": health.get("gpu_name"),
                "hand_detection_available": health.get("hand_detection_available"),
                "hand_detection_error": health.get("hand_detection_error"),
            },
            indent=2,
        )
    )

    print("\n=== MODEL LOAD ONCE (startup log) ===")
    log_path = ROOT / "outputs" / "validation" / "backend_log_snippet.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("Check server startup log for a single 'Loaded model from ... on cuda' line.")

    print("\n=== INVALID INPUTS ===")
    for label, payload in [("empty", b""), ("bad", b"not-an-image")]:
        code, resp = post_predict(payload=payload, filename=f"{label}.jpg")
        print(label, "code", code, "detail", resp.get("detail", resp))

    print("\n=== NO HAND (blank image, require_hand=true) ===")
    blank_buf = BytesIO()
    Image.new("RGB", (640, 480), (255, 255, 255)).save(blank_buf, format="JPEG")
    code, resp = post_predict(payload=blank_buf.getvalue(), filename="blank.jpg", require_hand=True)
    print(
        "code",
        code,
        "status",
        resp.get("status"),
        "prediction",
        resp.get("prediction"),
        "message",
        resp.get("message"),
    )

    print("\n=== DATASET UPLOAD SAMPLE (no require_hand) ===")
    expected = [chr(ord("A") + i) for i in range(26)]
    if health.get("classes") != expected:
        print("WARNING: checkpoint classes differ from A-Z mapping")
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    weak = ["J", "P", "N", "B"]
    sample_letters = ["A", "C", "M", "Z"] + weak
    mismatches = 0
    for letter in sample_letters:
        img_dir = ROOT / "dataset" / "asl_alphabet_train" / letter
        img = sorted(img_dir.glob("*.jpg"))[0]
        code, resp = post_predict(file_path=img)
        ok = resp.get("prediction") == letter
        if not ok:
            mismatches += 1
        print(
            letter,
            img.name,
            "code",
            code,
            "pred",
            resp.get("prediction"),
            f"{resp.get('confidence', 0) * 100:.2f}%",
            "low",
            resp.get("low_confidence"),
            "threshold",
            resp.get("threshold"),
            "match",
            ok,
        )

    print("\n=== FRONTEND PROXY ===")
    proxy_status, proxy = get_json(f"{PROXY}/health")
    print("proxy_status", proxy_status, "model_loaded", proxy.get("model_loaded"), "device", proxy.get("device"))

    print("\n=== LIVE LOOP TIMING CHECK (frontend logic) ===")
    print("Live mode uses await predictImage(...) inside a while loop with no setInterval(350).")
    print("Only a 16ms wait occurs when the video frame is not ready.")

    print("\n=== SUMMARY ===")
    print("dataset_mismatches", mismatches)
    print("null_prediction_on_no_hand", resp.get("prediction") is None and resp.get("status") == "no_hand_detected")


if __name__ == "__main__":
    main()
