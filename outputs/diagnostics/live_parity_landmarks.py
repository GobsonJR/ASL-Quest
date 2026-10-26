"""Does the running server's recognizer=landmarks answer match the offline evaluation? (read-only, diagnostic)"""
import io, json, sys, uuid, urllib.request
sys.path.insert(0, "d:/SIGN LANGUAGE/scripts"); sys.path.insert(0, "d:/SIGN LANGUAGE")
from collections import Counter
from PIL import Image
from _diag_common import load_webcam_rows, SERVER
from src.landmarks.classifier import LandmarkClassifier

def post(data: bytes, query: str) -> dict:
    b = uuid.uuid4().hex
    body = (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="x.jpg"\r\nContent-Type: image/jpeg\r\n\r\n').encode() + data + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(f"{SERVER}/predict?{query}", body, {"Content-Type": f"multipart/form-data; boundary={b}"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())

rows = load_webcam_rows(thin=True)
clf = LandmarkClassifier("d:/SIGN LANGUAGE/models/experiments/landmark_L2_s0.joblib")
mp_hand = {"Left": "right", "Right": "left"}
res = Counter(); conf_diff = []; q72 = Counter(); resnet_default_ok = 0
for r in rows:
    off = clf.predict_points(r["landmarks"], mp_hand.get(r["handedness_raw"]), top_k=1)
    raw = r["frame_path"].read_bytes()
    live = post(raw, "recognizer=landmarks&require_hand=true&top_k=1")
    res[live["status"]] += 1
    if live["status"] == "ok":
        res["same_top1"] += live["prediction"] == off["prediction"]
        conf_diff.append(abs(live["confidence"] - off["confidence"]))
    buf = io.BytesIO(); Image.open(r["frame_path"]).convert("RGB").save(buf, "JPEG", quality=72)
    l72 = post(buf.getvalue(), "recognizer=landmarks&require_hand=true&top_k=1")
    q72[l72["status"]] += 1
    q72["same_top1_as_offline"] += l72.get("prediction") == off["prediction"]
    d = post(raw, "require_hand=true&top_k=1")  # default request: must stay the ResNet and carry no landmark fields
    resnet_default_ok += ("recognizer" not in d) and ("landmark_model" not in d)
print("frames", len(rows)); print("live (q92 original bytes):", dict(res), "max conf diff", round(max(conf_diff), 6))
print("live (re-encoded at the camera's q72):", dict(q72)); print("default requests unchanged (no recognizer fields):", resnet_default_ok, "/", len(rows))
