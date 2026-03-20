from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2
import time
import os
import requests

app = Flask(__name__)

# ================= MODEL CONFIG =================

MODEL_PATH = "best.pt"

FILE_ID = "122HAvIL6ZQkNtj_oxUuVDvzjEwjXLlTy"
DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

# ================= MODEL DOWNLOAD =================

def download_model():

    if os.path.exists(MODEL_PATH):
        print("✅ Model already present")
        return

    print("⬇ Downloading model from Google Drive...")

    session = requests.Session()
    response = session.get(DOWNLOAD_URL, stream=True)

    # handle large file confirmation
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            confirm_url = DOWNLOAD_URL + "&confirm=" + value
            response = session.get(confirm_url, stream=True)

    if response.status_code != 200:
        raise RuntimeError("❌ Model download failed")

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)

    size = os.path.getsize(MODEL_PATH)
    print("Downloaded size:", size)

    if size < 10000000:
        raise RuntimeError("❌ Downloaded file is not valid model")

    print("✅ Model downloaded successfully")


download_model()

# ================= LOAD MODEL =================

print("🚀 Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("✅ Model Loaded")

# ================= CONFIG =================

CONF_TH = 0.30
ELEPHANT_SIZE_TH = 0.05
HUMAN_SIZE_TH = 0.01
VOTE_REQUIRED = 2

# ================= VERIFY =================

def verify_detection(results):

    detections = []

    for r in results:

        if r.boxes is None:
            continue

        H, W = r.orig_shape

        for b in r.boxes:

            conf = float(b.conf[0])
            cls = int(b.cls[0])

            xyxy = b.xyxy[0].cpu().numpy()

            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]

            area = (w * h) / (H * W)

            # elephant rule
            if cls == 0 and conf > CONF_TH and area > ELEPHANT_SIZE_TH:
                detections.append(0)

            # human rule
            if cls == 1 and conf > CONF_TH and area > HUMAN_SIZE_TH:
                detections.append(1)

    return detections

# ================= ROUTES =================

@app.route("/")
def home():
    return "Wildlife RGB AI Running"

@app.route("/predict", methods=["POST"])
def predict():

    start = time.time()

    if "image" not in request.files:
        return jsonify({"error": "no image"}), 400

    try:

        file = request.files["image"].read()

        img = cv2.imdecode(
            np.frombuffer(file, np.uint8),
            cv2.IMREAD_COLOR
        )

        if img is None:
            return jsonify({"error": "invalid image"}), 400

    except Exception as e:
        return jsonify({"error": "decode failed"}), 400

    votes = {0: 0, 1: 0}

    # frame voting
    for _ in range(3):

        results = model(
            img,
            imgsz=416,
            conf=0.25,
            verbose=False
        )

        dets = verify_detection(results)

        for d in dets:
            votes[d] += 1

    label = "uncertain"

    if votes[0] >= VOTE_REQUIRED:
        label = "elephant"

    elif votes[1] >= VOTE_REQUIRED:
        label = "human"

    latency = round(time.time() - start, 3)

    print("Votes:", votes, "Result:", label, "Latency:", latency)

    return jsonify({
        "label": label,
        "votes": votes,
        "latency": latency
    })

# ================= RUN =================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )
