from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2
import time
import os

app = Flask(__name__)

# ================= MODEL LOAD =================

print("===== Wildlife RGB AI Starting =====")
print("Loading YOLO model...")

model = YOLO("best.pt")

print("✅ Model Loaded Successfully")

# ================= CONFIG =================

CONF_TH = 0.30
ELEPHANT_SIZE_TH = 0.05     # elephant must be large
HUMAN_SIZE_TH = 0.01        # human can be smaller

VOTE_REQUIRED = 2

# ================= VERIFICATION =================

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

            # ---------- ELEPHANT RULE ----------
            if cls == 0:
                if conf > CONF_TH and area > ELEPHANT_SIZE_TH:
                    detections.append(0)

            # ---------- HUMAN RULE ----------
            if cls == 1:
                if conf > CONF_TH and area > HUMAN_SIZE_TH:
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
            return jsonify({"error": "bad image"}), 400

    except:
        return jsonify({"error": "decode failed"}), 400

    # ============ FRAME VOTING ============

    votes = {0: 0, 1: 0}

    for i in range(3):

        results = model(
            img,
            imgsz=416,
            conf=0.25,
            verbose=False
        )

        dets = verify_detection(results)

        for d in dets:
            votes[d] += 1

    print("Votes:", votes)

    # ============ DECISION LOGIC ============

    label = "uncertain"

    if votes[0] >= VOTE_REQUIRED:
        label = "elephant"

    elif votes[1] >= VOTE_REQUIRED:
        label = "human"

    latency = round(time.time() - start, 3)

    print("Result:", label, "Latency:", latency)

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