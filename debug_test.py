from ultralytics import YOLO

model = YOLO("best.pt")

results = model("test.jpg", conf=0.1)

for r in results:
    print(r.boxes.cls)
    print(r.boxes.conf)