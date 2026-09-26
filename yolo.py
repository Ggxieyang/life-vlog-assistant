import cv2

from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Open the video file
video_path = "vlog.mp4"
results = model.predict(video_path)
with open('bilibiliVideo.json','w',encoding="utf-8") as f:
    for res in results:
        s = res.to_json()
        f.write(s+"\n")
