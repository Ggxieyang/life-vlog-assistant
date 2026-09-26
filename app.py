import gradio as gr
import cv2
import os
import glob
from ultralytics import YOLO
from collections import defaultdict
import json

def analyze_video(video_path):
    model = YOLO("yolov8n.pt")

    # Open the video file
    results = model.predict(video_path,save=True,project="outputs",name="result",exist_ok=True,vid_stride=30)

    # 2. 直接去实际目录里找最新的视频文件，不拼接文件名
    search_dir = "runs/detect/outputs/result"

    video_files = glob.glob(os.path.join(search_dir, "*.mp4")) + \
                  glob.glob(os.path.join(search_dir, "*.avi"))

    if video_files:
        # 按修改时间取最新的那个
        output_path = max(video_files, key=os.path.getmtime)
    else:
        output_path = video_path  # 兜底

    frame_dict = defaultdict(list)
    detections = []
    for frame_idx, res in enumerate(results):  # 遍历每一帧
        for box in res.boxes:  # 遍历该帧每个检测框
            cls_id = int(box.cls[0])  # 类别ID（数字）
            cls_name = model.names[cls_id]  # 类别名字（如"person"）
            frame_dict[frame_idx].append(cls_name)

    for frame_idx, objects in frame_dict.items():
        detections.append({
            "timestamp": frame_idx,
            "objects": objects
        })
    with open("detections.jsonl", "w", encoding="utf-8") as f:
        for item in detections:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return output_path,detections


demo = gr.Interface(
    fn=analyze_video,
    inputs=gr.Video(label="上传或拍摄视频",sources=["upload","webcam"]),
    outputs=[
        gr.Video(label="标注后的视频"),
        gr.JSON(label="YOLO检测结果"),
    ],
    title="生活记录小助手"
)

demo.launch(share=True)  # 记得始终加上 share=True


