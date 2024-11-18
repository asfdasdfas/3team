import torch
import os
import json
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split  # Train-test split
from yolov5 import detect  # YOLOv5 detection script

# 감정 별 bounding box 색깔 설정 (감정: 총 7가지)
emotion_colors = {
    'happy': (0, 255, 0),       # Green
    'sad': (255, 0, 0),         # Blue
    'angry': (0, 0, 255),       # Red
    'surprised': (255, 255, 0), # Cyan
    'neutral': (128, 128, 128), # Gray
    'fear': (0, 255, 255),      # Yellow
    'disgust': (255, 0, 255)    # Magenta
}

# 데이터셋 경로 설정
dataset_dir = Path("D:/kor_face_ai/real_t")
images_dir = dataset_dir / "images"
labels_dir = Path("D:/kor_face_ai/Validation")  # 라벨링 데이터 경로 설정
output_dir = Path("D:/kor_face_ai/YOLOv5_output")
output_dir.mkdir(parents=True, exist_ok=True)

# 이미지 파일 리스트 생성
all_image_files = list(images_dir.glob("*.jpg"))

# 이미지 파일이 존재하는지 확인
if len(all_image_files) == 0:
    raise ValueError("No images found in the specified directory. Please check the dataset path.")

# Train-Validation 데이터셋 분할 (8:2 비율)
train_files, val_files = train_test_split(all_image_files, test_size=0.2, random_state=42)

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

# 결과 저장 디렉토리 설정
detection_output_dir = output_dir / "detection_results"
detection_output_dir.mkdir(parents=True, exist_ok=True)

# 감정 분석 결과를 바탕으로 bbox 그리기
def draw_bounding_box(image, bbox, emotion):
    color = emotion_colors.get(emotion, (255, 255, 255))
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

# 감정 감지 및 결과 처리
image_files = val_files[:20]
for image_path in image_files:
    # 이미지 로드
    image = cv2.imread(str(image_path))
    
    # YOLOv5 모델을 사용한 감지 실행
    results = model(str(image_path))
    labels, coordinates = results.xyxy[0][:, -1], results.xyxy[0][:, :-1]
    
    # JSON 레이블 파일 로드
    label_file = labels_dir / (image_path.stem + ".json")
    with open(label_file, 'r') as f:
        emotion_data = json.load(f)

    # Bounding Box 그리기
    for i, coords in enumerate(coordinates):
        x1, y1, x2, y2 = map(int, coords[:4])
        detected_emotion = emotion_data.get(str(i), "neutral")
        draw_bounding_box(image, (x1, y1, x2, y2), detected_emotion)

    # 결과 이미지 저장
    output_image_path = detection_output_dir / image_path.name
    cv2.imwrite(str(output_image_path), image)

# 평가 지표 계산
# (accuracy, precision, recall, F1, mAP50)
def calculate_metrics():
    # TODO: 실제 metric 계산을 위한 로직 구현 필요
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    f1_score = 0.0
    map50 = 0.0
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}, mAP50: {map50}")

calculate_metrics()
