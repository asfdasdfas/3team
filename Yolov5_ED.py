# 필요 라이브러리 임포트
import os
import json
import cv2
import torch
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

# 감정별 색상 설정
colors = {
    'angry': (0, 0, 255),        # 빨강
    'anxiety': (255, 165, 0),     # 주황
    'happy': (0, 255, 0),         # 초록
    'neutrality': (255, 255, 0),  # 노랑
    'panic': (255, 0, 0),         # 파랑
    'sad': (128, 0, 128),         # 보라
    'wound': (0, 255, 255)        # 청록
}

# 결과 저장 폴더 설정
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)

# 이미지 경로 설정
image_folder = 'D:/kor_face_ai/real_t'  # 원본 이미지 경로
label_folder = 'D:/kor_face_ai/Validation'  # 라벨링 데이터 경로
os.makedirs(label_folder, exist_ok=True)

image_paths = list(Path(image_folder).rglob('*.jpg'))

# 데이터셋 train/validation 분할
train_image_paths, val_image_paths = train_test_split(image_paths, test_size=0.2, random_state=42)

# JSON 라벨 파일 로드 및 YOLOv5 형식으로 변환
def convert_json_to_yolo(json_path, output_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        for item in data:
            image_name = item['image']
            annotations = item['annotations']
            with open(os.path.join(output_path, image_name.replace('.jpg', '.txt')), 'w') as out_f:
                for annotation in annotations:
                    class_id = annotation['class_id']
                    bbox = annotation['bbox']
                    x_center = (bbox[0] + bbox[2]) / 2
                    y_center = (bbox[1] + bbox[3]) / 2
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    out_f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# 라벨링 데이터 변환 및 저장
json_label_path = 'annotations.json'  # JSON 라벨링 파일 경로
convert_json_to_yolo(json_label_path, label_folder)

# 감정 감지 및 결과 이미지 저장
def detect_emotions(image_paths, model):
    for img_path in tqdm(image_paths):
        img = Image.open(img_path)
        results = model(img, size=640)
        
        # 바운딩 박스 그리기 및 결과 저장
        img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for *xyxy, conf, cls in results.xyxy[0]:
            label = model.names[int(cls)]
            color = colors.get(label, (255, 255, 255))
            cv2.rectangle(img_cv2, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
            cv2.putText(img_cv2, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # 결과 저장
        save_path = os.path.join(output_folder, os.path.basename(img_path))
        cv2.imwrite(save_path, img_cv2)

# 감정 감지 실행 (train 데이터)
detect_emotions(train_image_paths, model)

# YOLOv5 학습 실행 (train 데이터셋과 validation 데이터셋)
train_command = "python train.py --img 640 --batch 16 --epochs 50 --data dataset.yaml --weights yolov5s.pt --name emotion_detection"
os.system(train_command)

# 성능 지표 계산
def calculate_metrics(model, data_loader):
    # Precision, Recall, F1-score, mAP50 등의 성능 평가 지표 계산을 위한 함수 (예시)
    # 구체적인 구현은 데이터셋 및 평가 방식에 따라 달라짐
    pass

print("감정 감지 및 결과 저장 완료!")
