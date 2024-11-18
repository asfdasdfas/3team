import torch
import os
from pathlib import Path
import cv2
import shutil

# 경로 설정 (필요에 따라 변경)
ROOT_DIR = "D:/kor_face_ai/real_t"  # 이미지 폴더 경로  # 이미지 폴더 경로
OUTPUT_DIR = "D:/kor_face_ai/YOLOv5"  # 바운딩 박스 처리가 된 결과 저장 경로 
MODEL_PATH = "D:/git/3team/FACE/yolov5/yolov5s.pt"  # 사전 학습된 YOLOv5 모델 경로  

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, trust_repo=True)

# 결과 저장 폴더 초기화
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 이미지 경로 리스트 생성
image_paths = list(Path(ROOT_DIR).glob("*.jpg"))  # 확장자는 필요에 따라 변경 (ex: jpg, png 등)

# 이미지 순회하면서 바운딩 박스 처리
for img_path in image_paths:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Cannot read image {img_path}")
        continue

    # 모델 추론 수행
    results = model(img)
    
    # 바운딩 박스가 적용된 결과 이미지 생성
    results.render()
    rendered_img = results.imgs[0]  # YOLOv5에서 결과 이미지

    # 결과 이미지 저장
    output_path = os.path.join(OUTPUT_DIR, img_path.name)
    cv2.imwrite(output_path, rendered_img)
    print(f"Processed and saved: {output_path}")

print("처리가 완료되었습니다.")
