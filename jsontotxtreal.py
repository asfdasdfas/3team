
""""
json파일을 txt파일로 변환
"""
import os
import json
from PIL import Image

def convert_coco_to_yolo(json_path, image_dir, output_dir):
    # JSON 파일 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 출력 폴더 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 각 이미지 항목 처리
    for image_data in data:
        filename = image_data.get('filename')
        if not filename:
            continue
        
        # JSON 파일의 filename에서 뒤 15글자 추출
        filename_suffix = filename[-15:]
        
        # 이미지 폴더 내 파일 탐색
        image_found = False
        for image_name in os.listdir(image_dir):
            # 이미지 파일의 뒤에서 15글자 추출
            image_name_suffix = image_name[-15:]
            
            # 뒤 15글자가 동일한 이미지 찾기
            if filename_suffix == image_name_suffix:
                image_path = os.path.join(image_dir, image_name)
                image_found = True
                break
        
        # 이미지가 존재하지 않으면 넘어감
        if not image_found:
            print(f"Error: Image with filename suffix {filename_suffix} not found in {image_dir}.")
            continue
        
        # 이미지 크기 추출 (Pillow 사용)
        try:
            image = Image.open(image_path)
            image_width, image_height = image.size
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            continue
        
        # 어노테이션 A, B, C 처리
        for annot_key in ['annot_A', 'annot_B', 'annot_C']:
            annot = image_data.get(annot_key, {})
            boxes = annot.get('boxes', {})

            if boxes:
                x_min = boxes.get('minX')
                y_min = boxes.get('minY')
                x_max = boxes.get('maxX')
                y_max = boxes.get('maxY')
                
                if None in [x_min, y_min, x_max, y_max]:
                    continue  # 바운딩 박스 정보가 부족하면 넘어감
                
                # YOLO 형식 변환: x_center, y_center, width, height (정규화)
                x_center = (x_min + x_max) / 2 / image_width
                y_center = (y_min + y_max) / 2 / image_height
                width = (x_max - x_min) / image_width
                height = (y_max - y_min) / image_height
                
                # YOLO 클래스 ID (예: 0으로 설정, 실제 클래스에 맞게 수정)
                class_id = 0
                
                # TXT 파일로 저장
                txt_file_path = os.path.join(output_dir, f"{image_name.split('.')[0]}_{annot_key}.txt")
                with open(txt_file_path, 'a') as f_txt:
                    f_txt.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# 사용 예제
convert_coco_to_yolo(
    json_path="./face/train/label/angry.json",  # JSON 파일 경로 수정
    image_dir="./face/val/images",  # 이미지 폴더 경로
    output_dir="./face/val/labelsabc"  # 변환된 TXT 파일을 저장할 폴더
)
