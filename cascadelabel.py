import cv2
import os
from PIL import Image
import numpy as np
import urllib.parse  # 특수문자 URL 인코딩 처리

# Haar Cascade 얼굴 검출기 모델 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 이미지 폴더 경로
image_folder = 'C:/Users/KNUT/face/train/images'  # images 폴더 경로
output_folder = 'C:/Users/KNUT/face/train/casecaderesult3'  # 결과 이미지 저장 폴더
label_folder = 'C:/Users/KNUT/face/train/labels'  # 레이블 저장 폴더

# 결과 폴더가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if not os.path.exists(label_folder):
    os.makedirs(label_folder)

# 폴더 내 모든 파일 처리
for filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, filename)

    # 이미지가 아닌 파일을 건너뛰기
    if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    # 이미지 경로 출력 (디버깅용)
    print(f"이미지 경로: {image_path}")

    try:
        # 이미지 불러오기
        image = cv2.imread(image_path)
        
        # 이미지가 None이면, 다른 방법으로 열어보려고 시도
        if image is None:
            print(f"{filename} 이미지를 OpenCV로 불러올 수 없습니다. PIL로 시도합니다.")
            try:
                image_pil = Image.open(image_path)  # PIL로 이미지 열기
                image_pil = image_pil.convert('RGB')  # RGB로 변환 (투명한 배경을 처리하기 위함)
                image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
                print(f"PIL로 이미지 변환 완료: {filename}")
            except Exception as e:
                print(f"{filename} 이미지를 PIL로도 불러올 수 없습니다. 오류: {e}")
                continue  # PIL로도 불러올 수 없는 경우 건너뛰기

        # 그레이스케일로 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 얼굴 검출
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 얼굴에 바운딩 박스 그리기
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 저장할 파일 이름 처리 (특수 문자 URL 인코딩)
        safe_filename = urllib.parse.quote(filename)  # 특수 문자를 URL 인코딩으로 처리
        output_path = os.path.join(output_folder, f"result_{safe_filename}")
        
        # 이미지 저장
        if cv2.imwrite(output_path, image):
            print(f"이미지 저장 성공: {output_path}")
        else:
            print(f"이미지 저장 실패: {output_path}")

        # 레이블 파일 저장
        label_path = os.path.join(label_folder, f"{os.path.splitext(safe_filename)[0]}.txt")
        print(f"레이블 저장 경로: {label_path}")
        
        with open(label_path, 'w') as label_file:
            for (x, y, w, h) in faces:
                label_file.write(f"{x} {y} {w} {h}\n")

        print(f"{filename} 얼굴 검출 완료, 레이블 파일 저장됨.")

    except Exception as e:
        print(f"{filename} 이미지를 처리할 수 없습니다. 오류: {e}")

print("얼굴 검출 완료. 결과와 레이블 파일이 저장되었습니다.")
