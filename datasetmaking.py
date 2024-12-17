import os
import glob
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

class DataHandle:
    def __init__(self):
        self.db = []
        self.img = None
        self.X_position = (0, 0)
        self.Y_position = (0, 0)
        
        # 7가지 클래스 정의
        self.emotion = {
            'angry': 0, 
            'Anxiety': 1, 
            'happy': 2,
            'neutrality': 3, 
            'Panic': 4,
            'sad': 5,
            'Wound': 6
        }
        self.new_file_path = ''
    
    def _save_crop_img(self):
        try:
            img = self.img.copy()
            roi = img[
                self.Y_position[0]:self.Y_position[1],
                self.X_position[0]:self.X_position[1],
            ]
            img = cv2.resize(roi, (96, 96), interpolation=cv2.INTER_CUBIC)
            self.img = img  # 컬러 이미지 유지
            return True
        except Exception as e:
            print(f"이미지 자르기 오류: {e}")
            return False
    
    def _detect_face(self, img_path):
        try:
            # 먼저 OpenCV로 시도
            self.img = cv2.imread(img_path)
            
            if self.img is None:
                print(f"{img_path} 이미지를 OpenCV로 불러올 수 없습니다. PIL로 시도합니다.")
                try:
                    img_pil = Image.open(img_path)
                    img_pil = img_pil.convert('RGB')  # RGB로 변환
                    self.img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"{img_path} 이미지를 PIL로도 불러올 수 없습니다. 오류: {e}")
                    return False

            # 얼굴 인식을 위해 CascadeClassifier 사용
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                self.X_position = faces[0][0], faces[0][0] + faces[0][2]
                self.Y_position = faces[0][1], faces[0][1] + faces[0][3]
                print(f"얼굴 인식 성공: {img_path}")
                return True
            else:
                print(f"얼굴을 인식할 수 없습니다: {img_path}")
                return False
        except Exception as e:
            print(f"얼굴 인식 실패: {e}")
            return False
        
    def data_augmentation(self, img):
        def random_noise(x):
            x = x + np.random.normal(size=x.shape) * np.random.uniform(1, 5)
            x = x - x.min()
            x = x / x.max()
            return x * 255.0

        augmentation_dir = 'C:/Users/KNUT/face/augmentation'
        if not os.path.exists(augmentation_dir):
            os.makedirs(augmentation_dir)
        
        img = np.expand_dims(img, axis=0)  # (96, 96, 3) -> (1, 96, 96, 3)

        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.07,
            height_shift_range=0.07,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            preprocessing_function=random_noise
        )

        i = 0
        for _ in datagen.flow(img, batch_size=1, save_to_dir=augmentation_dir, save_prefix="aug", save_format='jpg'):
            i += 1
            if i > 5:
                break

        augmentation_img_list = glob.glob(os.path.join(augmentation_dir, '*.jpg'))
        for item in augmentation_img_list:
            item_name = os.path.basename(item)
            new_item_path = os.path.join('C:/Users/KNUT/face/1202dataset', item_name)

            self.db.append({
                'path': new_item_path,
                'label': item_name.split('_')[0]
            })

            if os.path.exists(item):
                os.rename(item, new_item_path)

    def work(self, img_path, emotion):
        self.new_file_path = f"C:/Users/KNUT/face/1202dataset/{os.path.basename(img_path)}"
        if self._detect_face(img_path) and self._save_crop_img():
            if not os.path.exists(os.path.dirname(self.new_file_path)):
                os.makedirs(os.path.dirname(self.new_file_path))
            
            success = cv2.imwrite(self.new_file_path, self.img)
            if success:
                self.db.append({
                    'path': self.new_file_path,
                    'label': self.emotion[emotion]
                })
                print(f"이미지 저장 성공: {self.new_file_path}")
            else:
                print(f"이미지 저장 실패: {self.new_file_path}")

    def resize_and_reshape(self):
        try:
            img_resized = cv2.resize(self.img, (96, 96))
            return img_resized
        except Exception as e:
            print(f"이미지 리사이즈 및 reshape 오류: {e}")
            return None

if __name__ == '__main__':
    def main():
        dbHandle = DataHandle()
        image_folder = 'C:/Users/KNUT/face/train/images'
        folder_list = glob.glob(os.path.join(image_folder, '*'))
        
        for folder in folder_list:
            img_list = glob.glob(os.path.join(folder, '*.jpg'))
            for img_path in img_list:
                print(f"Processing image: {img_path}")
                emotion = os.path.basename(folder)
                if emotion in dbHandle.emotion:
                    dbHandle.work(img_path, emotion)
                    if dbHandle.img is not None:
                        x = dbHandle.resize_and_reshape()
                        if x is not None:
                            dbHandle.data_augmentation(x)
                    else:
                        print(f"얼굴 인식 실패한 이미지: {img_path}")
        
        print(f"총 {len(dbHandle.db)} 개의 이미지 데이터가 DB에 추가되었습니다.")
        
        output_csv = 'C:/Users/KNUT/face/dataset.csv'
        try:
            pd.DataFrame(dbHandle.db).to_csv(output_csv, index=False)
            print(f"CSV 파일이 저장되었습니다: {output_csv}")
        except Exception as e:
            print(f"CSV 저장 실패: {e}")

    main()
