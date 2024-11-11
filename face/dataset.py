import json
import os
from torch.utils.data import Dataset
from PIL import Image

class EmotionDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        """
        Args:
            images_dir (str): 이미지가 저장된 디렉터리 경로
            labels_dir (str): 레이블이 저장된 디렉터리 경로 (JSON 파일)
            transform (callable, optional): 이미지에 적용할 변환 함수 (기본값: None)
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform

        # JSON 레이블 파일을 로드
        self.image_paths = []
        self.labels = []
        
        # 레이블 파일이 JSON 형식이므로 이를 읽음
        label_files = os.listdir(labels_dir)
        for label_file in label_files:
            label_path = os.path.join(labels_dir, label_file)
            
            with open(label_path, 'r') as f:
                data = json.load(f)
                for item in data:
                    # JSON 파일에서 사용하는 키를 확인하고 수정
                    image_name = item['filename']  # 'image_name' 대신 'filename'
                    emotion = item['emotion']  # 'emotion' 키를 사용 (예시 구조)
                    self.image_paths.append(image_name)
                    self.labels.append(emotion)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 이미지 경로 가져오기
        img_name = self.image_paths[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # 레이블 가져오기
        label = self.labels[idx]

        # 이미지 변환이 있다면 적용
        if self.transform:
            image = self.transform(image)

        return image, label
