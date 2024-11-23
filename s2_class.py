import torch
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
from PIL import ImageOps
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, random_split, Dataset, ConcatDataset
import cv2  # OpenCV for visualization purposes only

# YOLOv5 로드합니다.
yolov5_model_path = 'D:/git/3team/FACE/yolov5/yolov5s.pt'  # YOLOv5 모델 경로
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolov5_model_path).to(device)

# 클래스 이름 설정 (YOLOv5 모델의 클래스)
class_names = ['angry', 'Anxiety', 'happy', 'neutrality', 'Panic', 'sad', 'Wound']

# 데이터셋 경로 설정
dataset_path = 'D:/kor_face_ai/Training'  # 원천 데이터셋 경로
labeled_dataset_path = 'D:/kor_face_ai/Validation'  # 라벨링된 데이터셋 경로
output_path = 'D:/kor_face_ai/YOLOv5_output'
os.makedirs(output_path, exist_ok=True)

# 사용자 정의 데이터셋 클래스 정의
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            image = ImageOps.exif_transpose(image)
        except (OSError, IOError) as e:
            print(f"이미지 로드 오류: {image_path}, 오류: {e}")
            return None, -1, image_path
        image = ImageOps.exif_transpose(image)
        label = os.path.basename(os.path.dirname(image_path))
        label_idx = class_names.index(label) if label in class_names else -1

        if self.transform:
            image = self.transform(image)

        return image, label_idx, image_path

# 데이터 전처리 설정 (데이터증가 포함)
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # 모든 이미지를 640x640으로 크기 조정
    #transforms.RandomHorizontalFlip(),  # 활발 수평 뒤집기 증가
    #transforms.RandomRotation(10),  # 10도 이내의 위치 전환 증가
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 밝기, 대비, 발색, 발혜 변화 증가
    transforms.ToTensor(),
])

# 원천 데이터셋과 라벨링된 데이터셋 로드 및 병합
original_dataset = CustomImageDataset(root_dir=dataset_path, transform=transform)
labeled_dataset = CustomImageDataset(root_dir=labeled_dataset_path, transform=transform)
combined_dataset = ConcatDataset([original_dataset, labeled_dataset])

# 데이터셋 분할
train_size = int(0.8 * len(combined_dataset))
val_size = len(combined_dataset) - train_size
train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# YOLOv5를 이용한 감지 수행 및 바운딩 박스 정확도 향상
all_preds = []
all_labels = []
saved_images_count = 0  # 저장된 이미지 수를 추적

iou_threshold = 0.5  # mAP50 계산을 위한 IOU 임계값
correct_detections = 0
all_detections = 0
all_ground_truths = 0

for images, labels, paths in val_loader:
    # None 값을 가진 데이터를 필터링합니다.
    valid_data = [(img, lbl, path) for img, lbl, path in zip(images, labels, paths) if img is not None]
    if not valid_data:
        continue
    images, labels, paths = zip(*valid_data)
    for i in range(len(images)):
        frame = images[i].numpy().transpose((1, 2, 0))  # Tensor를 numpy 이미지로 변환
        frame = (frame * 255).astype(np.uint8)

        # YOLOv5 모델로 감지 수행
        results = model(frame, size=640)  # YOLOv5 모델에 이미지 입력, 크기 조정

        # 감지된 객체의 좌표, 레이블 및 신뢰도 가져오기
        detections = results.xyxy[0].cpu().numpy()

        # RGB에서 BGR로 변환 (OpenCV 사용을 위해)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        ground_truth_boxes = []  # 실제 정답 바운딩 박스 저장
        for *box, conf, cls in detections:
            if int(cls) < len(class_names):  # 클래스 번호가 유효한지 확인
                x1, y1, x2, y2 = map(int, box)
                label = f'{class_names[int(cls)]} {conf:.2f}'

                # 바운딩 박스 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), (155, 0, 0), 4)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (155, 0, 0), 4)

                # 예측값과 레이블 저장
                all_preds.append(int(cls))
                all_labels.append(labels[i].item())

                # 바운딩 박스 좌표 저장 (mAP50 계산용)
                ground_truth_boxes.append((x1, y1, x2, y2))

        # 실제 바운딩 박스와 감지된 바운딩 박스 비교
        for *box, _, cls in detections:
            x1, y1, x2, y2 = map(int, box)
            iou_max = 0
            for gt_box in ground_truth_boxes:
                x1_gt, y1_gt, x2_gt, y2_gt = gt_box
                # IOU 계산
                inter_x1 = max(x1, x1_gt)
                inter_y1 = max(y1, y1_gt)
                inter_x2 = min(x2, x2_gt)
                inter_y2 = min(y2, y2_gt)

                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    box_area = (x2 - x1) * (y2 - y1)
                    gt_box_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
                    union_area = box_area + gt_box_area - inter_area
                    iou = inter_area / union_area
                    iou_max = max(iou_max, iou)

            if iou_max >= iou_threshold:
                correct_detections += 1
            all_detections += 1
        all_ground_truths += len(ground_truth_boxes)

        # 결과 이미지를 저장 (최대 10장만 저장)
        if saved_images_count < 10:
            output_image_path = os.path.join(output_path, os.path.basename(paths[i]))
            cv2.imwrite(output_image_path, frame)
            saved_images_count += 1

# 평가 지표 계산
if all_preds and all_labels:
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f"최종 평가 - 정확도: {accuracy:.4f}, 정밀도: {precision:.4f}, 재현율: {recall:.4f}, F1 점수: {f1:.4f}")

    # mAP50 계산
    if all_detections > 0:
        map50 = correct_detections / all_detections
        print(f"mAP@0.5: {map50:.4f}")
