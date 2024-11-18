# yolov5_emotion_detection.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import os
from concurrent.futures import ThreadPoolExecutor
import warnings
import subprocess
from PIL import Image

# 경고 메시지 무시 (손상된 EXIF 데이터 등)
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")

# 데이터셋 경로 (절대 경로로 수정)
DATASET_PATH = 'D:/kor_face_ai/real_t'  # 모든 데이터가 있는 폴더 경로
VIDEO_DATASET_PATH = 'D:/kor_face_ai/real_t'  # 비디오 데이터셋 경로
OUTPUT_PATH = 'D:/kor_face_ai/YOLOv5'  # 비디오 검출 결과 저장 경로

# YOLOv5 모델 경로 및 detect.py 경로 설정
yolov5_path = 'D:/git/3team/FACE/yolov5/yolov5s.pt'  # YOLOv5의 절대 경로
detect_script = os.path.join(yolov5_path, 'detect.py')

# 데이터 전처리 및 증강
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(30),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 데이터셋 불러오기
dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 클래스 수 확인
class_names = dataset.classes
num_classes = len(class_names)
print("클래스 수:", num_classes)

# YOLOv5 얼굴 검출 함수 정의
def detect_faces_yolo(image_path, output_dir='yolov5_output', weights='yolov5s.pt', img_size=640, conf_thresh=0.25):
    os.makedirs(output_dir, exist_ok=True)
    command = [
        'python', detect_script,
        '--weights', weights,
        '--img', str(img_size),
        '--conf', str(conf_thresh),
        '--source', image_path,
        '--project', output_dir,
        '--exist-ok'
    ]
    subprocess.run(command, check=True)

# 비디오 파일에 얼굴 검출을 적용하는 함수 정의
def detect_faces_in_video(video_path, output_dir='yolov5_output', weights='yolov5s.pt', img_size=640, conf_thresh=0.25):
    os.makedirs(output_dir, exist_ok=True)
    command = [
        'python', detect_script,
        '--weights', weights,
        '--img', str(img_size),
        '--conf', str(conf_thresh),
        '--source', video_path,
        '--project', output_dir,
        '--exist-ok'
    ]
    subprocess.run(command, check=True)

# 비디오 데이터셋에서 얼굴 검출 수행 및 저장
def detect_faces_in_video_dataset(video_dataset_path, output_dir='video_detection_results', weights='yolov5s.pt', img_size=640, conf_thresh=0.25):
    os.makedirs(output_dir, exist_ok=True)
    for video_file in os.listdir(video_dataset_path):
        if video_file.endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(video_dataset_path, video_file)
            detect_faces_in_video(video_path, output_dir, weights, img_size, conf_thresh)

# 감정 분류 모델 정의
class EmotionClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EmotionClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3 * 224 * 224, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# 모델을 GPU로 이동 (가능한 경우)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
emotion_model = EmotionClassifier(num_classes).to(device)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(emotion_model.parameters(), lr=0.0001)

# 모델 학습 함수 정의
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=30):
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        train_acc_history.append(epoch_acc.item())

        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        val_acc = val_corrects.double() / len(val_dataset)
        val_acc_history.append(val_acc.item())

        print(f'Epoch {epoch+1}/{num_epochs}, Training Accuracy: {epoch_acc:.4f}, Validation Accuracy: {val_acc:.4f}')

    return train_acc_history, val_acc_history

# 모델 학습
train_acc_history, val_acc_history = train_model(emotion_model, criterion, optimizer, train_loader, val_loader, num_epochs=30)

# 학습 및 검증 정확도 그래프 그리기
plt.plot(train_acc_history, label='Training Accuracy')
plt.plot(val_acc_history, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# 최종 모델 평가
def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # numpy 배열로 변환
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 평가 지표 계산
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return accuracy, precision, recall, f1

# 최종 평가 수행
accuracy, precision, recall, f1 = evaluate_model(emotion_model, val_loader, device)
print(f"Final Evaluation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# 바운딩 박스 결과 저장
save_dir = "bounding_box_results"
os.makedirs(save_dir, exist_ok=True)
for i, (inputs, _) in enumerate(val_loader):
    if i >= 5:
        break
    image_path = os.path.join(save_dir, f"input_image_{i}.jpg")
    transforms.ToPILImage()(inputs[0]).save(image_path)
    detect_faces_yolo(image_path, save_dir)

# 비디오 데이터셋에서 얼굴 검출 수행 및 저장
detect_faces_in_video_dataset(VIDEO_DATASET_PATH, output_dir=OUTPUT_PATH)

print(f"비디오 데이터셋의 얼굴 검출 결과가 '{OUTPUT_PATH}' 디렉터리에 저장되었습니다.")
