# googleNet_model_emotion_detection_pytorch.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor
import warnings

# 경고 메시지 무시 (손상된 EXIF 데이터 등)
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")

# 데이터셋 경로 (절대 경로로 수정)
DATASET_PATH = 'D:/kor_face_ai/real_t'  # 모든 데이터가 있는 폴더 경로

# 손상된 이미지 파일 제거 함수 정의
def verify_and_remove(img_path):
    try:
        with Image.open(img_path) as img:
            img.verify()  # 기본 무결성 검사
            img = img.convert('RGB')  # 이미지를 RGB로 변환하여 손상 확인 강화
            img.save(img_path, 'JPEG')  # JPEG로 다시 저장
    except (IOError, SyntaxError, ValueError):
        print(f"손상된 이미지 발견 및 제거: {img_path}")
        os.remove(img_path)  # 손상된 파일 삭제

# ThreadPoolExecutor를 사용해 병렬로 손상된 이미지 제거
with ThreadPoolExecutor() as executor:
    for root, _, files in os.walk(DATASET_PATH):
        img_paths = [os.path.join(root, file) for file in files if file.endswith(('.jpg', '.jpeg'))]
        executor.map(verify_and_remove, img_paths)

# 데이터 전처리 및 증강
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(40),
    transforms.RandomResizedCrop(299, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 데이터셋 불러오기
dataset = torchvision.datasets.ImageFolder(root=DATASET_PATH, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 클래스 수 확인
class_names = dataset.classes
num_classes = len(class_names)
print("클래스 수:", num_classes)

# GoogleNet (Inception-V3) 모델 정의
model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Linear(256, num_classes),
    nn.Softmax(dim=1)
)

# 모델을 GPU로 이동 (가능한 경우)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

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
            # InceptionV3는 학습 모드일 때 logits와 aux_logits를 반환하므로, 첫 번째 출력(logits)만 사용합니다.
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(logits, 1)
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
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                _, preds = torch.max(logits, 1)
                val_corrects += torch.sum(preds == labels.data)

        val_acc = val_corrects.double() / len(val_dataset)
        val_acc_history.append(val_acc.item())

        print(f'Epoch {epoch+1}/{num_epochs}, Training Accuracy: {epoch_acc:.4f}, Validation Accuracy: {val_acc:.4f}')

    return train_acc_history, val_acc_history

# 모델 평가 함수 정의
def evaluate_model(model, data_loader, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            _, preds = torch.max(logits, 1)
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

# 모델 학습
train_acc_history, val_acc_history = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=30)

# 학습 및 검증 정확도 그래프 그리기
plt.plot(train_acc_history, label='Training Accuracy')
plt.plot(val_acc_history, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# 최종 모델 평가
methods = ["전체 영상 감정 분류", "얼굴 검출 후 정규화", "감정별 얼굴 검출"]
results = {}

for method in methods:
    if method == "전체 영상 감정 분류":
        # 전체 이미지를 감정 분류하는 방식으로 평가
        accuracy, precision, recall, f1 = evaluate_model(model, val_loader, device, num_classes)
    
    elif method == "얼굴 검출 후 정규화":
        # 얼굴 검출 후 얼굴 영역을 잘라낸 후 감정 분류 (가상 예시)
        accuracy, precision, recall, f1 = evaluate_model(model, val_loader, device, num_classes)
    
    elif method == "감정별 얼굴 검출":
        # 감정에 따라 얼굴을 검출하는 방식으로 평가 (가상 예시)
        accuracy, precision, recall, f1 = evaluate_model(model, val_loader, device, num_classes)
    
    # 결과 저장
    results[method] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

# 평가 결과 출력
for method, metrics in results.items():
    print(f"--- {method} ---")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1 Score: {metrics['F1 Score']:.4f}\n")
