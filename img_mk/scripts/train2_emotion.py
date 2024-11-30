import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from PIL import Image
import warnings
from torchvision.models.inception import InceptionOutputs

# 경고 메시지 무시 (손상된 EXIF 데이터 등)
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")

# 데이터셋 경로 (절대 경로로 수정)
DATASET_PATH = 'D:/kor_face_ai/real_t'  # 모든 데이터가 있는 폴더 경로

# 하이퍼파라미터 설정
BATCH_SIZE = 16
IMG_SIZE = (299, 299)
EPOCHS = 30
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.5

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 데이터셋 불러오기
train_dataset = datasets.ImageFolder(os.path.join(DATASET_PATH), transform=transform)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# GoogleNet (Inception-V3) 모델 정의
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.inception_v3(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(DROPOUT_RATE),
    nn.Linear(256, len(train_dataset.dataset.classes)),
    nn.LogSoftmax(dim=1)
)
model = model.to(device)

# 손실 함수와 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 모델 학습
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    model.train()
    train_acc = []
    val_acc = []
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            # InceptionV3는 InceptionOutputs 객체를 반환하므로, 메인 출력을 사용합니다.
            if isinstance(outputs, InceptionOutputs):
                outputs = outputs.logits

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_accuracy = correct / total
        train_acc.append(train_accuracy)

        # 검증 단계
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                # InceptionV3의 메인 출력 사용
                if isinstance(outputs, InceptionOutputs):
                    outputs = outputs.logits

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        val_acc.append(val_accuracy)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        model.train()

    return train_acc, val_acc

# 학습 실행
train_acc, val_acc = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)

# 학습된 모델 저장
model_save_path = "D:/git/img_mk/models/emotion_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"모델이 {model_save_path}에 저장되었습니다.")

# 학습 및 검증 정확도 그래프 그리기
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# 모델 불러오기 테스트
def load_emotion_model(model_path):
    model = models.inception_v3(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(DROPOUT_RATE),
        nn.Linear(256, len(train_dataset.dataset.classes)),
        nn.LogSoftmax(dim=1)
    )
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

# 모델 불러오기
loaded_model = load_emotion_model(model_save_path)
print(f"{model_save_path}에서 모델을 성공적으로 불러왔습니다.")

# 테스트 데이터셋 평가
def evaluate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # InceptionV3의 메인 출력 사용
            if isinstance(outputs, InceptionOutputs):
                outputs = outputs.logits

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")

# 테스트 데이터셋 로드 및 평가
evaluate_model(loaded_model, val_loader)
