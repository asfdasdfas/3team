import torch
from torchvision import models, transforms
from PIL import Image

# 표정 분류 모델 (ResNet 사용 예시)
class EmotionRecognitionModel(torch.nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionRecognitionModel, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = torch.nn.Linear(self.base_model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x)

# 학습된 모델 로드
emotion_model = EmotionRecognitionModel(num_classes=7)  # 예시로 7개의 표정
emotion_model.load_state_dict(torch.load('emotion_model.pth'))
emotion_model.eval()

# 얼굴 이미지를 표정 인식 모델에 입력
def predict_expression(face_image):
    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    face_tensor = transform(face_image).unsqueeze(0)  # 배치 차원 추가
    with torch.no_grad():
        output = emotion_model(face_tensor)
    _, predicted = torch.max(output, 1)
    
    return predicted.item()

# 얼굴에 대해 표정 예측
for face in faces:
    # 얼굴 영역을 잘라냄 (bounding box 기반)
    x1, y1, w, h = int(face[0] - face[2]/2), int(face[1] - face[3]/2), int(face[2]), int(face[3])
    face_img = img.crop((x1, y1, x1 + w, y1 + h))
    
    # 표정 예측
    expression = predict_expression(face_img)
    print(f'Predicted emotion for this face: {expression}')
