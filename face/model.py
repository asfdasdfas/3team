import torch
import torch.nn as nn
from torchvision import models

class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_classes=7):
        """
        EmotionRecognitionModel 클래스는 ResNet-18을 기반으로 하여 표정 인식 모델을 정의합니다.
        
        Args:
            num_classes (int): 분류할 표정의 개수 (기본값: 7)
        """
        super(EmotionRecognitionModel, self).__init__()
        
        # ResNet-18 모델을 미리 학습된 가중치로 불러옵니다.
        self.base_model = models.resnet18(pretrained=True)
        
        # ResNet의 마지막 Fully Connected Layer(fc)를 수정하여, 표정 개수에 맞는 출력 클래스를 설정합니다.
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        
    def forward(self, x):
        """
        모델의 순전파 함수. 입력 이미지를 모델을 통해 전달하여 예측 결과를 반환합니다.
        
        Args:
            x (Tensor): 모델에 입력되는 이미지 배치 텐서 (배치 크기, 채널, 높이, 너비)
        
        Returns:
            Tensor: 모델의 출력값 (배치 크기, num_classes)
        """
        return self.base_model(x)
