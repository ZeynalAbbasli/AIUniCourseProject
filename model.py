import torch
import torch.nn as nn
import torchvision.models as model


class ExModel(nn.Module):
    
    def __init__(self):
        super().__init__()
                
        self.resnet18 = model.resnet18()
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, 8)


    def forward(self, image):
        # Get predictions from ResNet18
        out = self.resnet18(image)
        
        return out
    