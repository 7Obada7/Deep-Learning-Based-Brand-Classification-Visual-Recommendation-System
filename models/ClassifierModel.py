import torch
import torch.nn as nn
from torchvision import models

class ClassifierModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ClassifierModel, self).__init__()
        
        # Load the pre-trained VGG-19 model
        self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

        # Modify the classifier part of the VGG model
        self.model.classifier[6] = nn.Linear(4096, num_classes)
        
    def forward(self, x):
        return self.model(x)
