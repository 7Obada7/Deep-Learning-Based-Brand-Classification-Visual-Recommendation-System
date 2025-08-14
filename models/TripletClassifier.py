import torch
import torch.nn as nn
from torchvision import models
import config
class TripletClassifier(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super(TripletClassifier, self).__init__()

        # Load pre-trained VGG-19 model
        self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

        # Freeze all layers (feature extractor and classifier)
        for param in self.model.parameters():
            param.requires_grad = False

        # Modify the classifier: replace the final fc layer
        self.model.classifier[6] = nn.Linear(4096, num_classes)

        # Only the final layer's parameters will be trainable
        for param in self.model.classifier[6].parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)
