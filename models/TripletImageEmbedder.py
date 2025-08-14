import torch
import torch.nn as nn
from torchvision import models

class TripletImageEmbedder(nn.Module):
    def __init__(self, embedding_dim=49):
        super(TripletImageEmbedder, self).__init__()

        # Load VGG-19 model
        self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

        # Freeze all convolutional layers
        for param in self.model.features.parameters():
            param.requires_grad = False

        self.model.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, embedding_dim),  # Final layer for embeddings
        )

        # Freeze the first two Linear layers (25088->4096 and 4096->4096)
        for layer in list(self.model.classifier.children())[:-1]:  # Freeze all except the last layer
            if isinstance(layer, nn.Linear):
                for param in layer.parameters():
                    param.requires_grad = False

        # Normalize embeddings for triplet loss
        self.normalize = nn.functional.normalize

    def forward(self, x):
        x = self.model.features(x)
        x = torch.flatten(x, 1)
        x = self.model.classifier(x)
        x = self.normalize(x, p=2, dim=1)  # Normalize output
        return x
