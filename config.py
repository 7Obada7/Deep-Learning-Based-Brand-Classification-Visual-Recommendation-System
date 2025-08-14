import os
import torch

# Training settings for Normal VGG
NORMAL_VGG_BATCH_SIZE = 32
NORMAL_VGG_NUM_EPOCHS = 200
NORMAL_VGG_LEARNING_RATE = 1e-4

# Training settings for Triplet Loss models
TRIPLET_LOSS_BATCH_SIZE = 32
TRIPLET_LOSS_NUM_EPOCHS = 200
TRIPLET_LOSS_LEARNING_RATE = 1e-4

# Paths
DATA_DIR = r'C:/Users/obama/OneDrive/سطح المكتب/test22/Custom_DataSet' # Set the path to your dataset
CHECKPOINT_DIR = "C:/Users/obama/OneDrive/سطح المكتب/00/Bitirme - Copy/Bitirme - Copy/checkpoints"
PLOTS_DIR = "./plots"

# Model Save Names
CLASSIFIER_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "classifier.pth")
TRIPLET_EMBEDDER_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "triplet_embedder.pth")
TRIPLET_CLASSIFIER_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "triplet_classifier.pth")

# Embedding dimension for triplet models
EMBEDDING_DIM = 49

# Number of classes
NUM_CLASSES = 49  # Set the number of classes here

# Image resize dimensions
IMAGE_SIZE = (224, 224)  # Image size for resizing during transformations

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
