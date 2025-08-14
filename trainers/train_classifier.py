import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.ClassifierModel import ClassifierModel
import config
import os
from utils.plotter import plot_training
from tqdm import tqdm
from torchvision import datasets, transforms
from utils.tester import evaluate_classifier

def train_classifier():
    # Dataloaders
    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),  # Using config for image size
        transforms.ToTensor(),
    ])
    train_dataset = datasets.ImageFolder(root=os.path.join(config.DATA_DIR, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(config.DATA_DIR, 'validation'), transform=transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(config.DATA_DIR, 'test'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config.NORMAL_VGG_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.NORMAL_VGG_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.NORMAL_VGG_BATCH_SIZE, shuffle=False)

    # Model
    model = ClassifierModel(num_classes=config.NUM_CLASSES).to(config.DEVICE)

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.NORMAL_VGG_LEARNING_RATE)

    # Training loop
    train_losses, val_losses = [], []
    for epoch in tqdm(range(config.NORMAL_VGG_NUM_EPOCHS), desc="Training Classifier"):
        model.train()
        total_loss = 0
        # Add tqdm for batch progress
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NORMAL_VGG_NUM_EPOCHS}", leave=False):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{config.NORMAL_VGG_NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    # Save model
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), config.CLASSIFIER_MODEL_PATH)

    # Plot losses
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    plot_training(train_losses, val_losses, filename=os.path.join(config.PLOTS_DIR, 'classifier_losses.png'))

    # Test model
    print("\nTesting on Test Set...")
    evaluate_classifier(model, test_loader)
