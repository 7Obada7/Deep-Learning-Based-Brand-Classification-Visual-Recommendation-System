import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.TripletImageEmbedder import TripletImageEmbedder
import config
import os
from utils.plotter import plot_training
from tqdm import tqdm  # For progress bar
from torchvision import datasets, transforms
from utils.tester import evaluate_triplet_embedder,evaluate_triplet_classifier
from datasets.triplet_dataset import TripletDataset

def train_triplet_embedder():
    # Check if the base classifier model exists
    if not os.path.exists(config.CLASSIFIER_MODEL_PATH):
        print("Error: Base classifier model not found. Please train the classifier model first.")
        return

    # Load the base classifier model
    model = TripletImageEmbedder().to(config.DEVICE)
    model.load_state_dict(torch.load(config.CLASSIFIER_MODEL_PATH))

    # Dataloaders
    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    train_dataset = TripletDataset(data_dir=os.path.join(config.DATA_DIR, 'train'), transform=transform)
    val_dataset = TripletDataset(data_dir=os.path.join(config.DATA_DIR, 'validation'), transform=transform)
    test_dataset = TripletDataset(data_dir=os.path.join(config.DATA_DIR, 'test'), transform=transform)


    train_loader = DataLoader(train_dataset, batch_size=config.TRIPLET_LOSS_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.TRIPLET_LOSS_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.TRIPLET_LOSS_BATCH_SIZE, shuffle=False)

    # Loss and optimizer
    criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=config.TRIPLET_LOSS_LEARNING_RATE)

    # Training loop
    train_losses, val_losses = [], []
    for epoch in tqdm(range(config.TRIPLET_LOSS_NUM_EPOCHS), desc="Training Triplet Embedder"):
        model.train()
        total_loss = 0
        for anchor, positive, negative in train_loader:
            anchor, positive, negative = anchor.to(config.DEVICE), positive.to(config.DEVICE), negative.to(config.DEVICE)

            # Forward pass
            anchor_output = model(anchor)
            positive_output = model(positive)
            negative_output = model(negative)

            # Calculate loss
            loss = criterion(anchor_output, positive_output, negative_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                anchor, positive, negative = anchor.to(config.DEVICE), positive.to(config.DEVICE), negative.to(config.DEVICE)
                anchor_output = model(anchor)
                positive_output = model(positive)
                negative_output = model(negative)

                loss = criterion(anchor_output, positive_output, negative_output)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{config.TRIPLET_LOSS_NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Save model
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), config.TRIPLET_EMBEDDER_MODEL_PATH)

    # Plot losses
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    plot_training(train_losses, val_losses, filename=os.path.join(config.PLOTS_DIR, 'triplet_embedder_losses.png'))

    # Test model
    print("\nTesting on Test Set...")
    evaluate_triplet_embedder(model, test_loader)

        # Test model
    print("\nTesting on Test Set...")
    evaluate_triplet_classifier(model, test_loader,test_dataset)

