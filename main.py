import os
import torch
import torchvision
from torchvision import transforms
from utils.tester import evaluate_classifier, evaluate_triplet_classifier,evaluate_embeddings
from trainers.train_classifier import train_classifier
from trainers.train_triplet_embedder import train_triplet_embedder
from trainers.train_triplet_classes import train_triplet_classifier
import config
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets.triplet_dataset import TripletDataset


def main():
    while True:
        print("\nSelect an option:")
        print("1. Train Classifier Model")
        print("2. Train Triplet Embedder Model")
        print("3. Train Triplet Classifier Model")
        print("4. Test Classifier Model")
        print("5. Test Triplet Embedder Model")
        print("6. Test Triplet Classifier Model")
        print("7. Exit")

        choice = input("Enter your choice (1-7): ")

        if choice == '1':
            print("\nTraining Classifier Model...")
            train_classifier()

        elif choice == '2':
            print("\nTraining Triplet Embedder Model...")
            train_triplet_embedder()

        elif choice == '3':
            print("\nTraining Triplet Classifier Model...")
            train_triplet_classifier()

        elif choice == '4':
            print("\nTesting Classifier Model...")
            if os.path.exists(config.CLASSIFIER_MODEL_PATH):
                model = torch.load(config.CLASSIFIER_MODEL_PATH)
                test_loader = torch.utils.data.DataLoader(dataset=torchvision.datasets.ImageFolder(os.path.join(config.DATA_DIR, 'test'), transform=transforms.Compose([transforms.Resize(config.IMAGE_SIZE), transforms.ToTensor()])), batch_size=config.NORMAL_VGG_BATCH_SIZE)
                evaluate_classifier(model, test_loader)
            else:
                print("Error: Classifier model is not found!")

        elif choice == '5':
            print("\nTesting Triplet Embedder Model...")
            if os.path.exists(config.TRIPLET_EMBEDDER_MODEL_PATH):
                # Initialize the model first
                from models.TripletImageEmbedder import TripletImageEmbedder
                model = TripletImageEmbedder().to(config.DEVICE)
                
                # Then load the state dict
                model.load_state_dict(torch.load(config.TRIPLET_EMBEDDER_MODEL_PATH))
                
                transform = transforms.Compose([
                    transforms.Resize(config.IMAGE_SIZE),
                    transforms.ToTensor(),
                ])
                test_dataset = TripletDataset(data_dir=os.path.join(config.DATA_DIR, 'test'), transform=transform)
                test_loader = DataLoader(test_dataset, batch_size=config.TRIPLET_LOSS_BATCH_SIZE, shuffle=False)
                
                evaluate_embeddings(model, test_loader, test_dataset)
            else:
                print("Error: Triplet Embedder model is not found!")

        elif choice == '6':
            print("\nTesting Triplet Classifier Model...")
            if os.path.exists(config.TRIPLET_CLASSIFIER_MODEL_PATH):
                # Import your model class first!
                from models.TripletImageEmbedder import TripletImageEmbedder  

                # Instantiate the model
                model = TripletImageEmbedder()
                
                # Load the saved weights
                model.load_state_dict(torch.load(config.TRIPLET_CLASSIFIER_MODEL_PATH))
                model = model.to(config.DEVICE)

                # Prepare test_loader
                test_loader = torch.utils.data.DataLoader(
                    dataset=torchvision.datasets.ImageFolder(
                        os.path.join(config.DATA_DIR, 'test'),
                        transform=transforms.Compose([
                            transforms.Resize(config.IMAGE_SIZE),
                            transforms.ToTensor()
                        ])
                    ),
                    batch_size=config.TRIPLET_LOSS_BATCH_SIZE
                )
                
                test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(config.DATA_DIR, 'test'))

                # Evaluate
                evaluate_triplet_classifier(model, test_loader,test_dataset)
            else:
                print("Error: Triplet Classifier model is not found!")

        elif choice == '7':
            print("Exiting...")
            break

        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()
