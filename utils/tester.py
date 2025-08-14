import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm
#import config
import os
from sklearn.neighbors import KNeighborsClassifier
#from models.TripletImageEmbedder import TripletImageEmbedder
import sys
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Now import other modules
from models.TripletImageEmbedder import TripletImageEmbedder
import config

def evaluate_embeddings(model, test_loader, test_dataset):
    """
    Evaluate triplet embeddings with flexible input handling
    """
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Case 1: Triplet input (anchor, positive, negative)
            if len(batch) == 3:
                anchor, positive, negative = batch
                images = anchor.to(config.DEVICE)
                # For triplets, we'll just evaluate on anchors
                # You might want to modify this depending on your needs
                labels.extend([0]*len(anchor))  # Dummy labels
                
            # Case 2: Standard input (images, labels)
            elif len(batch) == 2:
                images, label_batch = batch
                images = images.to(config.DEVICE)
                labels.extend(label_batch.cpu().numpy())
                
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")
            
            # Get embeddings
            emb = model(images)
            embeddings.append(emb.cpu())

    embeddings = torch.cat(embeddings).numpy()
    labels = np.array(labels)
    
    # Get class names if available
    if hasattr(test_dataset, 'classes'):
        class_names = test_dataset.classes
    else:
        class_names = [str(i) for i in np.unique(labels)]
        print("Warning: Using numerical labels - class names not found")

    # Only calculate metrics if we have real labels
    if len(np.unique(labels)) > 1:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(embeddings, labels)
        preds = knn.predict(embeddings)
        
        print("\nClassification Metrics:")
        print(classification_report(labels, preds, target_names=class_names))
        
        # Confusion matrix
        plt.figure(figsize=(15, 12))
        sns.heatmap(confusion_matrix(labels, preds), 
                   annot=False, fmt='d',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.show()
    else:
        print("Embeddings extracted successfully, but no labels for classification")
        print(f"Embedding shape: {embeddings.shape}")
    
    return embeddings


def evaluate_classifier(model, test_loader):
    """
    Evaluate classifier model performance on the test set
    Displays confusion matrix
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing Classifier", ncols=100):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, test_loader.dataset.classes, "Confusion Matrix for Classifier")

def evaluate_triplet_embedder(model, test_loader):
    """
    Evaluate triplet embedder model performance on the test set
    Displays confusion matrix based on anchor-positive-negative similarity comparison
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for anchor, positive, negative in tqdm(test_loader, desc="Testing Triplet Embedder", ncols=100):
            anchor, positive, negative = anchor.to(config.DEVICE), positive.to(config.DEVICE), negative.to(config.DEVICE)
            anchor_output = model(anchor)
            positive_output = model(positive)
            negative_output = model(negative)

            # Here we compute whether the positive pair is closer than the negative one
            positive_distance = F.pairwise_distance(anchor_output, positive_output)
            negative_distance = F.pairwise_distance(anchor_output, negative_output)

            # A simple threshold-based evaluation (if positive pair is closer, it's a correct prediction)
            correct = (positive_distance < negative_distance).float()
            all_preds.extend(correct.cpu().numpy())
            all_labels.extend([1] * len(anchor))  # All labels are positive for simplicity

    # Compute confusion matrix for triplet embedding comparison
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, ["Wrong", "Right"], "Confusion Matrix for Triplet Embedder")



def evaluate_triplet_classifier(model, test_loader, test_dataset):
    """
    Evaluate triplet classifier with full metrics:
    - Accuracy, Precision, Recall (Sensitivity), F1-Score, Specificity
    - Per-class and macro-averaged metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing Triplet Classifier"):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Accuracy
    accuracy = np.mean(all_preds == all_labels)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    #Classification Report (Precision, Recall, F1 per-class)
    class_names = test_dataset.classes
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    #Macro-Averaged Metrics
    macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print("\nMacro-Averaged Metrics:")
    print(f"Precision: {macro_precision:.4f}")
    print(f"Recall (Sensitivity): {macro_recall:.4f}")
    print(f"F1-Score: {macro_f1:.4f}")

    #Confusion Matrix (Normalized + Large Format)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Triplet Classifier)")
    plt.tight_layout()
    plt.show()

    #Per-Class Specificity
    specificity = []
    for i in range(len(class_names)):
        TP = cm[i, i]
        FN = sum(cm[i, :]) - TP
        FP = sum(cm[:, i]) - TP
        TN = cm.sum() - (TP + FP + FN)
        specificity.append(TN / (TN + FP + 1e-10))  # Avoid division by zero

    print("\nPer-Class Specificity:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {specificity[i]:.4f}")

def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", cmap='Blues'):
    """
    Plot the confusion matrix using seaborn
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()