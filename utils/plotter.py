import matplotlib.pyplot as plt

def plot_training(train_losses, val_losses, filename="training_curves.png"):
    """
    Plot and save training/validation loss curves.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
