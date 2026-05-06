import json
import matplotlib.pyplot as plt
import os

def plot_training_curves(history_path="../models/training_history.json", output_path="assets/training_curves.png"):
    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_acc"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # accuracy
    ax1.plot(epochs, history["train_acc"], "steelblue", label="Training")
    ax1.plot(epochs, history["val_acc"], "orange", label="Validation")
    ax1.axvspan(3, 6, color="lightgray", alpha=0.3, label="Overfitting Region")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # loss
    ax2.plot(epochs, history["train_loss"], "steelblue", label="Training")
    ax2.plot(epochs, history["val_loss"], "orange", label="Validation")
    ax2.axvline(x=3, color="lightgray", linewidth=3, zorder=1)
    ax2.text(
        3, 0.85, "Fitting Point", transform=ax2.get_xaxis_transform(),
        color="gray", verticalalignment="center", horizontalalignment="center",
        bbox=dict(facecolor="white", edgecolor="none")
    )
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Training curves saved to {output_path}")

if __name__ == '__main__':
    plot_training_curves()
