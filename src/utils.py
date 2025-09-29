import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def tensor_to_numpy(values):
    """Convert list of tensors or values to numpy array"""

    result = []
    for v in values:
        if torch.is_tensor(v):
            result.append(v.cpu().numpy() if v.dim() > 0 else v.cpu().item())
        else:
            result.append(v)

    return np.array(result)

def plot_training_history(history):
    """Plot training and validation metrics"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    axes[0].plot(tensor_to_numpy(history['train_acc']),
                 label='Train Accuracy', marker='o')
    axes[0].plot(tensor_to_numpy(history['val_acc']),
                 label='Validation Accuracy', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss plot
    axes[1].plot(tensor_to_numpy(history['train_loss']),
                 label='Train Loss', marker='o')
    axes[1].plot(tensor_to_numpy(history['val_loss']),
                 label='Validation Loss', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Generate confusion matrix visualization"""

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    return fig


def generate_classification_report(y_true, y_pred, class_names):
    """Generate detailed classification metrics"""

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )

    return report