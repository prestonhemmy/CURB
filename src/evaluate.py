import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizer

from .config import *
from .model import create_model
from .data_loader import get_data_loaders


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

def evaluate(output_path="assets/confusion_matrix.png"):
    # Print Per-Class Model Metrics

    model = create_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    _, _, test_loader = get_data_loaders(tokenizer)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for d in test_loader:
            input_ids = d['input_ids'].to(DEVICE)
            attention_mask = d['attention_mask'].to(DEVICE)
            targets = d['targets'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    print(classification_report(all_targets, all_preds, target_names=CLASS_NAMES))

    # Generate and Save Confusion Matrix

    cm = confusion_matrix(all_targets, all_preds)

    fix, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='PuBu',
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax
    )

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')

    print(f"Confusion Matrix saved to {output_path}")

if __name__ == '__main__':
    evaluate()
