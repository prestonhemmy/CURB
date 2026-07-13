import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os

from matplotlib.patches import Rectangle
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizer

from .config import *
from .model import create_model
from .data_loader import get_data_loaders


BUSINESS_IDX = CLASS_NAMES.index("Business")
SCIENCE_IDX = CLASS_NAMES.index("Science")

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

def evaluate(output_path="assets/confusion_matrix.png"):
    """
    Run evaluation on the test set.

    Prints a classification report (precision, recall, F1 per class),
    saves a confusion matrix heatmap, and prints misclassified examples
    for error analysis on Business (FNs) and Science (FPs).

    :param output_path: file path for the confusion matrix image
    """

    # Print Per-Class Model Metrics

    model = create_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    _, _, test_loader = get_data_loaders(tokenizer)

    all_preds = []
    all_targets = []
    all_texts = []

    with torch.no_grad():
        for d in test_loader:
            input_ids = d['input_ids'].to(DEVICE)
            attention_mask = d['attention_mask'].to(DEVICE)
            targets = d['targets'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_texts.extend(d['description_text'])

    print(classification_report(all_targets, all_preds, target_names=CLASS_NAMES))

    # Generate and Save Confusion Matrix

    cm = confusion_matrix(all_targets, all_preds)

    fix, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='PuBu', vmax=200,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
    )

    # Highlight Business-Science mirror cells
    ax.add_patch(Rectangle(
        (3, 2), 1, 1, fill=False, edgecolor='gold',
        linewidth=2.5, clip_on=False
    ))
    ax.add_patch(Rectangle(
        (2, 3), 1, 1, fill=False, edgecolor='gold',
        linewidth=2.5, clip_on=False
    ))

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')

    print(f"Confusion Matrix saved to {output_path}")

    # Print Error Analysis

    # print_error_analysis(all_texts, all_targets, all_preds)

def print_error_analysis(texts, targets, preds, n=20):
    """
    Print misclassified examples for targeted error analysis.

    :param texts: list of original article texts
    :param targets: list of true class indices
    :param preds: list of predicted class indices
    :param n: number of examples to print per category

    Outputs:
        Science FPs: texts predicted as Science but are actually some other category
        Business FNs: texts that are Business but predicted as some other category
    """
    science_fps = []
    business_fns = []

    for text, target, pred in zip(texts, targets, preds):
        if pred == SCIENCE_IDX and target != SCIENCE_IDX:
            science_fps.append((text, CLASS_NAMES[target], CLASS_NAMES[pred]))

        if target == BUSINESS_IDX and pred != BUSINESS_IDX:
            business_fns.append((text, CLASS_NAMES[target], CLASS_NAMES[pred]))

    print("\n" + "=" * 70)
    print(f"Science False Positives")
    print(f"Showing {min(n, len(science_fps))} of {len(science_fps)} total")
    print("=" * 70)

    for i, (text, actual, predicted) in enumerate(science_fps[:n]):
        truncated = text[:120] + "..." if len(text) > 120 else text
        print(f"\n  [{i + 1}] Actual: {actual} → Predicted: {predicted}")
        print(f"      {truncated}")

    print("\n" + "=" * 70)
    print(f"Business False Negatives")
    print(f"Showing {min(n, len(business_fns))} of {len(business_fns)} total")
    print("=" * 70)

    for i, (text, actual, predicted) in enumerate(business_fns[:n]):
        truncated = text[:120] + "..." if len(text) > 120 else text
        print(f"\n  [{i + 1}] Actual: {actual} → Predicted: {predicted}")
        print(f"      {truncated}")

    print()

if __name__ == '__main__':
    evaluate()
