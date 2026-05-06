import torch
from sklearn.metrics import classification_report
from transformers import BertTokenizer
from .config import *
from .model import create_model
from .data_loader import get_data_loaders

def get_per_class_metrics():
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

if __name__ == '__main__':
    get_per_class_metrics()
