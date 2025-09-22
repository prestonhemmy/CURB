import torch
from transformers import BertTokenizer
import torch.nn.functional as F
from .model import create_model
from .config import *

def predict_single_news(text, model_path):
    """
    API-level prediction class for end-users

    :param text: input string excerpt from body of news article
    :param model_path: path to saved model
    :return: dictionary containing predictions, confidence, and probabilities
    """

    # input text validation
    if not text or not isinstance(text, str) or text.strip() == "":
        raise ValueError("Missing field. News article text cannot be empty.")
    
    # load best model from /checkpoints
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # tokenization
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    # forward pass
    with torch.no_grad():
        outputs = model(
            input_ids = encoding['input_ids'].to(DEVICE),
            attention_mask = encoding['attention_mask'].to(DEVICE)
        )

        # post-processing
        probabilities  = F.softmax(outputs, dim=1)   # convert logits to probabilities
        confidence, index = torch.max(probabilities, dim=1)

    # type conversion
    predicted_index = index.item() - 1
    confidence_score = confidence.item()
    all_probabilities = probabilities.squeeze().cpu().numpy()

    # probability distribution creation
    class_probabilities = {
        CLASS_NAMES[i]: float(prob)
        for i, prob in enumerate(all_probabilities)
    }

    # sort probabilities in descending order
    sorted_predictions = sorted(
        class_probabilities.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # formatted output
    return {
        'predicted_class': CLASS_NAMES[predicted_index],
        'confidence': confidence_score,
        'all_probabilities': class_probabilities,
        'sorted_predictions': sorted_predictions,
        'input_text': text[:100] + '...' if len(text) > 100 else text
    }

























