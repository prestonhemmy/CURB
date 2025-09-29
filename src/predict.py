import torch
from transformers import BertTokenizer
import torch.nn.functional as F
from .model import create_model
from .config import *
from langdetect import detect
import time

class NewsClassifierService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

    def load_model(self, model_path):
        """
        Load model and tokenizer once at startup

        :param model_path: path to saved model
        :return: True if model is loaded and False otherwise
        """
        # print(f"Loading model from {model_path}... \nEstimated wait time: 10 seconds")
        # start_time = time.time()

        try:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

            self.model = create_model()
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            self.model.to(DEVICE)
            self.model.eval()

            self.is_loaded = True

            # load_time = time.time() - start_time
            # print(f"Model loaded in {load_time:.2f} seconds")

            # model service warm up
            self._warmup()

            return True

        except Exception as e:
            print(f"Failed to load model: {e} .")

            self.is_loaded = False

            return False

    def predict(self, text):
        """
        Predict news category for API

        :param text: news article text
        :return: dictionary with predictions and metadata
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Method load_model() must be called first.")

        # input validation
        self.validate(text)

        # tokenization
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        # forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding["input_ids"].to(DEVICE),
                attention_mask=encoding["attention_mask"].to(DEVICE)
            )

            # post-processing
            probabilities = F.softmax(outputs, dim=1)
            confidence, index = torch.max(probabilities, dim=1)

        # type conversion
        predicted_index = index.item()
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

    @staticmethod
    def validate(text):
        """Function-level validation of news text"""
        # falsy and type and whitespace checking
        if not text or not isinstance(text, str) or text.strip() == "":
            raise ValueError(f"Invalid text. Expected non-empty string, got {text} of type {type(text)} .")

        # char overflow checking
        if len(text) > 50000:
            raise ValueError(f"Invalid text length. Expected at most 50 000 characters, got {len(text)} .")

        # TODO: Consider removing since slow?
        # language checking
        if detect(text) != "en":
            raise ValueError(f"Invalid text language. Expected English ('en'), got '{detect(text)}' .")

    def _warmup(self):
        """Initialize to ensure "Just in time" compilation (for first response)"""
        if self.is_loaded:
            dummy_text = """As tensions flared over a new U.S. tariff, the tech giant Oracle announced a multi-billion
            dollar AI deal with OpenAI, distracting from the ongoing Israel-Gaza war and Max Verstappen's record-breaking
            lap at the Italian Grand Prix."""

            _ = self.predict(dummy_text)

            print("Model warmup complete")


# global singleton
classifier_service = NewsClassifierService()


# global singleton
classifier_service = NewsClassifierService()


def predict_single_news(text, model_path):
    """
    Prediction for batch processing / CLI (Legacy function)

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
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

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
    predicted_index = index.item()
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

























