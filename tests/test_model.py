import unittest
import tempfile
import torch
import sys
import os
from pathlib import Path

from src import predict_single_news
from src.config import MODEL_PATH

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


"""
While traditional software has deterministic outputs (ex. add(2, 3) always returns 5,
ML models have:
(i).   Non-deterministic training
(ii).  Large dependencies
(iii). Long execution times

Thus, ML pipeline integration testing in concerned with:
 -  Shape correctness
 -  Type correctness
 -  Pipeline completion

Recall ML pipeline:

Raw Text -> Tokenization -> Model -> Predictions -> Formatted output
(strings)    (tensors)    (weights)   (logits)       (probabilities)
"""

class TestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Create temporary directory for test files"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_model_path = Path(cls.temp_dir) / "test_model.pt"

        # sample test texts
        cls.test_texts = {
            'world': "The United Nations Security Council met today to discuss the ongoing conflict in the region. "
                     "Diplomats from various countries expressed concern about the humanitarian crisis.",
            'sports': "The championship game ended in overtime with a stunning victory. The team's star player "
                      "scored 35 points and grabbed 12 rebounds in the thrilling match.",
            'business': "The stock market rallied today as investors responded positively to quarterly earnings reports. "
                        "Tech stocks led the gains with several companies beating analyst expectations.",
            'science': "Scientists have discovered a new exoplanet using advanced telescope technology. "
                       "The planet, located 100 light-years away, shows potential signs of atmospheric water vapor."
        }

    def test_single_forward_pass(self):
        """Test shape of outputs """
        from src.model import create_model
        from src.data_loader import get_data_loaders
        from src.config import DEVICE

        model = create_model()
        train_loader, _, _ = get_data_loaders()

        # first batch only
        batch = next(iter(train_loader))

        # forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(
                batch['input_ids'].to(DEVICE),
                batch['attention_mask'].to(DEVICE)
            )

        batch_size = batch['input_ids'].shape[0]
        n_classes = 4

        self.assertEqual(outputs.shape, (batch_size, n_classes))

    def test_end_to_end_pipeline(self):
        """Test training pipeline"""
        from src.data_loader import create_data_loader, load_and_split_data
        from src.model import TopicClassifier
        from transformers import BertTokenizer
        import torch.nn as nn
        from torch.optim import AdamW

        print("Loading data...")
        df_train, df_val, df_test = load_and_split_data()

        # mini batch of 16 samples for testing
        df_train_mini = df_train.head(16)

        print("Creating model and tokenizer...")
        model = TopicClassifier()
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        device = torch.device('cpu')  # cpu device for testing
        model = model.to(device)

        print("Creating data loader...")
        train_loader = create_data_loader(
            df_train_mini, tokenizer, max_len=128, batch_size=8
        )

        print("Training one batch...")
        model.train()
        optimizer = AdamW(model.parameters())
        loss_fn = nn.CrossEntropyLoss()

        # train mini batch
        batch = next(iter(train_loader))
        outputs = model(
            batch['input_ids'].to(device),
            batch['attention_mask'].to(device)
        )
        loss = loss_fn(outputs, batch['targets'].to(device))

        loss.backward()
        optimizer.step()

        print(f"Training loss: {loss.item():.4f}")

        print("Saving model...")
        torch.save(model.state_dict(), self.test_model_path)

        self.assertTrue(self.test_model_path.exists(), "Model file was not saved")

        print("✓ Pipeline test passed!")

    def test_single_news_prediction(self):
        from src import predict_single_news
        from src.config import CLASS_NAMES, MODEL_PATH

        # testing with business news example
        test_text = self.test_texts['business']              # NOTE: change here
        result = predict_single_news(test_text, MODEL_PATH)

        # existence (output structure)
        self.assertIn('predicted_class', result)
        self.assertIn('confidence', result)
        self.assertIn('all_probabilities', result)
        self.assertIn('sorted_predictions', result)
        self.assertIn('input_text', result)

        # membership
        self.assertIn(result['predicted_class'], CLASS_NAMES)

        # def of probability
        self.assertGreater(result['confidence'], 0)
        self.assertLessEqual(result['confidence'], 1)

        # law of total probability
        prob_sum = sum(result['all_probabilities'].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=5)

        # subset inclusion
        for class_name in CLASS_NAMES:
            self.assertIn(class_name, result['all_probabilities'])

        # shape agreeance
        self.assertEqual(len(result['sorted_predictions']), len(CLASS_NAMES))

        # ordinality
        probs = [p[1] for p in result['sorted_predictions']]
        self.assertEqual(probs, sorted(probs, reverse=True))

        print(f"Prediction result: {result['predicted_class']} "
              f"(confidence: {result['confidence']:.4f})")

    def test_empty_text_prediction(self):
        """Test prediction of an empty text/whitespace"""
        from src.config import MODEL_PATH

        with self.assertRaises(ValueError) as context:
            predict_single_news("", MODEL_PATH)
        self.assertIn("empty", str(context.exception).lower())

        with self.assertRaises(ValueError) as context:
            predict_single_news(None, MODEL_PATH)
        self.assertIn("empty", str(context.exception).lower())

        with self.assertRaises(ValueError) as context:
            predict_single_news("   ", MODEL_PATH)
        self.assertIn("empty", str(context.exception).lower())


if __name__ == '__main__':
    unittest.main(verbosity=2)