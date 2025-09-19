import torch
from torch import nn
from transformers import BertModel
from .config import *

class TopicClassifier(nn.Module):
    """News topic classifier based on BERT"""

    def __init__(self):
        super(TopicClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=DROPOUT_PROB)
        self.out = nn.Linear(self.bert.config.hidden_size, N_CLASSES)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model

        :param input_ids:  tensor of token ides
        :param attention_mask: tensor of attention masks
        :return: tensor of logits
        """
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        output = self.drop(pooled_output)

        return self.out(output)



def create_model():
    """
    Create and initialize the model

    :return: new model instance
    """

    model = TopicClassifier()

    return model.to(DEVICE)

def load_model():
    """
    Load a model from a checkpoint

    :return: loaded model instance
    """

    model = TopicClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    return model