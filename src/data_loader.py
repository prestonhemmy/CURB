import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import *

class NewsArticlesDataset(Dataset):
    """Custom dataset for news article classification"""

    def __init__(self, descriptions, targets, tokenizer, max_len):
        """
        :param descriptions: array of text descriptions
        :param targets: array of class labels
        :param tokenizer: BERT tokenizer instance
        :param max_len: maximum sequence length
        """

        self.descriptions = descriptions
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, item):
        description = str(self.descriptions[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            description,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'description_text': description,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding ['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }



def create_data_loader(df, tokenizer, max_len, batch_size):
    """
    Create DataLoader from a dataframe

    :param df: dataframe columns corresponding to descriptions and targets
    :param tokenizer: BERT tokenizer instance
    :param max_len: maximum sequence length
    :param batch_size: batch size for DataLoader
    :return: DataLoader object
    """

    ds = NewsArticlesDataset(
        descriptions=df['Description'].to_numpy(),
        targets=df['Class Index'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0               # NOTE: Causes errors on Windows unless zero
    )

def load_and_split_data(val_split=0.05):
    """
    Load data; create train / validation / test split

    :param val_split: portion of training data to be used for validation
    :return: tuple of training, validation and test dataframes
    """

    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)

    # NOTE. Convert to 0-indexing which is required for cross-entropy loss
    df_train['Class Index'] = df_train['Class Index'] - 1
    df_test['Class Index'] = df_test['Class Index'] - 1

    df_train, df_val = train_test_split(
        df_train, test_size=val_split, random_state=RANDOM_SEED
    )

    return df_train, df_val, df_test

def get_data_loaders(tokenizer=None):
    """
    Create DataLoaders for training, validation and testing

    :param tokenizer: BERT tokenizer instance
    :return: tuple of training, validation and test DataLoaders
    """

    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    df_train, df_val, df_test = load_and_split_data()

    train_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
    test_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

    return train_loader, val_loader, test_loader