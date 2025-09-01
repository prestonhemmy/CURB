import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split

class NewsArticlesDataset(Dataset):

    def __init__(self, descriptions, targets, tokenizer, max_len):
        self.descriptions = descriptions    # array of inputs
        self.targets = targets              # array of outputs
        self.tokenizer = tokenizer
        self.max_len = max_len

    # NOTE. Since our NewsArticleDataset class inherits from the Dataset class of
    # torch.utils.data, note that this class requires the methods __len__ and __getitem__.
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

def get_data_loaders(train_path, test_path, tokenizer, max_len, batch_size, val_split=0.05, random_seed=42):
    """Load data and create training/validation/test dataloaders"""

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    df_train, df_val = train_test_split(
        df_train,
        test_size=val_split,
        random_seed=random_seed
    )

    train_loader = create_data_loader(df_train, tokenizer, max_len, batch_size)
    val_loader = create_data_loader(df_val, tokenizer, max_len, batch_size)
    test_loader = create_data_loader(df_test, tokenizer, max_len, batch_size)

    return train_loader, val_loader, test_loader, len(df_train), len(df_val), len(df_test)

    # NOTE: Returns each individual dataloader as well as the number of tuples in each