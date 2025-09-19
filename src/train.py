import torch
from torch import nn
from torch.optim import AdamW
from torch.xpu import device
from transformers import get_linear_schedule_with_warmup, BertTokenizer
from collections import defaultdict
import numpy as np
from .config import *
from .model import create_model
from .data_loader import get_data_loaders


class EarlyStopping:
    """Early stopping to preventing overfitting"""

    def __init__(self, min_delta=0.001):
        """
        :param min_delta: minimum change to qualify as improvement
        """
        self.patience = PATIENCE
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = None
        self.early_stop = False

    def __call__(self, val_acc, model=None):
        """
        Check if training should continue / be stopped
        :param val_acc: validation accuracy score
        :param model: model instance to save if early stopping
        :return: boolean if early stopping
        """
        if self.best_acc is None:
            self.best_acc = val_acc

        elif val_acc < self.best_acc:
            self.counter += 1
            if self.counter >= self.patience:  # check if patience constraint met
                self.early_stop = True
                return True                    # trigger early stop

        else:
            self.best_acc = val_acc
            self.counter = 0

        return False



def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, n_examples):
    """
    Training helper function

    :param model: model instance
    :param data_loader: training data loader
    :param loss_fn: loss function
    :param optimizer: optimizer instance
    :param scheduler: learning rate scheduler
    :param n_examples: number of training examples
    :return: tuple of model accuracy and average loss
    """

    model = model.train()  # enables dropout layers
                           # enables batch normalization

    # initialize tracking
    losses = []
    correct_predictions = 0

    # main loop to handle batches
    for d in data_loader:
        input_ids = d['input_ids'].to(DEVICE)
        attention_mask = d['attention_mask'].to(DEVICE)
        targets = d['targets'].to(DEVICE)

        # forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # predictions and loss
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        # update metrics
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        # backwards pass/backpropagation
        loss.backward()        # calculates gradient
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()       # update weights
        scheduler.step()       # update leaning rate
        optimizer.zero_grad()  # reset gradients for next batch

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, n_examples):
    """
    Evaluation helper function

    :param model: model instance
    :param data_loader: validation or test data loader
    :param loss_fn: loss function
    :param n_examples: number of training examples
    :return: tuple of model accuracy and average loss
    """

    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():   # disables gradient
        for d in data_loader:
            input_ids = d['input_ids'].to(DEVICE)
            attention_mask = d['attention_mask'].to(DEVICE)
            targets = d['targets'].to(DEVICE)

            # forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # prediction and loss
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)

            # update metrics
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

def train_model(model=None, save_best=True):
    """
    Core training function

    :param model: model instance to save if early stopping
    :param save_best: whether to save best model
    :return: dictionary of training history
    """

    if model is None:
        model = create_model()

    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    train_loader, val_loader, test_loader = get_data_loaders(tokenizer)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    n_train = len(train_loader)
    n_val = len(val_loader)
    total_steps = n_train * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(DEVICE)

    history = defaultdict(list)  # stores accuracy and loss training loop values
    best_accuracy = 0

    early_stop = EarlyStopping()

    for epoch in range(EPOCHS):

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print('-' * 10)

        # training
        train_acc, train_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, n_train
        )

        print(f"Training   loss {train_loss:.4f}\taccuracy {train_acc:.4f}")

        # validation
        val_acc, val_loss = eval_model(
            model, val_loader, loss_fn, n_val
        )

        print(f"Evaluation loss {val_loss:.4f}\taccuracy {val_acc:.4f}")

        # update history
        history['train_acc'].append(train_acc.cpu().numpy() if torch.is_tensor(train_acc) else train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy and save_best:
            torch.save(model.state_dict(), 'best_model_state.pt')
            best_accuracy = val_acc

        if early_stop(val_acc):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        print()

    return history

if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print(f"Training on device {DEVICE}")

    history = train_model()

    print("Training done!")
    print(f"Final validation accuracy: {history['val_acc'][-1]:.4f}")