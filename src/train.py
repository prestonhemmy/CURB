import torch
from torch import nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from collections import defaultdict
import numpy as np
from config import *

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    """Training helper function"""

    model = model.train()  # enables dropout layers
                           # enables batch normalization

    # initialize tracking
    losses = []
    correct_predictions = 0

    # main loop to handle batches
    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        targets = d['targets'].to(device)
        # NOTE: Moving tensors from CPU to GPU is critical for perf (100x speedup!)

        # forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # predictions and loss
        _, preds = torch.max(outputs, 1)
        loss = loss_fn(outputs, targets)

        # update metrics
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        # backwards pass/backpropagation
        loss.backward()  # calculates gradient
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
        optimizer.step()  # update weights
        scheduler.step()  # update leaning rate
        optimizer.zero_grad()  # reset gradients for next batch

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    """Evaluation helper function"""

    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():   # disables gradient
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            targets = d['targets'].to(device)

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

class EarlyStopping:
    """Early stopping to preventing overfitting"""

    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = None
        self.early_stop = False

    def __call__(self, val_acc, model=None):
        # check if first accuracy seen (update best)
        if self.best_acc is None:
            self.best_acc = val_acc

        # check if curr accuracy worse than best (increment counter)
        elif val_acc < self.best_acc:
            self.counter += 1
            if self.counter >= self.patience: # check if patience constraint met
                self.early_stop = True        # trigger early stop

        # o.w. better accuracy seen (update best; reset counter)
        else:
            self.best_acc = val_acc
            self.counter = 0

        return self.early_stop


def train_model(model, train_loader, val_loader, n_train, n_val, device):
    """Core training function"""

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(device)

    history = defaultdict(list)  # stores accuracy and loss training loop values
    best_accuracy = 0
    early_stop = EarlyStopping(patience=3)

    for epoch in range(EPOCHS):

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print('-' * 10)

        # training
        train_acc, train_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, device, scheduler, n_train
        )

        print(f"Training   loss {train_loss}\taccuracy {train_acc}")

        # validation
        val_acc, val_loss = eval_model(
            model, val_loader, loss_fn, device, n_val
        )

        print(f"Evaluation loss {val_loss}\taccuracy {val_acc}")

        # update (ordered) accuracy/loss metrics
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.pt')
            best_accuracy = val_acc

        if early_stop(val_acc):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        print()

    return history, best_accuracy