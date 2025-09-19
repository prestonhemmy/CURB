import torch

RANDOM_SEED = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
N_CLASSES = 4

# data processing
MAX_LEN = 150
BATCH_SIZE = 16

# training
EPOCHS = 10
LEARNING_RATE = 2e-5
DROPOUT_PROB = 0.3

# early stopping
PATIENCE = 3

# file paths
TRAIN_DATA_PATH = 'data/raw/train.csv'
TEST_DATA_PATH = 'data/raw/test.csv'
MODEL_PATH = 'models/checkpoints/best_model_state.pt'

CLASS_NAMES = ['World', 'Sports', 'Business', 'Sci/Tech']