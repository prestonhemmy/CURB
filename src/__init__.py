__version__ = '1.0.0'
__author__ = 'Preston Hemmy'

from .model import TopicClassifier, create_model, load_model
from .predict import predict_single_news
from .train import train_model
from .data_loader import get_data_loaders
from .config import CLASS_NAMES

__all__ = [
    'TopicClassifier',
    'create_model',
    'load_model',
    'predict_single_news',
    'train_model',
    'get_data_loaders',
    'CLASS_NAMES'
]