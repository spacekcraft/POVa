import os
import string

SAVE_MODEL_PATH = 'crnn.pth'

PERO_DATASET_PATH = os.environ.get('PERO_DATASET_PATH', "./pero/lines")
PERO_ANNOTATIONS_PATH_TRAIN = os.environ.get('PERO_ANNOTATIONS_TRAIN', "./pero/train.easy")
PERO_ANNOTATIONS_PATH_VAL = os.environ.get('PERO_ANNOTATIONS_VAL', "./pero/val.easy")

ALPHABET = string.ascii_letters + string.digits + string.punctuation + ' °§'
NUMBER_OF_CLASSES = len(ALPHABET)

TRAINING_PROGRESS_LOG_INTERVAL = 100
