import os
import string

SAVE_MODEL_PATH = 'crnn.pth'

PERO_DATASET_PATH = os.environ.get('PERO_DATASET_PATH', "./pero/lines")
PERO_ANNOTATIONS_PATH_TRAIN = os.environ.get('PERO_ANNOTATIONS_TRAIN', "./pero/train.easy")
PERO_ANNOTATIONS_PATH_VAL = os.environ.get('PERO_ANNOTATIONS_VAL', "./pero/val.easy")

ALPHABET = string.ascii_letters + string.digits + string.punctuation + ' ©°§ěščřžýáíéĚŠČŘŽÝÁÍÉďĎťŤňŇ'
NUMBER_OF_CLASSES = len(ALPHABET)

TRAINING_PROGRESS_LOG_INTERVAL = 100

LMDB_DATA_OUTPUT_PATH_TRAIN = '/tmp/lmdb_dataset_store/train'
LMDB_DATA_OUTPUT_PATH_VALID = '/tmp/lmdb_dataset_store/valid'
