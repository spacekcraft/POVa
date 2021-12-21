import os
import string


PERO_DATASET_PATH = os.environ.get('PERO_DATASET_PATH', "./pero/lines")
PERO_ANNOTATIONS_PATH = os.environ.get('PERO_ANNOTATIONS', "./pero/train.easy")

ALPHABET = string.ascii_letters + string.digits + string.punctuation + ' °'
NUMBER_OF_CLASSES = len(ALPHABET)

