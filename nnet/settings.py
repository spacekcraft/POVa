import os
import string


PERO_DATASET_PATH = os.environ.get('PERO_DATASET_PATH')
PERO_ANNOTATIONS_PATH = os.environ.get('PERO_ANNOTATIONS')

ALPHABET = string.ascii_letters + string.digits + string.punctuation + ' \n'
NUMBER_OF_CLASSES = len(ALPHABET)

