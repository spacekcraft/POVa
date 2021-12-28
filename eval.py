import argparse

import torch
from torchvision.transforms import Lambda, Resize, Grayscale
from torch.autograd import Variable
from matplotlib import pyplot as plt

from PIL import Image
import numpy as np

from models.crnn import CRNN
from nnet.trainer import Trainer
from nnet.dataloader import PeroDataset, make_dataloader, make_lmdb_dataloader
from nnet.utils import StrLabelConverter
from nnet.cwer import getCer, getWer
from nnet.settings import (
    PERO_DATASET_PATH,
    PERO_ANNOTATIONS_PATH_TRAIN,
    PERO_ANNOTATIONS_PATH_VAL,
    ALPHABET,
    NUMBER_OF_CLASSES,
    TRAINING_PROGRESS_LOG_INTERVAL,
    SAVE_MODEL_PATH,
    LMDB_DATA_OUTPUT_PATH_TRAIN,
    LMDB_DATA_OUTPUT_PATH_VALID
)


def parse_args():
    parser = argparse.ArgumentParser(description='Script for eval the model.')
    parser.add_argument('--dataset', '-d', default="./pero/train.easy", type=str)
    parser.add_argument('--samples', '-s', default=100, type=int)
    args = parser.parse_args()
    return args


def model_init(*args, pretrained=False):
    if pretrained:
        print("Loaded pretrained model")
        return torch.load("best.pt.tar",map_location ='cpu')
    return CRNN(*args)

def load_model(*args):
    nchanels, image_h, image_w = 1, 32, 512
    model = CRNN(image_h, nchanels, NUMBER_OF_CLASSES, 256)
    checkpoint = model_init(args,pretrained=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

def main():
    args = parse_args()
    
    nchanels, image_h, image_w = 1, 32, 512

    img_tranforms = torch.nn.Sequential(
            Resize((image_h, image_w)),
            Grayscale(nchanels),
        )

    dataset=PeroDataset(args.dataset,"./pero/lines",img_tranforms,False)
    converter = StrLabelConverter(dataset._alphabet)

    print("Dataset name:",args.dataset)
    print("Alphabet:'"+dataset._alphabet+"'")

    model = load_model(args)

    val_string_targets = []
    predicted_strings_all = []

    cer = 0
    wer = 0 
    num = 10

    for i in range(num):
        X,y=dataset.__getitem__(i)
        X = X.unsqueeze(0)
        pred = model(X)
        
        batch_size = X.shape[0]
        preds_size = torch.LongTensor([pred.shape[0]] * batch_size)

        _, preds = pred.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        decoded_strings = converter.decode(preds, preds_size)

        predicted_strings_all.extend(decoded_strings)
        val_string_targets.extend(y)

        cerT=getCer(hyp=decoded_strings,ref=y)
        werT=getWer(hyp=decoded_strings,ref=y)

        cer += cerT
        wer += werT 

        print("------------------------------------------------------------------")
        print(decoded_strings," | ",y)
        print("------------------------------------------------------------------")
        print("WER:{wer:.2f}%  | CER:{cer:.2f}%".format(wer = werT*100, cer = cerT*100))
        print("------------------------------------------------------------------")
        print()

    print("TOTAL")
    print("TOTAL WER:{wer:.2f}% TOTAL CER:{cer:.2f}%".format(wer = (wer/num)*100, cer = (cer/num)*100))

    
    
    

        
            

    



    
main()

