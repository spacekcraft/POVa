import argparse
from genericpath import exists

import os
from pathlib import Path

import torch as th
from torchvision.transforms import Lambda, Resize, Grayscale
from torch.autograd import Variable
from matplotlib import pyplot as plt




from PIL import Image, ImageOps, ImageDraw, ImageFont 
import numpy as np

import pdb

from models.crnn import CRNN
from nnet.trainer import Trainer
from nnet.dataloader import PeroDataset, make_dataloader, make_lmdb_dataloader
from nnet.utils import StrLabelConverter
from nnet.cwer import cer, wer
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
    parser.add_argument('--alphabet', '-al', required=True, type=str)
    parser.add_argument('--annotation', '-a', required=True, type=str)
    parser.add_argument('--images', '-i', required=True, type=str)
    parser.add_argument('--samples', '-s', default=100, type=int)
    parser.add_argument('--checkpoint', '-ch', required=True, type=str)

    parser.add_argument('--labels', '-l', required=False, action='store_true')
    args = parser.parse_args()
    return args

def load_checkpoint(resume:str, model:th.nn.Module):

        if th.cuda.is_available():  
            dev = "cuda:0" 
            print("thorch GPU")
        else:  
            dev = "cpu"  
            print("thorch CPU")

        print(dev)
        device = th.device(dev)  

        if not os.path.exists(resume):
                raise FileNotFoundError(
                    "Could not find resume checkpoint: {}".format(resume))
        cpt = th.load(resume, map_location=device)
        cur_epoch = cpt["epoch"]
        print("Resume from checkpoint {}: epoch {:d}".format(
            resume, cur_epoch))
        # load nnet
        model.load_state_dict(cpt["model_state_dict"])
        return model

def main():

    


    args = parse_args()
    
    nchanels, image_h, image_w = 1, 48, 512
    '''
    img_tranforms = torch.nn.Sequential(
            Resize((image_h, image_w)),
            Grayscale(nchanels),
        )'''

    alphabet=PeroDataset(args.alphabet,args.images,None,False, width=1810).get_alphabet()
    dataset=PeroDataset(args.annotation,args.images,None,False, width=1810)
    converter = StrLabelConverter(alphabet)
    NUMBER_OF_CLASSES = len(alphabet)

    print("Dataset name:",args.annotation)
    print("Alphabet:'"+alphabet+"'")

    model = CRNN(image_h, nchanels, NUMBER_OF_CLASSES, 256)
    model = load_checkpoint(args.checkpoint, model)
    print("LOADED!")
    val_string_targets = []
    predicted_strings_all = []

    sum_cer = 0
    sum_wer = 0 


    
    

    print("Result file:",args.annotation+".result")
    f = open(os.path.basename(args.annotation)+".result", "w")
    
    if(args.labels):
        path = os.path.dirname(os.path.abspath(__file__))
        if(os.path.isdir('results')):
            print("results found",path+"/results")   
        else:
            print("results created",path+"/results")     
            os.mkdir("results")


    i = 0
    path = os.path.dirname(os.path.abspath(__file__))+'/results'
    for X, y in dataset:
        X = X.unsqueeze(0)
        pred = model(X)
        
        batch_size = X.shape[0]
        preds_size = th.LongTensor([pred.shape[0]] * batch_size)

        _, preds = pred.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        decoded_strings = converter.decode(preds, preds_size)

        predicted_strings_all.extend(decoded_strings)
        val_string_targets.extend(y)

        cerT=cer(hypothesis=''.join(decoded_strings),reference=y)
        werT=wer(hypothesis=''.join(decoded_strings),reference=y)

        sum_cer += cerT
        sum_wer += werT 



        """
        print("------------------------------------------------------------------")
        print(decoded_strings," | ",y)
        print("------------------------------------------------------------------")
        print("WER:{wer:.2f}%  | CER:{cer:.2f}%".format(wer = werT*100, cer = cerT*100))
        print("------------------------------------------------------------------")
        """




        print(y+" "+decoded_strings)

        if(args.labels):
            # open image
            img = Image.open(args.images+"/"+y)
            color = "white"
            _, width = img.size

            myFont = ImageFont.truetype('FreeMono.ttf', int(width/3))
            border = (0, 0, 0, int(width/2))
            new_img = ImageOps.expand(img, border=border, fill=color)
            draw = ImageDraw.Draw(new_img)
            draw.text((5, width+int(0.1*width)), decoded_strings, fill=(0, 0, 0), font=myFont)

            new_img.save(path+"/"+y)

            



        f.write(y+" "+decoded_strings+"\n")
        

    f.close()

    #print("TOTAL")
    #print("TOTAL WER:{wer:.2f}% TOTAL CER:{cer:.2f}%".format(wer = (wer/num)*100, cer = (cer/num)*100))

    
    
    

        
            

    



    
main()

