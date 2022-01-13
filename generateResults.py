import argparse
from genericpath import exists

import os
from pathlib import Path
import subprocess

import torch as th
from torchvision.transforms import Lambda, Resize, Grayscale
from torch.autograd import Variable
from matplotlib import pyplot as plt


from tqdm import tqdm

from PIL import Image, ImageOps, ImageDraw, ImageFont 
import numpy as np

import pdb

from models.crnn import CRNN
from nnet.trainer import Trainer
from nnet.dataloader import PeroDataset, make_dataloader
from nnet.utils import StrLabelConverter
from nnet.cwer import cer, wer


def parse_args():
    parser = argparse.ArgumentParser(description='Script for eval the model.')
    parser.add_argument('--alphabet', '-al', required=True, nargs='+', type=str)
    parser.add_argument('--annotation', '-a', required=True, type=str)
    parser.add_argument('--images', '-i', required=True, type=str)
    parser.add_argument('--samples', '-s', default=100, type=int)
    parser.add_argument('--checkpoint', '-ch', required=True, type=str)
    parser.add_argument('--results', '-r', required=True, type=str)

    parser.add_argument('--labels', '-l', required=False, action='store_true')
    args = parser.parse_args()
    return args

def load_checkpoint(resume:str, model:th.nn.Module):

        if th.cuda.is_available():
            freeGpu = subprocess.check_output('nvidia-smi -q | grep "Minor\|Processes"| grep "None" -B1 | tr -d " " | cut -d ":" -f2 | sed -n "1p"', shell=True)
            if len(freeGpu) == 0: # if gpu not aviable use cpu
                raise RuntimeError("CUDA device unavailable...exist")
            dev = 'cuda:'+freeGpu.decode().strip()
            gpuid = (int(freeGpu.decode().strip()), )
            print("thorch GPU")
        else:  
            dev = "cpu"  
            print("thorch CPU")

        print(dev)
        device = th.device(dev)  

        if not os.path.exists(resume):
                raise FileNotFoundError(
                    "Could not find resume checkpoint: {}".format(resume))
        cpt = th.load(resume, map_location="cpu")
        cur_epoch = cpt["epoch"]
        print("Resume from checkpoint {}: epoch {:d}".format(
            resume, cur_epoch))
        # load nnet
        model.load_state_dict(cpt["model_state_dict"])
        model = model.to(device)
        return model, device, gpuid

def main():
    args = parse_args()
    
    nchanels, image_h, image_w = 1, 48, 512
    '''
    img_tranforms = torch.nn.Sequential(
            Resize((image_h, image_w)),
            Grayscale(nchanels),
        )'''

    ALPHABET = ""
    for annot in args.alphabet:
        alphabetLoader = PeroDataset(annotation_path = annot, img_path = "")
        ALPHABET += alphabetLoader.get_alphabet()
    ALPHABET = ''.join(sorted(set(ALPHABET)))
    print(f"ALPHABET: {ALPHABET}")
    NUMBER_OF_CLASSES = len(ALPHABET)

    dataloader = make_dataloader(
            args.annotation,
            args.images,
            batch_size=16,
            shuffle = False,
            verbose=True,
            num_workers=0,
            transform=None
        )

    dataset=PeroDataset(args.annotation,args.images,None,False, width=1810)

    converter = StrLabelConverter(ALPHABET)
    NUMBER_OF_CLASSES = len(ALPHABET)

    print("Dataset name:",args.annotation)
    print("Alphabet:'"+ALPHABET+"'")

    model = CRNN(image_h, nchanels, NUMBER_OF_CLASSES, 256)
    model, device, gpuid = load_checkpoint(args.checkpoint, model)
    print("LOADED!")

    
    

    print("Result file:",args.annotation+".result")
    f = open(args.results, "w")
    
    if(args.labels):
        path = os.path.dirname(os.path.abspath(__file__))
        if(os.path.isdir('results')):
            print("results found",path+"/results")   
        else:
            print("results created",path+"/results")     
            os.mkdir("results")

    with th.cuda.device(gpuid[0]):
        i = 0
        path = os.path.dirname(os.path.abspath(__file__))+'/results'
        for X, y in tqdm(dataloader):
            #X = X.unsqueeze(0)
            X = X.to(device)
            pred = model(X)
            
            batch_size = X.shape[0]
            preds_size = th.LongTensor([pred.shape[0]] * batch_size)
            _, preds = pred.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            decoded_strings_list = converter.decode(preds, preds_size)
            #pdb.set_trace()
            for id, decoded_strings in enumerate(decoded_strings_list):
                print(y[id]+" "+decoded_strings)

                if(args.labels):
                    # open image
                    img = Image.open(args.images+"/"+y[id])
                    color = "white"
                    _, width = img.size

                    myFont = ImageFont.truetype('FreeMono.ttf', int(width/3))
                    border = (0, 0, 0, int(width/2))
                    new_img = ImageOps.expand(img, border=border, fill=color)
                    draw = ImageDraw.Draw(new_img)
                    draw.text((5, width+int(0.1*width)), decoded_strings, fill=(0, 0, 0), font=myFont)

                    new_img.save(path+"/"+y)
                f.write(y[id]+" "+decoded_strings+"\n")
            

    f.close()

    
    
    

        
            

    



    
main()