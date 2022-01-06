import os

import torch as th
from nnet.dataloader import PeroDataset
from torchvision.utils import save_image
from tqdm import tqdm


def pad_dataset(annotation, images, pad_path):
    dataset = PeroDataset(annotation, images, width = 1810)
    
    if not os.path.isdir(pad_path):
        os.mkdir(pad_path)
    keys = dataset.get_keys()
    for id, (img, _) in tqdm(enumerate(dataset)):
        img = img.float()/th.max(img)
        save_image(img, f"{pad_path}/{keys[id]}".strip())

if __name__ == "__main__":
    print("Starting pipeline:")
    print("Inicialize dataset -> pad dataset to same width -> create lmdb database files -> train -> test")
    print("Initialize dataset and pad it")
    pad_dataset( "./pero/valid.easy", "./pero/lines", "./pero/padlines_lines")
    