import os

import torch as th
from torchvision import transforms
from torchvision.transforms.transforms import Grayscale
from nnet.dataloader import PeroDataset
from torchvision.utils import save_image
from tqdm import tqdm


def pad_dataset(annotation, images, pad_path):
    img_transforms = th.nn.Sequential(
            Grayscale(1)
        )
    dataset = PeroDataset(annotation, images, width = 1810, transform=img_transforms)
    
    if not os.path.isdir(pad_path):
        os.mkdir(pad_path)
    keys = dataset.get_keys()
    for id, (img, _) in enumerate(tqdm(dataset)):
        img = img.float()/th.max(img)
        save_image(img, f"{pad_path}/{keys[id]}".strip())

if __name__ == "__main__":
    print("Starting pipeline:")
    print("Inicialize dataset -> pad dataset to same width -> create lmdb database files -> train -> test")
    print("Initialize dataset and pad it")
    pad_dataset( "../dataset/train.easy", "../dataset/lines", "../dataset/padlines_lines")
    pad_dataset( "../dataset/valid.easy", "../dataset/lines", "../dataset/padlines_lines")
