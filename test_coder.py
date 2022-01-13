import os

import torch as th
from torch.utils import data
from nnet.dataloader import PeroDataset, make_dataloader
from nnet.utils import StrLabelConverter, load_json
from torchvision.utils import save_image
import pdb

def test_coder():
    print("===============TEST CODER =================")
    dataset = PeroDataset("./pero/train.easy", "./pero/lines")
    alphabet = dataset.load_alphabet()
    print(f"Alphabet: '{alphabet}'")

    coder = StrLabelConverter(alphabet=alphabet, ignore_case=False)
    text, length = coder.encode("Ahoj jak se mas?")
    encoded = coder.decode(text, length)
    print(f"STR: {encoded}")

def test_batch_coder():
    print("===============TEST BATCH CODER =================")
    dataset = PeroDataset("./pero/train.easy", "./pero/lines")
    alphabet = dataset.load_alphabet()
    print(f"Alphabet: '{alphabet}'")

    dataloader = make_dataloader("./pero/train.easy", "./pero/lines", batch_size = 10, shuffle = False, verbose = True)
    
    coder = StrLabelConverter(alphabet=alphabet, ignore_case=False)
    _, text = list(dataloader)[0]
    pdb.set_trace()
    text, length = coder.encode(text)
    encoded = coder.decode(text, length)
    print(f"STR: {encoded}")

def test_dataset_x_dataloader_length():
    print("===============TEST DATASET X DATALOADER =================")
    dataset = PeroDataset("./pero/train.easy", "./pero/lines")
    dataloader = make_dataloader("./pero/train.easy", "./pero/lines", batch_size = 10, shuffle = False, verbose = True)
    print(f"Dataset: {len(dataset)} Dataloader: {len(dataloader)}")


def pad_dataset():
    dataset = PeroDataset("./pero/valid.easy", "./pero/lines")
    '''
    pad_path = "./pero/padlines_lines"
    if not os.path.isdir(pad_path):
        os.mkdir(pad_path)
    keys = dataset.get_keys()
    for id, (img, _) in enumerate(dataset):
        img = img.float()/th.max(img)
        save_image(img, f"{pad_path}/{keys[id]}".strip())
        pdb.set_trace()
    '''

if __name__ == "__main__":
    config = load_json("./pipeline.json")
    #test_coder()
    #test_dataset_x_dataloader_length()
    #print("\n\n\n\n\n\n\n\n")
    #test_batch_coder()
    pad_dataset()