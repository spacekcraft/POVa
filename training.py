import argparse

import torch
from torchvision import transforms
from torchvision.transforms import Lambda, Resize, Grayscale
from torch.autograd import Variable

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
    parser = argparse.ArgumentParser(description='Script for training the model.')
    parser.add_argument('--batch-size', '-b', default=64, type=int)
    parser.add_argument('--epochs', '-e', default=1000, type=int)
    parser.add_argument('--learning-rate', '-lr', default=1e-3, type=float)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--resume', '-r', type=str, default=None, help="Path to the resumed checkpoitn")
    parser.add_argument('--train-annotation', '-ta', type=str, help="Path to the annotation file for train data")
    parser.add_argument('--validate-annotation', '-va', type=str, help="Path to the annotation file for validation data")
    parser.add_argument('--alphabet-annotation', '-aa', type=str, help="Path to the annotation file for loading alphabet")
    parser.add_argument('--image-path', '-ip', type=str, help="Path to the images")
    parser.add_argument('--checkpoint', '-ch', type=str, help="Path to the checkpoint dir")
    parser.add_argument('--comment', '-cm', type=str, default = "", help="Experiment comment")
    parser.add_argument('--run', '-rn', type=str, default = "./runs/run", help="Run")
    parser.add_argument('--lmdb', '-d', action='store_true')
    args = parser.parse_args()
    return args



def get_dataloaders(train_annotation, validate_annotation, image_path, batch_size, image_shape, use_lmdb=False, verbose=False):
    nchanels, image_h, image_w = image_shape

    if use_lmdb:
        transform = transforms.Compose([Resize((image_h, image_w)), transforms.Lambda(lambda x: x.repeat(3,1,1))]) # lambda transform image to 3 channels
        train_dataloader = make_lmdb_dataloader(
            train_annotation,
            batch_size,
            shuffle=True,
            transform=transform,
            verbose=verbose,
        )
        val_dataloader = make_lmdb_dataloader(
            validate_annotation,
            batch_size,
            shuffle=True,
            transform=transform,
            verbose=verbose,
        )
    else:
        img_tranforms = torch.nn.Sequential(
            Resize((image_h, image_w)),
            Grayscale(nchanels),
        )
        # converter = StrLabelConverter(ALPHABET)
        train_dataloader = make_dataloader(
            train_annotation,
            image_path,
            batch_size,
            shuffle=True,
            verbose=verbose,
            transform=img_tranforms
        )

        val_dataloader = make_dataloader(
            validate_annotation,
            image_path,
            batch_size,
            shuffle=True,
            verbose=verbose,
            transform=img_tranforms
        )

    return train_dataloader, val_dataloader

def model_init(*args, pretrained=False):
    if pretrained:
        print("Loaded pretrained model")
        return torch.load(SAVE_MODEL_PATH)

    #return CRNN(image_h, nchanels, NUMBER_OF_CLASSES, 256)
    return CRNN(*args)

def main():
    args = parse_args()

    alphabetLoader = PeroDataset(annotation_path = args.alphabet_annotation, img_path = "")
    ALPHABET = alphabetLoader.load_alphabet()

    nchanels, image_h, image_w = 1, 32, 512
    train_dataloader, val_dataloader = get_dataloaders(
        train_annotation = args.train_annotation,
        validate_annotation = args.validate_annotation,
        image_path = args.image_path,
        batch_size = args.batch_size,
        image_shape = (nchanels, image_h, image_w),
        use_lmdb=args.lmdb,
        verbose = args.verbose
    )

    model = CRNN(image_h, nchanels, NUMBER_OF_CLASSES, 256)
    
    trainer = Trainer(model = model, checkpoint = args.checkpoint, tensorboard_dir=args.run, comment=args.comment, alphabet=ALPHABET, learning_rate=args.learning_rate, verbose = args.verbose, resume_path=args.resume)
    trainer.run(train_dataloader, val_dataloader, num_epochs=args.epochs)

    return

if __name__ == "__main__":
    main()
