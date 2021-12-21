import argparse

import torch
from torchvision.transforms import Lambda, Resize, Grayscale
from torch.autograd import Variable

from models.crnn import CRNN
from nnet.dataloader import make_dataloader
from nnet.trainer import Trainer
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
)


def parse_args():
    parser = argparse.ArgumentParser(description='Script for training the model.')
    parser.add_argument('--batch-size', '-b', default=64, type=int)
    parser.add_argument('--epochs', '-e', default=1000, type=int)
    parser.add_argument('--learning-rate', '-l', default=1e-3, type=float)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--resume', '-r', type=str, default=None, help="Path to the resumed checkpoitn")
    parser.add_argument('--train-annotation', '-ta', type=str, help="Path to the annotation file for train data")
    parser.add_argument('--validate-annotation', '-va', type=str, help="Path to the annotation file for validation data")
    parser.add_argument('--image-path', '-ip', type=str, help="Path to the images")
    parser.add_argument('--checkpoint', '-ch', type=str, help="Path to the checkpoint dir")
    parser.add_argument('--comment', '-cm', type=str, default = "", help="Experiment comment")
    parser.add_argument('--run', '-rn', type=str, default = "./runs/run", help="Run")
    args = parser.parse_args()
    return args

def get_dataloaders(train_annotation, validate_annotation, image_path, batch_size, image_shape, verbose=False):
    nchanels, image_h, image_w = image_shape

    # converter = StrLabelConverter(ALPHABET)
    train_dataloader = make_dataloader(
        train_annotation,
        image_path,
        batch_size,
        shuffle=True,
        verbose=verbose,
        transform=torch.nn.Sequential(
            Resize((image_h, image_w)),
            Grayscale(nchanels),
        ),
        # target_transform=Lambda(lambda y: converter.encode(y)) # TODO try to use it this way
    )

    val_dataloader = make_dataloader(
        validate_annotation,
        image_path,
        batch_size,
        shuffle=True,
        verbose=verbose,
        transform=torch.nn.Sequential(
            Resize((image_h, image_w)),
            Grayscale(nchanels),
        ),
        # target_transform=Lambda(lambda y: converter.encode(y)) # TODO
    )

    return train_dataloader, val_dataloader

def train_loop(dataloader, model, loss_fn, optimizer):
    converter = StrLabelConverter(ALPHABET)
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        t, l = converter.encode(y)

        batch_size = X.shape[0]
        preds_size = torch.LongTensor([pred.shape[0]] * batch_size)
        loss = loss_fn(pred, t, preds_size, l)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % TRAINING_PROGRESS_LOG_INTERVAL == 0 and batch:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def validation_loop(epoch_number, dataloader, model):
    print(f"Validation")
    converter = StrLabelConverter(ALPHABET)
    size = len(dataloader.dataset)

    val_string_targets = []
    predicted_strings_all = []

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)

            batch_size = X.shape[0]
            preds_size = torch.LongTensor([pred.shape[0]] * batch_size)

            _, preds = pred.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            decoded_strings = converter.decode(preds, preds_size)

            predicted_strings_all.extend(decoded_strings)
            val_string_targets.extend(y)
            break

    word_error_rate = getWer(predicted_strings_all, val_string_targets)

    print(f"Word Error Rate: {word_error_rate}")
    #print(f"Character Error Rate: {char_error_rate}")


def model_init(*args, pretrained=False):
    if pretrained:
        print("Loaded pretrained model")
        return torch.load(SAVE_MODEL_PATH)

    #return CRNN(image_h, nchanels, NUMBER_OF_CLASSES, 256)
    return CRNN(*args)

def main():
    args = parse_args()

    nchanels, image_h, image_w = 1, 32, 512
    train_dataloader, val_dataloader = get_dataloaders(
        train_annotation = args.train_annotation,
        validate_annotation = args.validate_annotation,
        image_path = args.image_path,
        batch_size = args.batch_size,
        image_shape = (nchanels, image_h, image_w),
        verbose = args.verbose
    )

    model = CRNN(image_h, nchanels, NUMBER_OF_CLASSES, 256)
    
    trainer = Trainer(model = model, checkpoint = args.checkpoint, tensorboard_dir=args.run, comment=args.comment, alphabet=ALPHABET, learning_rate=args.learning_rate, verbose = args.verbose, resume_path=args.resume)
    trainer.run(train_dataloader, val_dataloader, num_epochs=args.epochs)

    return

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.CTCLoss(zero_infinity=True)

    for t in range(args.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        validation_loop(t, train_dataloader, model)
        

    torch.save(model, SAVE_MODEL_PATH)

    print("Training has finished")


if __name__ == "__main__":
    main()
