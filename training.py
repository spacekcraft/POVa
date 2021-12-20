import argparse

import torch
from torchvision.transforms import Lambda, Resize

from models.crnn import CRNN
from nnet.dataloader import make_dataloader
from nnet.utils import StrLabelConverter
from nnet.settings import (
    PERO_DATASET_PATH,
    PERO_ANNOTATIONS_PATH,
    ALPHABET,
    NUMBER_OF_CLASSES
)

def parse_args():
    parser = argparse.ArgumentParser(description='Script for training the model.')
    parser.add_argument('--batch-size', '-b', default=64, type=int)
    parser.add_argument('--epochs', '-e', default=1000, type=int)
    parser.add_argument('--learning-rate', '-l', default=1e-3, type=float)
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    return args


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def main():
    args = parse_args()

    nchanels, image_h, image_w = 3, 32, 100

    converter = StrLabelConverter(ALPHABET)
    train_dataloader = make_dataloader(
        PERO_ANNOTATIONS_PATH,
        PERO_DATASET_PATH,
        args.batch_size,
        shuffle=True,
        verbose=args.verbose,
        transform=Resize((image_h, image_w)),
        target_transform=Lambda(lambda y: converter.encode(y))
    )

    model = CRNN(image_h, nchanels, NUMBER_OF_CLASSES, 256)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.CTCLoss(zero_infinity=True)

    for t in range(args.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)

    torch.save(model, 'crnn.pth')

    print("Training has finished")


if __name__ == "__main__":
    main()
