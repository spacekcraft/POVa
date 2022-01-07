import os
import argparse
import shutil
import sys
import io

from tqdm import tqdm
import lmdb
import torch
import matplotlib.pyplot as plt

from nnet.settings import (
    LMDB_DATA_OUTPUT_PATH_TRAIN,
    LMDB_DATA_OUTPUT_PATH_VALID,
    PERO_DATASET_PATH,
)

from nnet.dataloader import PeroDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', type = str, help = 'path to file which contains the image path and label', required=True)
    parser.add_argument('--images', '-i', type = str, help = 'path to dir which contains images', required=True)
    parser.add_argument('--database', '-db', type = str, help = 'path to database', required=True)

    return parser.parse_args()


def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if type(k) == str:
                k = k.encode()
            if type(v) == str:
                v = v.encode()
            txn.put(k,v)


def create_dataset(outputPath, dataset):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        dataset       : dataset to store
    """
    # If lmdb file already exists, remove it. Or the new data will add to it.
    if os.path.exists(outputPath):
        shutil.rmtree(outputPath)
        os.makedirs(outputPath)
    else:
        os.makedirs(outputPath)

    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    for i, (image, label)  in enumerate(tqdm(dataset)):
        buffer = io.BytesIO()
        torch.save(image, buffer)
        buffer.seek(0)
        image_bin = buffer.read()

        imageKey = f'image-{i}'
        labelKey = f'label-{i}'
        cache[imageKey] = image_bin
        cache[labelKey] = label
        if i % 1000 == 0:
            write_cache(env, cache)
            cache = {}

    nSamples = len(dataset)
    cache['num-samples'] = str(nSamples)
    write_cache(env, cache)
    env.close()

    print('Created dataset with %d samples' % nSamples)


def main():
    args = parse_args()
    pero_dataset = PeroDataset(args.file, args.images)
    create_dataset(args.database, pero_dataset)

if __name__ == '__main__':
    main()
