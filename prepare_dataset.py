import os
import argparse
import shutil
import sys

from tqdm import tqdm
import numpy as np
import cv2
import lmdb

from nnet.settings import (
    LMDB_DATA_OUTPUT_PATH_TRAIN,
    LMDB_DATA_OUTPUT_PATH_VALID,
    PERO_DATASET_PATH,
)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', type = str, help = 'path to file which contains the image path and label', required=True)
    parser.add_argument('--images', '-i', type = str, help = 'path to dir which contains images', required=True)
    parser.add_argument('--database', '-db', type = str, help = 'path to database', required=True)
    parser.add_argument('--valid', action='store_true')

    return parser.parse_args()


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False

    try:
        imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        imgH, imgW = img.shape[0], img.shape[1]
    except:
        return False
    else:
        if imgH * imgW == 0:
            return False

    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if type(k) == str:
                k = k.encode()
            if type(v) == str:
                v = v.encode()
            txn.put(k,v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    # If lmdb file already exists, remove it. Or the new data will add to it.
    if os.path.exists(outputPath):
        shutil.rmtree(outputPath)
        os.makedirs(outputPath)
    else:
        os.makedirs(outputPath)

    assert (len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in tqdm(range(nSamples)):
        imagePath = imagePathList[i]
        label = labelList[i]

        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            #print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    env.close()
    print('Created dataset with %d samples' % nSamples)


def read_data_from_file(file_path, images):
    image_path_list = []
    label_list = []
    with open(file_path) as f:
        for line in f:
            splitted = line.split(" ", maxsplit=1)

            image_path = splitted[0].strip()
            label = splitted[-1].strip('\n')
            image_path_list.append(f'{images}/{image_path}')
            label_list.append(label)

    return image_path_list, label_list


def main():
    args = parse_args()
    image_path_list, label_list = read_data_from_file(args.file, args.images)
    db_path = LMDB_DATA_OUTPUT_PATH_TRAIN if not args.valid else LMDB_DATA_OUTPUT_PATH_VALID
    createDataset(args.database, image_path_list, label_list)

if __name__ == '__main__':
    main()
