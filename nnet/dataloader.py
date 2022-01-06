# POVa
# xpavlu10
"""Functions and dataset class for creating dataset and dataloader for Pero dataset.
"""

import io
from typing import Dict, List, Tuple

import lmdb
import torch as th

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

import pdb

def collate_variable_length(batch):
    imgs, labels = zip(*batch)
    return th.stack(imgs), labels

def make_dataloader(annotation_path: str, img_path: str, batch_size: int, shuffle: bool, verbose: bool, num_workers = 0, transform: transforms = None, target_transform: transforms = None) -> DataLoader:
    """Creates dataloader and dataset from given parameters.

    Args:
        annotation_path (str): Path to the annotation file.
        img_path (str): Path to images folder.
        batch_size (int): Batch size.
        shuffle (bool): If true dataloader will shuffle images in batches. For validation set to False.
        verbose (bool): Verbose prints.
        transform (transforms, optional): Transform that should be performed on the loaded images. Defaults to None.

    Returns:
        DataLoader: Created dataloader.
    """
    if verbose:
        print("Initializing PeroDataset...")
    dataset = PeroDataset(
        annotation_path=annotation_path,
        img_path=img_path,
        transform=transform,
        target_transform=target_transform,
        verbose=verbose
    )
    if verbose:
        print("Creating dataloader...")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_variable_length
    )
    if verbose:
        print("Done")
    return loader


class PeroDataset(Dataset):
    def __init__(self, annotation_path: str, img_path: str, transform: transforms = None, target_transform: transforms = None, verbose: bool = True, width:int = None):
        """Init dataset.

        Args:
            annotation_path (str): Path to the annotation file
            img_path (str): Path to images folder.
            transform (transforms, optional): Transform that should be performed on the loaded images. Defaults to None.
            verbose (bool, optional): Verbose prints. Defaults to True.

        Raises:
            Exception: [description]
        """
        self._verbose = verbose
        self._img_path = img_path
        if self._verbose:
            print(
                f"PeroDataset: Loading annotations from {annotation_path}...")
        try:
            self._annotation = self._load_annotation(annotation_path)
        except Exception as e:
            raise Exception(f"Pero dataset: {e}")

        self._keys = list(self._annotation.keys())
        self._transform = transform
        self._target_transform = target_transform
        self._alphabet = self._load_alphabet()


        self._max_width = width if width is not None else self._get_max_width()


        if self._verbose:
            print(
                f"PeroDataset: Loaded annotations for {len(self)} images on path {img_path}.")
    def get_keys(self)->List:
        return self._keys

    def get_alphabet(self)->List:
        """Get dataset alphabet

        Returns:
            List: Alphabet
        """        
        return self._alphabet

    def _load_alphabet(self) -> str:
        """Get unique chars in anotation.

        Returns:
            str: Dataset alphabet.
        """
        values=list(self._annotation.values())
        unique = set()
        for label in values:
            unique.update(set(label))
        	
        return ''.join(sorted(unique))

    def _get_max_width(self):
        max_width = 0
        hist = dict()
        for key in tqdm(self._keys) if self._verbose else self._keys:
            image = Image.open(f"{self._img_path}/{key}".strip())
            #pdb.set_trace()
            #max_width = max(max_width, image.size[0])
            if image.size[0] in hist.keys():
                hist[image.size[0]] += 1
            else:
                hist[image.size[0]] = 1
            if max_width < image.size[0]: 
                max_width = image.size[0]
                print(max_width)
        print(f"{hist}")
        pdb.set_trace()
        return max_width

    def _load_annotation(self, path: str) -> Tuple[Dict, int]:
        """Load annotation file ang get maximum width.

        Args:
            path (str): Path to the annotation file.

        Raises:
            e: If annotation file is not on the path, raise error.

        Returns:
            Tuple[Dict,int]: Loaded annotations and maximum width.
        """
        try:
            with open(path, "r") as f:
                annotation = dict()
                for line in tqdm(f.readlines()) if self._verbose else f.readlines():
                    splitted = line.split(" ", maxsplit=1)
                    annotation[splitted[0]] = splitted[-1].strip('\n')
                return annotation
        except FileNotFoundError as e:
            raise e

    def __len__(self) -> int:
        """Get dataset size.

        Returns:
            int: Dataset size.
        """
        return len(self._keys)

    def __getitem__(self, idx: int) -> Tuple[th.Tensor, str]:
        """Get image and label on index idx.

        Args:
            idx (int): Index of file and label.

        Returns:
            Tuple[th.Tensor, str]: Image, label.
        """
        
        image = read_image(f"{self._img_path}/{self._keys[idx]}".strip()) #UPRAVA PRE ODSTRANENIE KONCA RIADKU V CESTE K SUBORU - MINO
        image = th.nn.functional.pad(image, (0, self._max_width - image.shape[-1]), "constant", 0)
        if self._transform:
            image = self._transform(image)
        annotation = self._annotation[self._keys[idx]]
        if self._target_transform:
            annotation = self._target_transform(annotation)
        return image, annotation

def make_lmdb_dataloader(lmdb_data_path, batch_size, transform = None, target_transform = None, shuffle=False, verbose=False, num_workers  = 0):
    if verbose:
        print("Initializing PeroDataset...")
    dataset = PeroLmdbDataset(
        lmdb_data_path,
        transform=transform,
        target_transform=target_transform,
    )
    if verbose:
        print("Creating dataloader...")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers = num_workers,
        collate_fn=collate_variable_length
    )
    if verbose:
        print("Done")

    return loader

class PeroLmdbDataset(Dataset):

    def __init__(self, root=None, transform=None, target_transform=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode('utf-8')))
            self.nSamples = nSamples

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode('utf-8'))

            buf = io.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
                transform = transforms.ToTensor()
                img = transform(img)
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            if self.transform is not None:
                img = self.transform(img)

            label_key = 'label-%09d' % index
            label = txn.get(label_key.encode('utf-8')).decode('utf-8')

            if self.target_transform is not None:
                label = self.target_transform(label)

        return (img, label)
