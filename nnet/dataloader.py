# POVa
# xpavlu10
"""Functions and dataset class for creating dataset and dataloader for Pero dataset.
"""

from typing import Dict, Tuple
import torch as th
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
from tqdm import tqdm


def collate_variable_length(batch):
    imgs, labels = zip(*batch)
    return imgs, labels

def make_dataloader(annotation_path: str, img_path: str, batch_size: int, shuffle: bool, verbose: bool, transform: transforms = None, target_transform: transforms = None) -> DataLoader:
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
        collate_fn=collate_variable_length
    )
    if verbose:
        print("Done")
    return loader


class PeroDataset(Dataset):
    def __init__(self, annotation_path: str, img_path: str, transform: transforms = None, target_transform: transforms = None, verbose: bool = True):
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
        if self._verbose:
            print(
                f"PeroDataset: Loading annotations from {annotation_path}...")
        try:
            self._annotation = self._load_annotation(annotation_path)
        except Exception as e:
            raise Exception(f"Pero dataset: {e}")

        self._keys = list(self._annotation.keys())
        self._img_path = img_path
        self._transform = transform
        self._target_transform = target_transform

        if self._verbose:
            print(
                f"PeroDataset: Loaded annotations for {len(self)} images on path {img_path}.")

    def _load_annotation(self, path: str) -> Dict:
        """Load annotation file.

        Args:
            path (str): Path to the annotation file.

        Raises:
            e: If annotation file is not on the path, raise error.

        Returns:
            Dict: Loaded annotations.
        """
        try:
            with open(path, "r") as f:
                annotation = dict()
                for line in tqdm(f.readlines()) if self._verbose else f.readlines():
                    splitted = line.split(" ", maxsplit=1)
                    annotation[splitted[0]] = splitted[-1]
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
        image = read_image(f"{self._img_path}/{self._keys[idx]}")
        if self._transform:
            image = self._transform(image)
        annotation = self._annotation[self._keys[idx]]
        if self._target_transform:
            annotation = self._target_transform(annotation)
        return image, annotation
