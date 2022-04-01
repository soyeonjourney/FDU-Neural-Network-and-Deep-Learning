import numpy as np
from typing import Callable, Optional
from latte.utils.data import Dataset
import os
import requests
import gzip
import matplotlib.pyplot as plt

#########################################################################################
#                                        Dataset                                        #
#########################################################################################


class VisionDataset(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.root = root
        self.train = train  # training set or test set
        self.transform = transform if transform is not None else lambda x: x
        self.target_transform = (
            target_transform if target_transform is not None else lambda x: x
        )

        self.data = None
        self.labels = None
        self._load_dataset()

    def __getitem__(self, index):
        assert np.isscalar(index), "Index must be a scalar"

        if self.labels is None:
            return self.transform(self.data[index]), None
        else:
            return (
                self.transform(self.data[index]),
                self.target_transform(self.labels[index]),
            )

    def _load_dataset(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)


class MNIST(VisionDataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, train, transform, target_transform)

    def _load_dataset(self):
        url = 'http://yann.lecun.com/exdb/mnist/'
        if self.train:
            data_filename = 'train-images-idx3-ubyte.gz'
            labels_filename = 'train-labels-idx1-ubyte.gz'
        else:
            data_filename = 't10k-images-idx3-ubyte.gz'
            labels_filename = 't10k-labels-idx1-ubyte.gz'

        data_filepath = get_data(self.root, url, data_filename)
        labels_filepath = get_data(self.root, url, labels_filename)

        self.data = self._load_data(data_filepath)
        self.labels = self._load_labels(labels_filepath)

    def _load_data(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data

    def _load_labels(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels

    def show(self, row: int = 10, col: int = 10):
        h, w = 28, 28
        img = np.zeros((h * row, w * col))

        for r in range(row):
            for c in range(col):
                idx = np.random.randint(len(self.data))
                img[r * h : (r + 1) * h, c * w : (c + 1) * w] = self.data[idx].reshape(
                    h, w
                )

        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.show()

    @staticmethod
    def label():
        labels = {
            0: '0',
            1: '1',
            2: '2',
            3: '3',
            4: '4',
            5: '5',
            6: '6',
            7: '7',
            8: '8',
            9: '9',
        }
        return labels


#########################################################################################
#                                        Utility                                        #
#########################################################################################


def get_data(root: str, url: str, filename: str) -> None:
    """
    Download a file from a url if it does not exist in the current directory.
    """
    filepath = os.path.join(root, filename)

    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        r = requests.get(url + filename)
        with open(filepath, 'wb') as f:
            f.write(r.content)
        print(f"Downloaded {filename}!")

    return filepath
