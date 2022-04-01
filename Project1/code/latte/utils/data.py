import numpy as np


class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError


class DataLoader:
    def __init__(self, dataset: 'Dataset', batch_size: int, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_iter = len(self.dataset) // self.batch_size
        self.iter = 0
        self.reset()

    def reset(self):
        self.iter = 0
        if self.shuffle:
            self.indices = np.random.permutation(len(self.dataset))
        else:
            self.indices = np.arange(len(self.dataset))

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter == self.max_iter:
            self.reset()
            raise StopIteration
        else:
            self.iter += 1
            indices = self.indices[
                self.iter * self.batch_size : (self.iter + 1) * self.batch_size
            ]
            return self.dataset[indices]
