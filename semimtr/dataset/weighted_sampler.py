from typing import Iterator, Sequence, Tuple
import numpy as np

import torch
from torch.utils.data.sampler import Sampler


class WeightedDatasetRandomSampler(Sampler[int]):
    r"""Samples datasets from ``[0,..,len(weights)-1]`` with given probabilities (weights),
    and provide a random index for the chosen dataset.
    Args:
        dataset_weights (sequence)   : a sequence of weights, necessary summing up to one
        dataset_sizes (sequence): size of each dataset
    Example:
        >>> WeightedDatasetRandomSampler([0.2, 0.8], [1, 7])
        [(1, 6),(1, 2),(1, 0),(0, 0),(1, 5),(1, 3),(1, 1),(0, 0),(1, 4)]
    """

    def __init__(self, dataset_weights: Sequence[float], dataset_sizes: Sequence[int]) -> None:
        try:
            np.random.choice(len(dataset_sizes), p=dataset_weights)
        except ValueError as e:
            raise e
        self.dataset_weights = dataset_weights
        self.dataset_sizes = dataset_sizes
        self.sum_cum = np.cumsum([0] + self.dataset_sizes)
        self.num_datasets = len(dataset_sizes)

    def __iter__(self) -> Iterator[int]:
        self.perm_lists = [EndlessGeneratePermutedIndices(ds_size) for ds_size in self.dataset_sizes]
        return self

    def __next__(self) -> int:
        if all([perm_list.finished for perm_list in self.perm_lists]):
            raise StopIteration
        dataset_idx = np.random.choice(self.num_datasets, p=self.dataset_weights)
        return self.sum_cum[dataset_idx] + next(self.perm_lists[dataset_idx])

    def __len__(self) -> int:
        return int(max([ds_size / ds_weight for ds_size, ds_weight in zip(self.dataset_sizes, self.dataset_weights)]))


class EndlessGeneratePermutedIndices:
    def __init__(self, length: int) -> None:
        self.length = length
        self.finished = False
        self._sample_perm()

    def _sample_perm(self) -> None:
        self.perm_list = torch.randperm(self.length).tolist()

    def __iter__(self):
        self.finished = False
        self._sample_perm()

    def __next__(self) -> int:
        if len(self.perm_list) == 0:
            self._sample_perm()
            self.finished = True
        return self.perm_list.pop()
