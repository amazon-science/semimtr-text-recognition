from typing import Iterator, Sequence, Tuple, Optional, List
import numpy as np

import torch
from torch.utils.data.sampler import Sampler
import torch.distributed as dist


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

    def __init__(self, dataset_weights: Sequence[float], dataset_sizes: List[int], adopt_to_ddp: bool = False) -> None:
        try:
            np.random.choice(len(dataset_sizes), p=dataset_weights)
        except ValueError as e:
            raise e
        self.dataset_weights = torch.Tensor(dataset_weights)
        self.dataset_sizes = dataset_sizes
        self.sum_cum = np.cumsum([0] + self.dataset_sizes)
        self.num_datasets = len(dataset_sizes)
        self.num_samples = int(max([ds_size / ds_weight for ds_size, ds_weight in zip(dataset_sizes, dataset_weights)]))
        self.epoch = 0
        self.ddp_mode = False
        if adopt_to_ddp:
            self._distributed_sampler()

    def _distributed_sampler(self):
        try:
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
        except:
            return
        self.ddp_mode = True
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = self.num_samples // self.num_replicas

    def __iter__(self) -> Iterator[int]:
        # deterministically shuffle based on epoch
        self.generator = torch.Generator()
        self.generator.manual_seed(self.epoch)

        if not self.ddp_mode:
            self.perm_lists = [EndlessGeneratePermutedIndices(ds_size, g) for ds_size in self.dataset_sizes]
        else:
            print(f"Init Sampler with rank of {self.rank} and num_replicas {self.num_replicas}")
            self.perm_lists = [
                EndlessGeneratePermutedIndicesNew(torch.arange(ds_size)[self.rank:ds_size:self.num_replicas],
                                                  self.epoch) for ds_size in self.dataset_sizes]
        return self

    def __next__(self) -> int:
        if all([perm_list.finished for perm_list in self.perm_lists]):
            raise StopIteration
        dataset_idx = torch.multinomial(torch.Tensor(self.dataset_weights), 1, generator=self.generator)
        return self.sum_cum[dataset_idx] + next(self.perm_lists[dataset_idx])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class EndlessGeneratePermutedIndices:
    def __init__(self, length: int, generator: torch.Generator = None) -> None:
        self.length = length
        self.finished = False
        self.generator = generator
        self._sample_perm()

    def _sample_perm(self) -> None:
        self.perm_list = torch.randperm(self.length, generator=self.generator).tolist()

    def __iter__(self):
        self.finished = False
        self._sample_perm()

    def __next__(self) -> int:
        if len(self.perm_list) == 0:
            self._sample_perm()
            self.finished = True
        return self.perm_list.pop()


class EndlessGeneratePermutedIndicesDistributed(EndlessGeneratePermutedIndices):
    def __init__(self, indices: torch.Tensor, generator: torch.Generator = None) -> None:
        self.indices = indices
        self.finished = False
        self.generator = generator
        self._sample_perm()

    def _sample_perm(self) -> None:
        self.perm_list = self.indices[torch.randperm(len(self.indices), generator=self.generator)].tolist()


class EndlessGeneratePermutedIndicesNew:
    def __init__(self, indices: torch.Tensor, epoch: int) -> None:
        self.indices = indices
        self.epoch = epoch
        self.__iter__()

    def _sample_perm(self) -> None:
        # torch.randperm(self.length, generator=self.generator).tolist()
        self.perm_list = self.indices[torch.randperm(len(self.indices), generator=self.generator)].tolist()

    def __iter__(self):
        self.generator = torch.Generator()
        self.generator.manual_seed(self.epoch)
        self.finished = False
        self._sample_perm()

    def __next__(self) -> int:
        if len(self.perm_list) == 0:
            self.finished = True
            self._sample_perm()
        return self.perm_list.pop()
