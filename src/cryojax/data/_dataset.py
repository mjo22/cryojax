"""Functionality in `cryojax` for representing datasets."""

import abc
from typing import Generic, TypeVar


T = TypeVar("T")


class AbstractDataset(abc.ABC, Generic[T]):
    """An abstraction of a dataset in `cryojax`. To create an
    `AbstractDataset`, implement its `__init__`, `__getitem__`, and
    `__len__` methods.

    This follows the
    [`torch.utils.data.Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)
    API and can easily be wrapped into a pytorch `Dataset` with the
    following pattern:

    ```python
    import numpy as np
    from torch.utils.data import Dataset

    class CustomTorchDataset(Dataset):

        def __init__(cryojax_dataset: AbstractDataset):
            self.cryojax_dataset = cryojax_dataset

        def __getitem___(self, index) -> dict[str, Array]:
            particle_stack = self.cryojax_dataset[index]
            return dict(index=index, image_stack=np.asarray(particle_stack.image_stack))

        def __len__(self) -> int:
            return len(self.cryojax_dataset)
    ```

    JAX also includes packages for dataloaders, such as
    [`jax-dataloaders`](https://github.com/BirkhoffG/jax-dataloader/tree/main) and
    [`grain`](https://github.com/google/grain).
    """

    @abc.abstractmethod
    def __getitem__(self, index) -> T:
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def is_loading_on_cpu(self) -> bool:
        raise NotImplementedError

    @is_loading_on_cpu.setter
    @abc.abstractmethod
    def is_loading_on_cpu(self, value: bool):
        raise NotImplementedError
