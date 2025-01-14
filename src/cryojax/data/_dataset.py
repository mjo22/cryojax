"""Functionality in `cryojax` for representing datasets."""

import abc
import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class AbstractDataset(metaclass=abc.ABCMeta):
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

    !!! question "How do I implement an `AbstractDataset`?"

        Implementing an `AbstractDataset` is not like implementing
        most other classes in `cryojax`, which are `equinox.Module`s.
        An `equinox.Module` is just a pytree, so it can be safely
        passed to `jax` transformations. However, an `AbstractDataset`
        can *not* be passed to `jax` transformations. Therefore, it is
        not a pytree. Rather, it is a [frozen dataclass](https://docs.python.org/3/library/dataclasses.html#frozen-instances).

        There are gotchas when implementing frozen dataclasses. For example,
        the following will result in a `FrozenInstanceError`:

        ```python
        import dataclasses

        @dataclasses.dataclass(frozen=True)
        class BuggyFrozenDataclass:

            a: int

            def __init__(self, a: int):
                self.a = a

        BuggyFrozenDataclass(10)  # Will raise the FrozenInstanceError
        ```

        As of the time of writing this, frozen dataclasses must use the following
        syntax to set attributes in a custom `__init__`:

        ```python
        import dataclasses

        @dataclasses.dataclass(frozen=True)
        class CorrectFrozenDataclass:

            a: int

            def __init__(self, a: int):
                object.__setattr__(self, "a", a)

        example = CorrectFrozenDataclass(10)  # Will not raise a FrozenInstanceError
        print(example.a)  # This will print a, which here was 10
        ```
    """

    @abc.abstractmethod
    def __init__(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, index) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError
