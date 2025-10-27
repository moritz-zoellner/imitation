from typing import Any, Optional, Sequence, Union, Dict
from dataclasses import dataclass

import minari

from .base import Dataset


@dataclass(kw_only=True)
class MinariDataset(Dataset):

    dataset_impl: Any

    @staticmethod
    def from_name(name):

        dataset_impl = minari.load_dataset(name)

        return MinariDataset(dataset_impl=dataset_impl)
