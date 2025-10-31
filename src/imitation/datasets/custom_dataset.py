from typing import Any, Optional, Sequence, Union, Dict
from dataclasses import dataclass
from importlib.resources import files

import numpy as np

import imitation
from .base import Dataset


@dataclass(kw_only=True)
class CustomDataset(Dataset):

    data: Any

    def get_eval_env(self):
        raise NotImplementedError()

    @staticmethod
    def from_resource_path(path):
        full_path = files(imitation) / path
        return CustomDataset(data=np.load(full_path))
