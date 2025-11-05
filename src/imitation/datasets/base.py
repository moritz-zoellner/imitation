from dataclasses import dataclass


@dataclass
class BaseDataset:

    batch_size: int = 256

    def sample(self):
        raise NotImplementedError("sample not implemented")
