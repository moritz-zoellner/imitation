from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple, Sequence, Union


@dataclass
class BaseAlgorithm:

    def get_make_policy(self, *, env) -> Callable:
        raise NotImplementedError("get_make_policy should be implemented by subclass")

    def train(self, *, run_config, env, dataset, progress_fn) -> Tuple[Callable, Any, dict[str, Any]]:
        raise NotImplementedError("train function should be implemented by subclass")
