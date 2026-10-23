"""Random reward smoke implementation for the world-model RL setup.

This module intentionally ignores the rollout contents and returns a uniformly
random reward in ``[0, 1)``.  It exists to smoke-test the miles GSPO pipeline
without depending on the in-cluster judge or similarity services.
"""

from __future__ import annotations

import random
from typing import Any


async def custom_rm(args: Any, sample: Any, **kwargs: Any) -> Any:
    """miles custom reward entry point with random rewards.

    ``sample`` may be a single sample or a list of samples; the batched form is
    used for group reward scoring and the single form for normal rollout
    scoring.
    """
    if isinstance(sample, (list, tuple)):
        return [random.random() for _ in sample]
    return random.random()
