from dataclasses import dataclass, field
from typing import Dict, Sequence

import torch


@dataclass
class AlgorithmOutput:
    posterior_samples: Sequence[torch.Tensor]
    num_simulations: int
    validation_loss: Sequence[float] = field(default_factory=lambda: [float("nan")])
    avg_log_ratio: float = field(default=float("nan"))
    state_dicts: dict[int, dict] = field(default_factory=dict)
    log_prob_true_parameters: Sequence[float] = field(
        default_factory=lambda: [float("nan")] * 10
    )
