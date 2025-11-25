# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for activation monitors (probes) attached to LLM layers."""

import hashlib
from typing import Any, Literal

from pydantic import ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass
from typing_extensions import Self

from vllm.config.utils import config
from vllm.logger import init_logger

logger = init_logger(__name__)

ProbeType = Literal["logistic", "mlp"]
AggregationMethod = Literal["mean", "max", "last"]
ActivationType = Literal["relu", "gelu"]


@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ActivationMonitorConfig:
    """Configuration for activation monitors (probes) attached to LLM layers.

    Activation monitors are lightweight classification heads that can be attached
    to intermediate layers of an LLM to produce auxiliary scores alongside the
    normal next-token prediction. They are useful for tasks like toxicity
    detection, sentiment analysis, or other real-time content classification.

    The monitors are trained separately using the probelab library and loaded
    from HuggingFace repositories or local paths.
    """

    monitor_model: str = Field(default=...)
    """Path to activation monitor weights. Can be a HuggingFace repository
    (e.g., 'my-org/llama-toxicity-probe') or a local directory path."""

    monitor_layers: tuple[int, ...] | None = None
    """Which layers to capture activations from. If None, will be read from
    the probe config file. Layer indices are 0-based."""

    probe_type: ProbeType = "logistic"
    """Probe architecture type. Must match the trained probe:
    - 'logistic': Single linear layer (fastest, good for linear separability)
    - 'mlp': Multi-layer perceptron (more expressive)
    """

    hidden_dim: int = Field(default=128, ge=1)
    """Hidden dimension for MLP probes. Ignored for logistic probes."""

    num_classes: int = Field(default=1, ge=1)
    """Number of output classes. Use 1 for binary classification (sigmoid
    output) or N for multi-class (softmax output)."""

    activation: ActivationType = "gelu"
    """Activation function for MLP probes. Ignored for logistic probes."""

    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    """Dropout rate for MLP probes. Ignored for logistic probes."""

    per_token_scores: bool = True
    """Whether to return scores per token or aggregated per sequence.
    Per-token scores are useful for identifying specific problematic tokens."""

    aggregation_method: AggregationMethod = "mean"
    """How to aggregate scores when per_token_scores=False:
    - 'mean': Average over all tokens
    - 'max': Maximum score (most conservative for safety)
    - 'last': Score of the last token only
    """

    weights_file: str = "probe.safetensors"
    """Weight file name within the monitor_model directory.
    Supports .safetensors (preferred) and .pt formats."""

    config_file: str = "probe_config.json"
    """Configuration file name within the monitor_model directory."""

    def compute_hash(self) -> str:
        """Compute a hash for this config that affects the computation graph."""
        factors: list[Any] = []
        factors.append(self.monitor_model)
        factors.append(self.monitor_layers)
        factors.append(self.probe_type)
        factors.append(self.hidden_dim)
        factors.append(self.num_classes)
        factors.append(self.activation)
        factors.append(self.dropout)
        # per_token_scores and aggregation_method don't affect the graph
        hash_str = hashlib.md5(
            str(factors).encode(), usedforsecurity=False
        ).hexdigest()
        return hash_str

    @model_validator(mode="after")
    def _validate_config(self) -> Self:
        # Validate probe_type specific settings
        if self.probe_type == "logistic":
            if self.hidden_dim != 128:
                logger.warning(
                    "hidden_dim is ignored for logistic probes, "
                    "using single linear layer"
                )
            if self.dropout != 0.0:
                logger.warning("dropout is ignored for logistic probes")

        return self
