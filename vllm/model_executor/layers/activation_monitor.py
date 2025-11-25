# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Activation monitor (probe) implementations for vLLM.

This module provides lightweight classification heads that can be attached
to intermediate layers of an LLM to produce auxiliary scores alongside
normal token generation. The probes are compatible with weights trained
using the probelab library.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ActivationMonitorConfig

logger = init_logger(__name__)


@dataclass
class ProbeConfig:
    """Configuration loaded from probe_config.json."""

    probe_type: str
    monitor_layers: tuple[int, ...]
    d_model: int  # Hidden size of base model
    hidden_dim: int = 128
    num_classes: int = 1
    activation: str = "gelu"
    dropout: float = 0.0

    @classmethod
    def from_file(cls, path: Path) -> "ProbeConfig":
        """Load probe configuration from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            probe_type=data["probe_type"],
            monitor_layers=tuple(data["monitor_layers"]),
            d_model=data["d_model"],
            hidden_dim=data.get("hidden_dim", 128),
            num_classes=data.get("num_classes", 1),
            activation=data.get("activation", "gelu"),
            dropout=data.get("dropout", 0.0),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "probe_type": self.probe_type,
            "monitor_layers": list(self.monitor_layers),
            "d_model": self.d_model,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "activation": self.activation,
            "dropout": self.dropout,
        }


class LogisticProbe(nn.Module):
    """Logistic regression probe - compatible with probelab.probes.Logistic.

    A simple linear classifier that maps hidden states to class probabilities.
    This is the fastest probe type and works well when classes are linearly
    separable in the activation space.
    """

    def __init__(self, input_dim: int, num_classes: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes, bias=True)
        self.num_classes = num_classes

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [num_tokens, input_dim]

        Returns:
            scores: [num_tokens, num_classes] probabilities
        """
        logits = self.linear(hidden_states)
        if self.num_classes == 1:
            return torch.sigmoid(logits)
        return F.softmax(logits, dim=-1)


class MLPProbe(nn.Module):
    """MLP probe - compatible with probelab.probes.MLP.

    A multi-layer perceptron classifier with one hidden layer. More expressive
    than logistic regression, suitable for non-linearly separable classes.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 1,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.num_classes = num_classes

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [num_tokens, input_dim]

        Returns:
            scores: [num_tokens, num_classes] probabilities
        """
        x = self.fc1(hidden_states)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        if self.num_classes == 1:
            return torch.sigmoid(logits)
        return F.softmax(logits, dim=-1)


class ActivationMonitor(nn.Module):
    """Main activation monitor that wraps probe architectures.

    Handles multi-layer input concatenation and provides a unified interface
    for computing probe scores from captured hidden states.

    This class is designed to be CUDA graph compatible when used with
    pre-allocated output buffers.
    """

    def __init__(
        self,
        probe_config: ProbeConfig,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.probe_config = probe_config
        self.monitor_layers = probe_config.monitor_layers
        self.num_layers = len(probe_config.monitor_layers)

        # Input dim = d_model * num_layers (concatenated hidden states)
        input_dim = probe_config.d_model * self.num_layers

        # Create appropriate probe architecture
        if probe_config.probe_type == "logistic":
            self.probe = LogisticProbe(
                input_dim=input_dim,
                num_classes=probe_config.num_classes,
            )
        elif probe_config.probe_type == "mlp":
            self.probe = MLPProbe(
                input_dim=input_dim,
                hidden_dim=probe_config.hidden_dim,
                num_classes=probe_config.num_classes,
                activation=probe_config.activation,
                dropout=probe_config.dropout,
            )
        else:
            raise ValueError(f"Unsupported probe type: {probe_config.probe_type}")

        self.to(device=device, dtype=dtype)

    def forward(
        self,
        aux_hidden_states: list[torch.Tensor],
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute monitor scores from captured hidden states.

        Args:
            aux_hidden_states: List of [num_tokens, d_model] tensors from
                monitored layers. Must have exactly self.num_layers tensors.
            out: Optional pre-allocated output tensor for CUDA graph
                compatibility. Shape should be [num_tokens, num_classes].

        Returns:
            scores: [num_tokens, num_classes] probability scores
        """
        assert len(aux_hidden_states) == self.num_layers, (
            f"Expected {self.num_layers} hidden states, "
            f"got {len(aux_hidden_states)}"
        )

        # Concatenate hidden states from all monitored layers
        # Each tensor: [num_tokens, d_model]
        # Combined: [num_tokens, d_model * num_layers]
        combined = torch.cat(aux_hidden_states, dim=-1)

        # Compute probe scores
        scores = self.probe(combined)

        # Write to pre-allocated buffer if provided (CUDA graph compatibility)
        if out is not None:
            out.copy_(scores)
            return out

        return scores

    def load_probelab_weights(self, weights_path: Path) -> None:
        """Load weights saved by probelab's probe.save() method.

        probelab saves the internal network state dict with keys like:
        - For Logistic: 'linear.weight', 'linear.bias'
        - For MLP: 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'

        This method maps those keys to our probe module structure.
        """
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

        # probelab may save with various prefixes, try to normalize
        mapped = {}
        for key, value in state_dict.items():
            # Remove potential prefixes from probelab internal classes
            new_key = key
            for prefix in ["_LogisticNetwork.", "_MLPNetwork.", "network."]:
                new_key = new_key.replace(prefix, "")
            mapped[f"probe.{new_key}"] = value

        self.load_state_dict(mapped, strict=True)
        logger.info(f"Loaded probe weights from {weights_path}")

    def load_safetensors_weights(self, weights_path: Path) -> None:
        """Load weights from safetensors format."""
        from safetensors.torch import load_file

        state_dict = load_file(weights_path)

        # Map keys to our structure
        mapped = {}
        for key, value in state_dict.items():
            # Remove potential prefixes
            new_key = key
            for prefix in ["_LogisticNetwork.", "_MLPNetwork.", "network.", "probe."]:
                new_key = new_key.replace(prefix, "")
            mapped[f"probe.{new_key}"] = value

        self.load_state_dict(mapped, strict=True)
        logger.info(f"Loaded probe weights from {weights_path}")


def load_activation_monitor(
    monitor_config: "ActivationMonitorConfig",
    model_hidden_size: int,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> ActivationMonitor:
    """Factory function to load an activation monitor from config.

    Downloads from HuggingFace if needed, loads config and weights.

    Args:
        monitor_config: The activation monitor configuration
        model_hidden_size: Hidden size of the base model (d_model)
        device: Device to load the monitor on
        dtype: Data type for the monitor weights

    Returns:
        Initialized ActivationMonitor with loaded weights
    """
    from huggingface_hub import snapshot_download

    monitor_path = Path(monitor_config.monitor_model)

    # Download from HuggingFace if not local
    if not monitor_path.exists():
        logger.info(f"Downloading monitor from {monitor_config.monitor_model}")
        local_dir = snapshot_download(
            repo_id=monitor_config.monitor_model,
            allow_patterns=["*.pt", "*.json", "*.safetensors"],
        )
        monitor_path = Path(local_dir)

    # Load probe config from file if it exists
    config_path = monitor_path / monitor_config.config_file
    if config_path.exists():
        probe_config = ProbeConfig.from_file(config_path)
        logger.info(f"Loaded probe config from {config_path}")
    else:
        # Construct from ActivationMonitorConfig
        if monitor_config.monitor_layers is None:
            raise ValueError(
                f"No probe config file found at {config_path} and "
                "monitor_layers not specified in ActivationMonitorConfig"
            )
        probe_config = ProbeConfig(
            probe_type=monitor_config.probe_type,
            monitor_layers=monitor_config.monitor_layers,
            d_model=model_hidden_size,
            hidden_dim=monitor_config.hidden_dim,
            num_classes=monitor_config.num_classes,
            activation=monitor_config.activation,
            dropout=monitor_config.dropout,
        )
        logger.info("Constructed probe config from ActivationMonitorConfig")

    # Override layers if specified in vLLM config
    if monitor_config.monitor_layers is not None:
        probe_config = ProbeConfig(
            probe_type=probe_config.probe_type,
            monitor_layers=monitor_config.monitor_layers,
            d_model=probe_config.d_model,
            hidden_dim=probe_config.hidden_dim,
            num_classes=probe_config.num_classes,
            activation=probe_config.activation,
            dropout=probe_config.dropout,
        )
        logger.info(
            f"Overriding monitor layers to {monitor_config.monitor_layers}"
        )

    # Create monitor
    monitor = ActivationMonitor(
        probe_config=probe_config,
        device=device,
        dtype=dtype,
    )

    # Load weights - try safetensors first, then .pt
    safetensors_path = monitor_path / monitor_config.weights_file.replace(
        ".pt", ".safetensors"
    )
    pt_path = monitor_path / monitor_config.weights_file

    if monitor_config.weights_file.endswith(".safetensors"):
        safetensors_path = monitor_path / monitor_config.weights_file
        pt_path = monitor_path / monitor_config.weights_file.replace(
            ".safetensors", ".pt"
        )

    if safetensors_path.exists():
        monitor.load_safetensors_weights(safetensors_path)
    elif pt_path.exists():
        monitor.load_probelab_weights(pt_path)
    else:
        raise FileNotFoundError(
            f"No weights found. Tried:\n"
            f"  - {safetensors_path}\n"
            f"  - {pt_path}"
        )

    return monitor
