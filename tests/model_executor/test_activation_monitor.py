# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for activation monitors."""

import tempfile
from pathlib import Path

import pytest
import torch

from vllm.config.activation_monitor import (
    ActivationMonitorConfig,
    ActivationType,
    AggregationMethod,
    ProbeType,
)
from vllm.model_executor.layers.activation_monitor import (
    ActivationMonitor,
    LogisticProbe,
    MLPProbe,
    ProbeConfig,
)


class TestProbeConfig:
    """Tests for ProbeConfig dataclass."""

    def test_probe_config_creation(self):
        """Test that ProbeConfig can be created with all parameters."""
        config = ProbeConfig(
            probe_type="logistic",
            input_dim=4096,
            num_classes=2,
            hidden_dim=128,
            activation="gelu",
            dropout=0.1,
        )
        assert config.probe_type == "logistic"
        assert config.input_dim == 4096
        assert config.num_classes == 2
        assert config.hidden_dim == 128
        assert config.activation == "gelu"
        assert config.dropout == 0.1

    def test_probe_config_default_values(self):
        """Test ProbeConfig default values."""
        config = ProbeConfig(
            probe_type="mlp",
            input_dim=2048,
            num_classes=3,
        )
        assert config.hidden_dim == 128
        assert config.activation == "gelu"
        assert config.dropout == 0.0


class TestLogisticProbe:
    """Tests for LogisticProbe layer."""

    def test_logistic_probe_forward(self):
        """Test LogisticProbe forward pass."""
        input_dim = 128
        num_classes = 2
        batch_size = 4

        probe = LogisticProbe(input_dim=input_dim, num_classes=num_classes)
        x = torch.randn(batch_size, input_dim)

        output = probe(x)

        assert output.shape == (batch_size, num_classes)

    def test_logistic_probe_single_class(self):
        """Test LogisticProbe with single output (binary classification)."""
        input_dim = 64
        num_classes = 1
        batch_size = 8

        probe = LogisticProbe(input_dim=input_dim, num_classes=num_classes)
        x = torch.randn(batch_size, input_dim)

        output = probe(x)

        assert output.shape == (batch_size, num_classes)


class TestMLPProbe:
    """Tests for MLPProbe layer."""

    def test_mlp_probe_forward(self):
        """Test MLPProbe forward pass."""
        input_dim = 256
        hidden_dim = 64
        num_classes = 3
        batch_size = 4

        probe = MLPProbe(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            activation="gelu",
            dropout=0.0,
        )
        x = torch.randn(batch_size, input_dim)

        output = probe(x)

        assert output.shape == (batch_size, num_classes)

    def test_mlp_probe_with_dropout(self):
        """Test MLPProbe with dropout enabled."""
        input_dim = 128
        hidden_dim = 32
        num_classes = 2
        batch_size = 4

        probe = MLPProbe(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            activation="relu",
            dropout=0.5,
        )
        probe.train()  # Enable dropout
        x = torch.randn(batch_size, input_dim)

        output = probe(x)

        assert output.shape == (batch_size, num_classes)

    @pytest.mark.parametrize("activation", ["gelu", "relu", "tanh", "silu"])
    def test_mlp_probe_activations(self, activation):
        """Test MLPProbe with different activation functions."""
        input_dim = 64
        hidden_dim = 32
        num_classes = 2

        probe = MLPProbe(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            activation=activation,
            dropout=0.0,
        )
        x = torch.randn(2, input_dim)

        output = probe(x)

        assert output.shape == (2, num_classes)


class TestActivationMonitor:
    """Tests for ActivationMonitor module."""

    def test_activation_monitor_logistic(self):
        """Test ActivationMonitor with logistic probe."""
        config = ProbeConfig(
            probe_type="logistic",
            input_dim=128,  # 2 layers * 64 hidden_size
            num_classes=2,
        )
        monitor = ActivationMonitor(
            probe_config=config,
            device="cpu",
            dtype=torch.float32,
        )

        # Simulate aux_hidden_states from 2 layers
        aux_hidden_states = [
            torch.randn(4, 64),  # Layer 1: [batch, hidden_size]
            torch.randn(4, 64),  # Layer 2: [batch, hidden_size]
        ]

        output = monitor(aux_hidden_states)

        assert output.shape == (4, 2)  # [batch, num_classes]

    def test_activation_monitor_mlp(self):
        """Test ActivationMonitor with MLP probe."""
        config = ProbeConfig(
            probe_type="mlp",
            input_dim=192,  # 3 layers * 64 hidden_size
            num_classes=3,
            hidden_dim=32,
        )
        monitor = ActivationMonitor(
            probe_config=config,
            device="cpu",
            dtype=torch.float32,
        )

        # Simulate aux_hidden_states from 3 layers
        aux_hidden_states = [
            torch.randn(8, 64),
            torch.randn(8, 64),
            torch.randn(8, 64),
        ]

        output = monitor(aux_hidden_states)

        assert output.shape == (8, 3)

    def test_activation_monitor_preallocated_output(self):
        """Test ActivationMonitor with pre-allocated output tensor."""
        config = ProbeConfig(
            probe_type="logistic",
            input_dim=64,
            num_classes=2,
        )
        monitor = ActivationMonitor(
            probe_config=config,
            device="cpu",
            dtype=torch.float32,
        )

        aux_hidden_states = [torch.randn(4, 64)]
        out = torch.zeros(4, 2)

        result = monitor(aux_hidden_states, out=out)

        assert result is out  # Should use the pre-allocated buffer
        assert not torch.all(result == 0)  # Should have been written to


class TestActivationMonitorConfig:
    """Tests for ActivationMonitorConfig."""

    def test_config_creation(self):
        """Test ActivationMonitorConfig creation."""
        config = ActivationMonitorConfig(
            monitor_model="test/model",
            monitor_layers=(10, 20, 30),
            probe_type="mlp",
            hidden_dim=64,
            num_classes=3,
        )
        assert config.monitor_model == "test/model"
        assert config.monitor_layers == (10, 20, 30)
        assert config.probe_type == "mlp"
        assert config.hidden_dim == 64
        assert config.num_classes == 3

    def test_config_compute_hash(self):
        """Test that config hash is consistent."""
        config1 = ActivationMonitorConfig(
            monitor_model="test/model",
            monitor_layers=(10, 20),
            probe_type="logistic",
            num_classes=2,
        )
        config2 = ActivationMonitorConfig(
            monitor_model="test/model",
            monitor_layers=(10, 20),
            probe_type="logistic",
            num_classes=2,
        )
        config3 = ActivationMonitorConfig(
            monitor_model="test/model",
            monitor_layers=(10, 20),
            probe_type="mlp",  # Different probe type
            num_classes=2,
        )

        assert config1.compute_hash() == config2.compute_hash()
        assert config1.compute_hash() != config3.compute_hash()

    def test_config_default_values(self):
        """Test ActivationMonitorConfig default values."""
        config = ActivationMonitorConfig(monitor_model="test/model")
        assert config.monitor_layers is None
        assert config.probe_type == "logistic"
        assert config.hidden_dim == 128
        assert config.num_classes == 1
        assert config.activation == "gelu"
        assert config.dropout == 0.0
        assert config.per_token_scores is True
        assert config.aggregation_method == "mean"


class TestActivationMonitorWeightLoading:
    """Tests for weight loading functionality."""

    def test_load_safetensors_weights(self):
        """Test loading weights from safetensors format."""
        # Create a probe and save its weights
        config = ProbeConfig(
            probe_type="logistic",
            input_dim=64,
            num_classes=2,
        )
        monitor = ActivationMonitor(
            probe_config=config,
            device="cpu",
            dtype=torch.float32,
        )

        # Save the weights
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            weights_path = Path(f.name)

        try:
            from safetensors.torch import save_file

            # Save probe weights with the expected key prefix
            state_dict = {
                "probe.weight": monitor.probe.linear.weight,
                "probe.bias": monitor.probe.linear.bias,
            }
            save_file(state_dict, weights_path)

            # Create a new monitor and load the weights
            monitor2 = ActivationMonitor(
                probe_config=config,
                device="cpu",
                dtype=torch.float32,
            )
            monitor2.load_safetensors_weights(weights_path)

            # Verify the weights match
            assert torch.allclose(
                monitor.probe.linear.weight, monitor2.probe.linear.weight
            )
            assert torch.allclose(monitor.probe.linear.bias, monitor2.probe.linear.bias)
        finally:
            weights_path.unlink(missing_ok=True)
