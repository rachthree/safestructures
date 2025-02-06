"""Test example code from docs."""

import os
from pathlib import PosixPath
from typing import Callable

import tensorflow as tf
import torch
from tf_hooks import register_forward_hook
from torchvision.models.resnet import resnet50, ResNet50_Weights

from safestructures import load_file, save_file

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class IOHook:
    """Base class for forward hook testing."""

    is_equal_fn: Callable
    framework: str

    def __init__(self, layer_name: str, save_dir: PosixPath):
        """Construct the hook."""
        self.layer_name = layer_name
        self.save_dir = save_dir
        self.times_called = 0

    def __call__(self, module, inputs, outputs):
        """Save and check inputs and outputs."""
        input_filename = f"{self.layer_name}_inputs_{self.times_called}.safestructures"
        output_filename = (
            f"{self.layer_name}_outputs_{self.times_called}.safestructures"
        )
        input_save_file = self.save_dir / input_filename
        output_save_file = self.save_dir / output_filename
        save_file(inputs, input_save_file)
        save_file(outputs, output_save_file)

        loaded_inputs = load_file(input_save_file, framework=self.framework)
        loaded_outputs = load_file(output_save_file, framework=self.framework)

        for expected_tensor, loaded_tensor in zip(inputs, loaded_inputs):
            self.__class__.is_equal_fn(expected_tensor, loaded_tensor)

        self.__class__.is_equal_fn(loaded_outputs, outputs)

        self.times_called += 1

        return


class TorchIOHook(IOHook):
    """PyTorch test forward hook."""

    is_equal_fn = torch.testing.assert_close
    framework = "pt"


class TFIOHook(IOHook):
    """TensorFlow test forward hook."""

    is_equal_fn = tf.debugging.assert_near
    framework = "tf"

    def __call__(self, layer, args, kwargs, outputs):
        """Overload IOHook.__call__."""
        # No kwargs yet for these tests
        super().__call__(layer, args, outputs)
        return


def test_io_torch(tmp_path):
    """Test PyTorch example."""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.eval()

    for n, m in model.named_modules():
        layer_name = n if n else "model"

        io_hook = TorchIOHook(layer_name, tmp_path)
        m.register_forward_hook(io_hook)

    test_input = torch.randn(8, 3, 224, 224)

    with torch.no_grad():
        model(test_input)


def test_io_tf(tmp_path):
    """Test TensorFlow example."""
    model = tf.keras.applications.ResNet50(weights="imagenet")

    for layer in model.layers:
        io_hook = TFIOHook(layer.name, tmp_path)
        register_forward_hook(layer, io_hook)

    test_input = tf.random.uniform((8, 224, 224, 3), maxval=1)
    model(test_input)
