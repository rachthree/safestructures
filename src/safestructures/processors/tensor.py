"""Plugins to process tensors."""

import numpy as np

from safestructures.processors.base import TensorProcessor
from safestructures.utils.module import is_available


class NumpyProcessor(TensorProcessor):
    """Numpy array processor."""

    data_type = np.ndarray

    def to_cpu(self, tensor: np.ndarray) -> np.ndarray:
        """Passthrough since this is already a Numpy array."""
        return tensor

    def to_numpy(self, tensor: np.ndarray):
        """Passthrough since this is already a Numpy array."""
        return tensor


if is_available("torch"):
    import torch

    class TorchProcessor(TensorProcessor):
        """Pytorch tensor processor."""

        data_type = torch.Tensor

        def to_cpu(self, tensor: torch.Tensor) -> np.ndarray:
            """Overload `TensorProcessor.to_cpu`."""
            tensor = tensor.detach().cpu().contiguous()
            if torch.is_floating_point(tensor):
                tensor = tensor.float()
            return tensor

        def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
            """Overload `TensorProcessor.to_numpy`."""
            return tensor.numpy()


if is_available("tensorflow"):
    import tensorflow as tf
    from tensorflow.python.framework.ops import EagerTensor

    class TFProcessor(TensorProcessor):
        """TensorFlow tensor processor."""

        data_type = EagerTensor

        def to_cpu(self, tensor: tf.Tensor) -> np.ndarray:
            """Overload `TensorProcessor.to_cpu`."""
            if tensor.device != "CPU:0":
                with tf.device("CPU:0"):
                    tensor = tf.identity(tensor)
            return tensor

        def to_numpy(self, tensor: tf.Tensor) -> np.ndarray:
            """Overload `TensorProcessor.to_numpy`."""
            return tensor.numpy()
