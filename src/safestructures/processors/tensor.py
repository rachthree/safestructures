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

        def to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
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

        def to_cpu(self, tensor: EagerTensor) -> EagerTensor:
            """Overload `TensorProcessor.to_cpu`."""
            dtype = tensor.dtype
            if dtype.is_floating:
                dtype = tf.float32
                tensor = tf.cast(tensor, dtype=dtype)

            if "CPU:0" not in tensor.device:
                with tf.device("CPU:0"):
                    tensor = tf.identity(tensor)

            return tensor

        def to_numpy(self, tensor: EagerTensor) -> np.ndarray:
            """Overload `TensorProcessor.to_numpy`."""
            return tensor.numpy()


if is_available("jax"):
    import jax
    import jax.numpy as jnp
    from jaxlib.xla_extension import ArrayImpl

    cpus = jax.devices("cpu")
    cpu_device = cpus[0]

    class JaxProcessor(TensorProcessor):
        """JAX array processor."""

        data_type = ArrayImpl

        def to_cpu(self, tensor: ArrayImpl) -> ArrayImpl:
            """Overload `TensorProcessor.to_cpu`."""
            tensor = jnp.copy(tensor)
            dtype = tensor.dtype
            if jnp.issubdtype(tensor.dtype, jnp.floating):
                dtype = jnp.float32
                tensor = tensor.astype(dtype)

            if tensor.device.platform != "cpu":
                tensor = jax.device_put(tensor, device=cpu_device)

            return tensor

        def to_numpy(self, tensor: ArrayImpl) -> np.ndarray:
            """Overload `TensorProcessor.to_numpy`."""
            return np.asarray(tensor)
