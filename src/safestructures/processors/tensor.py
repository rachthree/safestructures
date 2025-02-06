"""Plugins to process tensors."""

import numpy as np

from safestructures.processors.base import TensorProcessor
from safestructures.utils.module import is_available


class NumpyProcessor(TensorProcessor):
    """Numpy array processor."""

    data_type = np.ndarray

    def to_numpy(self, tensor: np.ndarray):
        """Passthrough since this is already a Numpy array."""
        return tensor


if is_available("torch"):
    import torch

    class TorchProcessor(TensorProcessor):
        """Pytorch tensor processor."""

        data_type = torch.Tensor

        def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
            """Overload `TensorProcessor.to_numpy`."""
            tensor = tensor.detach().contiguous()
            if torch.is_floating_point(tensor):
                tensor = tensor.float()

            return tensor.numpy()


if is_available("tensorflow"):
    import tensorflow as tf
    from tensorflow.python.framework.ops import EagerTensor

    class TFProcessor(TensorProcessor):
        """TensorFlow tensor processor."""

        data_type = EagerTensor

        def to_numpy(self, tensor: EagerTensor) -> np.ndarray:
            """Overload `TensorProcessor.to_numpy`."""
            dtype = tensor.dtype
            if dtype.is_floating:
                dtype = tf.float32
                tensor = tf.cast(tensor, dtype=dtype)

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

        def to_numpy(self, tensor: ArrayImpl) -> np.ndarray:
            """Overload `TensorProcessor.to_numpy`."""
            tensor = jnp.copy(tensor)
            dtype = tensor.dtype
            if jnp.issubdtype(tensor.dtype, jnp.floating):
                dtype = jnp.float32
                tensor = tensor.astype(dtype)
            return np.asarray(tensor)
