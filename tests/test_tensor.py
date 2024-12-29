"""Tests for the basic datatype processors."""

from typing import Callable
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf
import torch

from safestructures.processors.base import TensorProcessor
from safestructures.processors.tensor import NumpyProcessor, TFProcessor, TorchProcessor
from safestructures.serializer import Serializer

PROCESSOR_CLS_LIST = [NumpyProcessor, TorchProcessor, TFProcessor]


@pytest.fixture
def mock_serializer():
    """Provide mocked Serializer fixture."""
    mock_serializer = mock.MagicMock()
    assert hasattr(
        Serializer(), "tensors"
    ), "Test expects Serializer.tensors to exist, and this is no longer valid."
    mock_serializer.tensors = {}
    return mock_serializer


MAX_DIM = 8
MAX_DIM_SIZE = 4


def _generate_dims():
    """Generate random dimensions for test tensors."""
    return np.random.randint(1, MAX_DIM, size=np.random.randint(1, MAX_DIM_SIZE))


def _random_tensor_numpy():
    """Generate a random numpy tensor."""
    dims = _generate_dims()
    return np.random.rand(*dims)


def _check_cpu_numpy(tensor: np.ndarray):
    """Passthrough function since Numpy arrays are already on CPU."""
    pass


def _random_tensor_torch():
    """Generate a random torch tensor."""
    dims = _generate_dims()
    return torch.randn(*dims)


def _check_cpu_torch(tensor: torch.Tensor):
    """Check that the torch tensor is on CPU."""
    assert tensor.device == torch.device("cpu")


def _random_tensor_tf():
    """Generate a random numpy tensor."""
    dims = _generate_dims()
    return tf.random.uniform(shape=dims)


def _check_cpu_tf(tensor: tf.Tensor):
    """Check that the tensorflow tensor is on CPU."""
    assert "CPU:0" in tensor.device


def _check_torch_tensors(test_tensor: np.ndarray, expected_tensor: torch.Tensor):
    """Check that the test numpy tensor and expected torch tensor are equal."""
    torch.testing.assert_close(torch.from_numpy(test_tensor), expected_tensor)


def _check_tf_tensors(test_tensor: np.ndarray, expected_tensor: tf.Tensor):
    assert tf.math.reduce_all(
        tf.equal(tf.convert_to_tensor(test_tensor), expected_tensor)
    ), "Tensors are not equal"


N_TENSORS = 10
serialize_test_cases = [
    (NumpyProcessor, _random_tensor_numpy, _check_cpu_numpy, np.testing.assert_equal),
    (TorchProcessor, _random_tensor_torch, _check_cpu_torch, _check_torch_tensors),
    (TFProcessor, _random_tensor_tf, _check_cpu_tf, _check_tf_tensors),
]


@pytest.mark.parametrize(
    "processor_cls,random_tensor_fn,check_cpu_fn,is_equal_fn", serialize_test_cases
)
def test_serialize_tensor(
    mock_serializer,
    processor_cls: TensorProcessor,
    random_tensor_fn: Callable,
    check_cpu_fn: Callable,
    is_equal_fn: Callable,
):
    """Test tensor processor serialization."""
    processor = processor_cls(mock_serializer)
    test_tensors = []
    with (
        mock.patch.object(processor, "to_cpu", wraps=processor.to_cpu) as mock_to_cpu,
        mock.patch.object(
            processor, "to_numpy", wraps=processor.to_numpy
        ) as mock_to_numpy,
        mock.patch.object(
            processor, "process_tensor", wraps=processor.process_tensor
        ) as mock_process_tensor,
    ):
        for _ in range(N_TENSORS):
            mock_to_cpu.reset_mock()
            mock_to_numpy.reset_mock()
            mock_process_tensor.reset_mock()

            test_input = random_tensor_fn()
            test_tensors.append(test_input)

            processor.serialize(test_input)
            mock_to_cpu.assert_called_once()
            mock_to_numpy.assert_called_once()
            mock_process_tensor.assert_called_once()

    mock_serializer.serialize.assert_not_called()
    mock_serializer.deserialize.assert_not_called()

    for i in range(N_TENSORS):
        _expected_id = str(i)
        assert isinstance(mock_serializer.tensors[_expected_id], np.ndarray)
        check_cpu_fn(mock_to_numpy.call_args.args[0])
        is_equal_fn(mock_serializer.tensors[_expected_id], test_tensors[i])


@pytest.mark.parametrize("processor_cls", PROCESSOR_CLS_LIST)
def test_deserialize_tensor(mock_serializer, processor_cls: TensorProcessor):
    """Test tensor processor deserialization."""
    test_tensors = []
    for i in range(N_TENSORS):
        input_tensor = _random_tensor_numpy()
        mock_serializer.tensors[str(i)] = input_tensor
        test_tensors.append(input_tensor)

    processor = processor_cls(mock_serializer)
    for i in range(N_TENSORS):
        np.testing.assert_equal(processor.deserialize(str(i)), test_tensors[i])

    mock_serializer.serialize.assert_not_called()
    mock_serializer.deserialize.assert_not_called()
