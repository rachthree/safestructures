"""Test `Serializer`."""
import json
from unittest import mock

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.numpy import save_file

from safestructures.constants import (
    SCHEMA_FIELD,
    SCHEMA_VERSION,
    TYPE_FIELD,
    VALUE_FIELD,
    VERSION_FIELD,
)
from safestructures.defaults import DEFAULT_PROCESS_MAP
from safestructures.serializer import Serializer

MOCK_DEFAULT_PROCESS_MAP = {
    data_type: mock.MagicMock() for data_type in DEFAULT_PROCESS_MAP
}

FRAMEWORK_TENSOR_TYPE_MAP = {
    "np": np.ndarray,
    "pt": torch.Tensor,
}


class TestSerializer:
    """`Serializer` test cases."""

    def setup_method(self, method):
        """Set up before each test method."""
        self.mock_processor_instances = {
            data_type: mock.MagicMock() for data_type in DEFAULT_PROCESS_MAP
        }
        self.mock_default_process_map = {
            data_type: mock.MagicMock(
                return_value=self.mock_processor_instances[data_type]
            )
            for data_type in DEFAULT_PROCESS_MAP
        }

    def teardown_method(self, method):
        """Tear down after each test method."""
        for v in self.mock_default_process_map.values():
            v.reset_mock()
        for v in self.mock_processor_instances.values():
            v.reset_mock()

    @pytest.mark.parametrize("data_type", DEFAULT_PROCESS_MAP.keys())
    def test_serialize(self, data_type):
        """Test `Serializer.serialize`."""

        def mock_type_fn(x):
            return data_type

        serializer = Serializer()
        mock_deserialize = mock.MagicMock()
        with (
            mock.patch.object(serializer, "process_map", self.mock_default_process_map),
            mock.patch("safestructures.serializer.type", mock_type_fn),
            mock.patch.object(serializer, "deserialize", mock_deserialize),
        ):
            mock_data = mock.Mock()
            serializer.serialize(mock_data)

            self.mock_default_process_map[data_type].assert_called_once_with(serializer)
            self.mock_processor_instances[data_type].assert_called_once_with(mock_data)
            mock_deserialize.assert_not_called()

    @pytest.mark.parametrize("data_type", DEFAULT_PROCESS_MAP.keys())
    def test_deserialize(self, data_type):
        """Test `Serializer.deserialize`."""
        mock_get_data_type = mock.MagicMock(return_value=data_type)
        mock_serialize = mock.MagicMock()
        serializer = Serializer()
        with (
            mock.patch.object(serializer, "process_map", self.mock_default_process_map),
            mock.patch.object(serializer, "_get_data_type", mock_get_data_type),
            mock.patch.object(serializer, "serialize", mock_serialize),
        ):
            mock_type_str = "mock_type"
            mock_schema = {TYPE_FIELD: mock_type_str}
            serializer.deserialize(mock_schema)

            mock_get_data_type.assert_called_once_with(mock_type_str)
            self.mock_default_process_map[data_type].assert_called_once_with(serializer)
            self.mock_processor_instances[data_type].assert_called_once_with(
                mock_schema
            )
            mock_serialize.assert_not_called()

    def test_save_with_tensor(self, tmp_path):
        """Test `Serializer.save` with tensors."""
        mock_schema = {TYPE_FIELD: "mock_type", VALUE_FIELD: "mock_value"}
        test_tensor = np.random.rand(4, 3, 224, 224)
        test_tensor_id = "mock_tensor_id"
        mock_data = mock.Mock()
        test_other_metadata = {"other_field": "other_value"}

        def mock_serialize_with_tensor(self, data):
            self.tensors[test_tensor_id] = test_tensor
            return mock_schema

        file_name_with_tensor = "test_with_tensor.safestructures"

        # Test saving with tensor
        temp_with_tensor_path = tmp_path / file_name_with_tensor
        with mock.patch(
            "safestructures.serializer.Serializer.serialize", mock_serialize_with_tensor
        ):
            serializer = Serializer()
            serializer.save(
                mock_data, temp_with_tensor_path, metadata=test_other_metadata
            )

        with safe_open(temp_with_tensor_path, framework="np") as f:
            tensors = {}
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
            metadata = f.metadata()

        assert len(tensors) == 1
        np.testing.assert_allclose(tensors[test_tensor_id], test_tensor)

        assert metadata["other_field"] == "other_value"
        assert metadata[SCHEMA_FIELD] == json.dumps(mock_schema)
        assert metadata[VERSION_FIELD] == SCHEMA_VERSION

    def test_save_no_tensor(self, tmp_path):
        """Test `Serializer.save` without tensors."""
        mock_schema = {TYPE_FIELD: "mock_type", VALUE_FIELD: "mock_value"}
        mock_data = mock.Mock()
        test_other_metadata = {"other_field": "other_value"}

        def mock_serialize_no_tensor(self, data):
            return mock_schema

        file_name_no_tensor = "test_no_tensor.safestructures"

        # Test saving with no tensor
        temp_no_tensor_path = tmp_path / file_name_no_tensor
        with mock.patch(
            "safestructures.serializer.Serializer.serialize", mock_serialize_no_tensor
        ):
            serializer = Serializer()
            serializer.save(
                mock_data, temp_no_tensor_path, metadata=test_other_metadata
            )

        with safe_open(temp_no_tensor_path, framework="np") as f:
            tensors = {}
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
            metadata = f.metadata()

        assert len(tensors) == 1
        np.testing.assert_allclose(tensors["null"], np.array([0]))

        assert metadata["other_field"] == "other_value"
        assert metadata[SCHEMA_FIELD] == json.dumps(mock_schema)
        assert metadata[VERSION_FIELD] == SCHEMA_VERSION

    @pytest.mark.parametrize("framework", ["np", "pt"])
    def test_load(self, tmp_path, framework):
        """Test `Serialize.load`."""
        mock_schema = {TYPE_FIELD: "mock_type", VALUE_FIELD: "mock_value"}
        test_metadata = {SCHEMA_FIELD: json.dumps(mock_schema)}

        temp_file_path = tmp_path / "test.safetensors"
        tensor_id = "test_tensor_id"
        mock_tensor_dict = {tensor_id: np.array([1, 2, 3, 4])}
        save_file(mock_tensor_dict, temp_file_path, metadata=test_metadata)

        serializer = Serializer()
        mock_results = mock.Mock()
        mock_deserializer = mock.MagicMock(return_value=mock_results)
        with mock.patch.object(serializer, "deserialize", mock_deserializer):
            results = serializer.load(temp_file_path, framework=framework)
            mock_deserializer.assert_called_once_with(mock_schema)
            assert results == mock_results
            assert isinstance(
                serializer.tensors[tensor_id], FRAMEWORK_TENSOR_TYPE_MAP[framework]
            )
