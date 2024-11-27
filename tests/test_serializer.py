"""Test `Serializer`."""
from unittest import mock

import pytest

from safestructures import Serializer
from safestructures.constants import TYPE_FIELD
from safestructures.defaults import DEFAULT_PROCESS_MAP

MOCK_DEFAULT_PROCESS_MAP = {
    data_type: mock.MagicMock() for data_type in DEFAULT_PROCESS_MAP
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
        with (
            mock.patch.object(serializer, "process_map", self.mock_default_process_map),
            mock.patch("safestructures.serializer.type", mock_type_fn),
        ):
            mock_data = mock.Mock()
            serializer.serialize(mock_data)

            self.mock_default_process_map[data_type].assert_called_once_with(serializer)
            self.mock_processor_instances[data_type].assert_called_once_with(mock_data)

    @pytest.mark.parametrize("data_type", DEFAULT_PROCESS_MAP.keys())
    def test_deserialize(self, data_type):
        """Test `Serializer.deserialize`."""
        mock_get_data_type = mock.MagicMock(return_value=data_type)
        serializer = Serializer()
        with (
            mock.patch.object(serializer, "process_map", self.mock_default_process_map),
            mock.patch.object(serializer, "_get_data_type", mock_get_data_type),
        ):
            mock_type_str = "mock_type"
            mock_schema = {TYPE_FIELD: mock_type_str}
            serializer.deserialize(mock_schema)

            mock_get_data_type.assert_called_once_with(mock_type_str)
            self.mock_default_process_map[data_type].assert_called_once_with(serializer)
            self.mock_processor_instances[data_type].assert_called_once_with(
                mock_schema
            )
