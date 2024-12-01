"""Test the wrapper functions."""
from unittest import mock

from safestructures import DataProcessor, load_file, save_file


class MockPlugin(DataProcessor):
    """Mock plugin."""

    data_type = mock.Mock

    def serialize(self, data):
        """Overload `DataProcessor.serialize`."""
        pass

    def deserialize(self, serialized):
        """Overload `DataProcessor.deserialize`."""
        pass


def test_save_file():
    """Test the `save_file` wrapper function."""
    mock_input = mock.Mock()
    mock_save_path = "path/to/save.safetensors"
    mock_metadata = {"mock_field": "mock_value"}
    mock_serializer_instance = mock.MagicMock()
    mock_serializer = mock.MagicMock(return_value=mock_serializer_instance)

    with mock.patch("safestructures.wrapper.Serializer", mock_serializer):
        save_file(
            mock_input, mock_save_path, metadata=mock_metadata, plugins=MockPlugin
        )
        mock_serializer.assert_called_once_with(plugins=[MockPlugin])
        mock_serializer_instance.save.assert_called_once_with(
            mock_input, mock_save_path, metadata=mock_metadata
        )


def test_load_file():
    """Test the `load_file` wrapper function."""
    mock_load_path = "path/to/save.safetensors"
    mock_framework = "mock_framework"
    mock_device = "mock_device"
    mock_serializer_instance = mock.MagicMock()
    mock_serializer = mock.MagicMock(return_value=mock_serializer_instance)

    with mock.patch("safestructures.wrapper.Serializer", mock_serializer):
        load_file(
            mock_load_path,
            framework=mock_framework,
            device=mock_device,
            plugins=MockPlugin,
        )
        mock_serializer.assert_called_once_with(plugins=[MockPlugin])
        mock_serializer_instance.load.assert_called_once_with(
            mock_load_path, framework=mock_framework, device=mock_device
        )
