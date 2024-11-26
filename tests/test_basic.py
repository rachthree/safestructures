"""Tests for the basic datatype processors."""
from unittest import mock

import pytest

from safestructures.processors.basic import (
    BoolProcessor,
    ComplexProcessor,
    FloatProcessor,
    IntProcessor,
    NoneProcessor,
    StringProcessor,
)

numeric_test_cases = [
    (IntProcessor, 42),
    (IntProcessor, -42),
    (FloatProcessor, 1.61803398874989),
    (FloatProcessor, -1.61803398874989),
    (ComplexProcessor, complex(1, 2)),
]


@pytest.fixture
def mock_serializer():
    """Provide mocked Serializer fixture."""
    return mock.MagicMock()


def test_serialize_str(mock_serializer):
    """Test string serialization to schema."""
    test_input = "mock_input"
    expected_value = test_input
    result = StringProcessor(mock_serializer).serialize(test_input)

    assert result == expected_value
    mock_serializer.serialize.assert_not_called()
    mock_serializer.deserialize.assert_not_called()


def test_deserialize_str(mock_serializer):
    """Test deserialization to string."""
    test_value = "mock_input"
    result = StringProcessor(mock_serializer).deserialize(test_value)

    assert result == test_value
    mock_serializer.serialize.assert_not_called()
    mock_serializer.deserialize.assert_not_called()


@pytest.mark.parametrize("cls,test_input", numeric_test_cases)
def test_serialize_numeric(mock_serializer, cls, test_input):
    """Test numeric serialization to schema."""
    expected_value = str(test_input)
    result = cls(mock_serializer).serialize(test_input)

    assert result == expected_value
    mock_serializer.serialize.assert_not_called()
    mock_serializer.deserialize.assert_not_called()


@pytest.mark.parametrize("cls,test_input", numeric_test_cases)
def test_deserialize_numeric(mock_serializer, cls, test_input):
    """Test deserialization to numeric."""
    expected_value = test_input
    result = cls(mock_serializer).deserialize(test_input)

    assert result == expected_value
    mock_serializer.serialize.assert_not_called()
    mock_serializer.deserialize.assert_not_called()


@pytest.mark.parametrize("test_input", [True, False])
def test_serialize_bool(mock_serializer, test_input):
    """Test boolean serialization to schema."""
    expected_value = test_input
    result = BoolProcessor(mock_serializer).serialize(test_input)

    assert result == expected_value
    mock_serializer.serialize.assert_not_called()
    mock_serializer.deserialize.assert_not_called()


@pytest.mark.parametrize("test_input", [True, False])
def test_deserialize_bool(mock_serializer, test_input):
    """Test deserialization to boolean."""
    expected_value = test_input
    result = BoolProcessor(mock_serializer).deserialize(test_input)

    assert result == expected_value
    mock_serializer.serialize.assert_not_called()
    mock_serializer.deserialize.assert_not_called()


def test_serialize_null(mock_serializer):
    """Test null serialization to schema."""
    test_input = None
    expected_value = test_input
    result = NoneProcessor(mock_serializer).serialize(test_input)

    assert result == expected_value
    mock_serializer.serialize.assert_not_called()
    mock_serializer.deserialize.assert_not_called()


def test_deserialize_null(mock_serializer):
    """Test deserialization to None."""
    test_value = None
    result = NoneProcessor(mock_serializer).deserialize(test_value)

    assert result is None
    mock_serializer.serialize.assert_not_called()
    mock_serializer.deserialize.assert_not_called()
