"""Tests for the basic datatype processor."""
from unittest import mock

import pytest
from utils import check_schema, generate_schema

from safestructures.processors.basic import BasicProcessor

numeric_test_cases = [
    (42, "int"),
    (-42, "int"),
    (1.61803398874989, "float"),
    (-1.61803398874989, "float"),
    (complex(1, 2), "complex"),
]


@pytest.fixture
def mock_serializer():
    """Provide mocked Serializer fixture."""
    return mock.MagicMock()


def test_serialize_str(mock_serializer):
    """Test string serialization to schema."""
    test_input = "mock_input"
    expected_value = test_input
    result = BasicProcessor(mock_serializer).serialize(test_input)

    check_schema(result, "str", expected_value)
    mock_serializer.serialize.assert_not_called()
    mock_serializer.deserialize.assert_not_called()


def test_deserialize_str(mock_serializer):
    """Test deserialization to string."""
    test_value = "mock_input"
    test_schema = generate_schema("str", test_value)
    result = BasicProcessor(mock_serializer).deserialize(test_schema)

    assert result == test_value
    mock_serializer.serialize.assert_not_called()
    mock_serializer.deserialize.assert_not_called()


@pytest.mark.parametrize("test_input,expected_type", numeric_test_cases)
def test_serialize_numeric(mock_serializer, test_input, expected_type):
    """Test numeric serialization to schema."""
    expected_value = str(test_input)
    result = BasicProcessor(mock_serializer).serialize(test_input)

    check_schema(result, expected_type, expected_value)
    mock_serializer.serialize.assert_not_called()
    mock_serializer.deserialize.assert_not_called()


@pytest.mark.parametrize("expected_value,schema_type", numeric_test_cases)
def test_deserialize_numeric(mock_serializer, expected_value, schema_type):
    """Test deserialization to numeric."""
    schema_input = str(expected_value)
    test_schema = generate_schema(schema_type, schema_input)
    result = BasicProcessor(mock_serializer).deserialize(test_schema)

    assert result == expected_value
    mock_serializer.serialize.assert_not_called()
    mock_serializer.deserialize.assert_not_called()


@pytest.mark.parametrize("test_input", [True, False])
def test_serialize_bool(mock_serializer, test_input):
    """Test boolean serialization to schema."""
    expected_value = test_input
    result = BasicProcessor(mock_serializer).serialize(test_input)

    check_schema(result, "bool", expected_value)
    mock_serializer.serialize.assert_not_called()
    mock_serializer.deserialize.assert_not_called()


@pytest.mark.parametrize("test_input", [True, False])
def test_deserialize_bool(mock_serializer, test_input):
    """Test deserialization to boolean."""
    expected_value = test_input
    test_schema = generate_schema("bool", test_input)
    result = BasicProcessor(mock_serializer).deserialize(test_schema)

    assert result == expected_value
    mock_serializer.serialize.assert_not_called()
    mock_serializer.deserialize.assert_not_called()


def test_serialize_null(mock_serializer):
    """Test null serialization to schema."""
    test_input = None
    expected_value = test_input
    result = BasicProcessor(mock_serializer).serialize(test_input)

    check_schema(result, "NoneType", expected_value)
    mock_serializer.serialize.assert_not_called()
    mock_serializer.deserialize.assert_not_called()


def test_deserialize_null(mock_serializer):
    """Test deserialization to None."""
    test_value = None
    test_schema = generate_schema("NoneType", test_value)
    result = BasicProcessor(mock_serializer).deserialize(test_schema)

    assert result is None
    mock_serializer.serialize.assert_not_called()
    mock_serializer.deserialize.assert_not_called()
