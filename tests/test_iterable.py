"""Tests for the iterable datatype processors."""
from unittest import mock

import pytest

from safestructures.constants import TYPE_FIELD, VALUE_FIELD
from safestructures.processors.iterable import (  # DataclassProcessor,; DictProcessor,
    ListProcessor,
    SetProcessor,
    TupleProcessor,
)


@pytest.fixture
def mock_serializer():
    """Provide mocked Serializer fixture."""
    return mock.MagicMock()


def _serialize_listlike_test(mock_serializer, iterable_type, any_order=False):
    """Test serializing simple listlike objects."""
    test_input = iterable_type(
        ["somebody", 42, None, "once told me", complex(9001, 3.14), 1.618]
    )

    mock_calls = [mock.call(t) for t in test_input]
    cls_map = {
        list: ListProcessor,
        set: SetProcessor,
        tuple: TupleProcessor,
    }
    result = cls_map[iterable_type](mock_serializer).serialize(test_input)

    assert result[TYPE_FIELD] == iterable_type.__name__
    assert len(result[VALUE_FIELD]) == len(test_input)
    mock_serializer.serialize.assert_has_calls(mock_calls, any_order=any_order)
    mock_serializer.deserialize.assert_not_called()


def _deserialize_listlike_test(mock_serializer, iterable_type):
    """Test deserializing schemas for simple listlike objects."""
    test_schema = {
        TYPE_FIELD: iterable_type.__name__,
        VALUE_FIELD: [
            {
                TYPE_FIELD: "str",
                VALUE_FIELD: "somebody",
            },
            {
                TYPE_FIELD: "int",
                VALUE_FIELD: "42",
            },
            {
                TYPE_FIELD: "NoneType",
                VALUE_FIELD: None,
            },
            {
                TYPE_FIELD: "str",
                VALUE_FIELD: "once told me",
            },
            {
                TYPE_FIELD: "complex",
                VALUE_FIELD: "(9001+3.14j)",
            },
            {
                TYPE_FIELD: "float",
                VALUE_FIELD: "1.618",
            },
        ],
    }

    mock_calls = [mock.call(t) for t in test_schema[VALUE_FIELD]]
    result = ListProcessor(mock_serializer).deserialize(test_schema)

    assert isinstance(result, iterable_type)
    assert len(result) == len(test_schema[VALUE_FIELD])
    mock_serializer.serialize.assert_not_called()
    mock_serializer.deserialize.assert_has_calls(mock_calls, any_order=False)


@pytest.mark.parametrize(
    "iterable_type,any_order", [(list, False), (set, True), (tuple, False)]
)
def test_serialize_listlike(mock_serializer, iterable_type, any_order):
    """Test list serialization to schema."""
    _serialize_listlike_test(mock_serializer, iterable_type, any_order)


@pytest.mark.parametrize("iterable_type", [list, set, tuple])
def test_deserialize_listlike(mock_serializer, iterable_type):
    """Test deserialization to list."""
    _deserialize_listlike_test(mock_serializer, list)
