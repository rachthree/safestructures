"""Tests for the iterable datatype processors."""
from dataclasses import dataclass, fields, is_dataclass
from unittest import mock

import pytest

from safestructures.constants import TYPE_FIELD, VALUE_FIELD
from safestructures.processors.iterable import (
    DataclassProcessor,
    DictProcessor,
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
    assert isinstance(result[VALUE_FIELD], list)
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


def test_serialize_dict(mock_serializer):
    """Test dict serialization to schema."""
    test_input = {"name": "anakin", "midichlorian_count": 27000, "chosen_one": True}

    mock_calls = [mock.call(v) for v in test_input.values()]
    result = DictProcessor(mock_serializer).serialize(test_input)

    assert result[TYPE_FIELD] == "dict"
    assert isinstance(result[VALUE_FIELD], dict)
    assert len(result[VALUE_FIELD]) == len(test_input)
    for k in test_input:
        try:
            result[VALUE_FIELD][k]
        except KeyError:
            raise KeyError(f"Key {k} not found in result's value field section.")

    mock_serializer.serialize.assert_has_calls(mock_calls, any_order=True)
    mock_serializer.deserialize.assert_not_called()


def test_deserialize_dict(mock_serializer):
    """Test deserialization to dict."""
    test_schema = {
        TYPE_FIELD: "dict",
        VALUE_FIELD: {
            "name": {
                TYPE_FIELD: "str",
                VALUE_FIELD: "anakin",
            },
            "midichlorian_count": {
                TYPE_FIELD: "int",
                VALUE_FIELD: "27000",
            },
            "chosen_one": {
                TYPE_FIELD: "bool",
                VALUE_FIELD: True,
            },
        },
    }

    mock_calls = [mock.call(v) for v in test_schema[VALUE_FIELD].values()]
    result = DictProcessor(mock_serializer).deserialize(test_schema)

    assert isinstance(result, dict)
    assert len(result) == len(test_schema[VALUE_FIELD])
    for k in test_schema[VALUE_FIELD]:
        try:
            result[k]
        except KeyError:
            raise KeyError(f"Key {k} not found in deserialized result.")

    mock_serializer.serialize.assert_not_called()
    mock_serializer.deserialize.assert_has_calls(mock_calls, any_order=False)


def test_serialize_dataclass(mock_serializer):
    """Test dataclass serialization to schema."""

    @dataclass
    class TestDC:
        name: str
        midichlorian_count: int
        chosen_one: bool

    test_input = TestDC(name="anakin", midichlorian_count=27000, chosen_one=True)
    fields = [
        "name",
        "midichlorian_count",
        "chosen_one",
    ]  # being explicit here instead of using dataclasses.fields
    mock_calls = [mock.call(getattr(test_input, f)) for f in fields]
    result = DataclassProcessor(mock_serializer).serialize(test_input)

    assert result[TYPE_FIELD] == "Dataclass"
    assert isinstance(result[VALUE_FIELD], dict)
    assert len(result[VALUE_FIELD]) == len(fields)
    for f in fields:
        try:
            result[VALUE_FIELD][f]
        except KeyError:
            raise KeyError(
                f"Key {f} representing a dataclass field"
                " not found in result's value field section."
            )

    mock_serializer.serialize.assert_has_calls(mock_calls, any_order=True)
    mock_serializer.deserialize.assert_not_called()


def test_deserialize_dataclass(mock_serializer):
    """Test deserialization to a dataclass."""
    test_schema = {
        TYPE_FIELD: "Dataclass",
        VALUE_FIELD: {
            "name": {
                TYPE_FIELD: "str",
                VALUE_FIELD: "anakin",
            },
            "midichlorian_count": {
                TYPE_FIELD: "int",
                VALUE_FIELD: "27000",
            },
            "chosen_one": {
                TYPE_FIELD: "bool",
                VALUE_FIELD: True,
            },
        },
    }

    mock_calls = [mock.call(v) for v in test_schema[VALUE_FIELD].values()]
    result = DataclassProcessor(mock_serializer).deserialize(test_schema)

    assert is_dataclass(result)
    assert len(fields(result)) == len(test_schema[VALUE_FIELD])
    for f in test_schema[VALUE_FIELD]:
        try:
            getattr(result, f)
        except AttributeError:
            raise AttributeError(f"Field {f} not found in deserialized result.")

    mock_serializer.serialize.assert_not_called()
    mock_serializer.deserialize.assert_has_calls(mock_calls, any_order=False)
