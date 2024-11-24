"""Tests for the iterable datatype processors."""
import ast
from dataclasses import dataclass, fields, is_dataclass
from unittest import mock

import pytest
from utils import compare_nested_schemas, compare_values

from safestructures import Serializer
from safestructures.constants import KEYS_FIELD, Mode, TYPE_FIELD, VALUE_FIELD
from safestructures.processors.iterable import (
    DataclassProcessor,
    DictProcessor,
    ListProcessor,
    SetProcessor,
    TupleProcessor,
)


@pytest.fixture
def mock_serializer():
    """Provide mocked Serializer fixture in save mode."""
    mock_serializer = mock.MagicMock()
    serializer = Serializer()
    serializer.mode = Mode.SAVE
    mock_serializer.serialize = mock.MagicMock(wraps=serializer.serialize)
    return mock_serializer


@pytest.fixture
def mock_deserializer():
    """Provide mocked Serializer fixture in load mode."""
    mock_serializer = mock.MagicMock()
    serializer = Serializer()
    serializer.mode = Mode.LOAD
    mock_serializer.deserialize = mock.MagicMock(wraps=serializer.deserialize)
    return mock_serializer


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

    assert isinstance(result, list)
    assert len(result) == len(test_input)
    mock_serializer.serialize.assert_has_calls(mock_calls, any_order=any_order)
    mock_serializer.deserialize.assert_not_called()


def _deserialize_listlike_test(mock_serializer, iterable_type):
    """Test deserializing schemas for simple listlike objects."""
    test_serialized = [
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
    ]
    cls_map = {
        list: ListProcessor,
        set: SetProcessor,
        tuple: TupleProcessor,
    }

    mock_calls = [mock.call(t) for t in test_serialized]
    result = cls_map[iterable_type](mock_serializer).deserialize(test_serialized)

    assert isinstance(result, iterable_type)
    assert len(result) == len(test_serialized)
    mock_serializer.serialize.assert_not_called()
    mock_serializer.deserialize.assert_has_calls(mock_calls, any_order=False)


@pytest.mark.parametrize(
    "iterable_type,any_order", [(list, False), (set, True), (tuple, False)]
)
def test_serialize_listlike(mock_serializer, iterable_type, any_order):
    """Test list serialization to schema."""
    _serialize_listlike_test(mock_serializer, iterable_type, any_order)


@pytest.mark.parametrize("iterable_type", [list, set, tuple])
def test_deserialize_listlike(mock_deserializer, iterable_type):
    """Test deserialization to list."""
    _deserialize_listlike_test(mock_deserializer, list)


def test_serialize_dict(mock_serializer):
    """Test dict serialization to schema."""
    test_input = {
        "name": "anakin",
        "midichlorian_count": 27000,
        "chosen_one": True,
        (1, 2): 3,
    }

    mock_calls = [mock.call(v) for v in test_input.values()]
    result = DictProcessor(mock_serializer).serialize(test_input)
    extra_results = DictProcessor(mock_serializer).serialize_extra(test_input)

    assert isinstance(result, dict)
    assert len(result) == len(test_input)
    assert len(extra_results[KEYS_FIELD]) == len(test_input)
    for k in test_input:
        try:
            result[str(k)]
        except KeyError:
            raise KeyError(f"Key {k} not found in result's value field section.")

    mock_serializer.serialize.assert_has_calls(mock_calls, any_order=True)
    mock_serializer.deserialize.assert_not_called()


def test_deserialize_dict(mock_deserializer):
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
            "(1, 2)": {
                TYPE_FIELD: "int",
                VALUE_FIELD: "3",
            },
        },
        KEYS_FIELD: {
            "name": {
                TYPE_FIELD: "str",
                VALUE_FIELD: "name",
            },
            "midichlorian_count": {
                TYPE_FIELD: "str",
                VALUE_FIELD: "midichlorian_count",
            },
            "chosen_one": {
                TYPE_FIELD: "str",
                VALUE_FIELD: "chosen_one",
            },
            "(1, 2)": {
                TYPE_FIELD: "tuple",
                VALUE_FIELD: [
                    {TYPE_FIELD: "int", VALUE_FIELD: "1"},
                    {TYPE_FIELD: "int", VALUE_FIELD: "2"},
                ],
            },
        },
    }
    serialized = {
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
        "(1, 2)": {
            TYPE_FIELD: "int",
            VALUE_FIELD: "3",
        },
    }
    kwargs = {
        TYPE_FIELD: "dict",
        KEYS_FIELD: {
            "name": {
                TYPE_FIELD: "str",
                VALUE_FIELD: "name",
            },
            "midichlorian_count": {
                TYPE_FIELD: "str",
                VALUE_FIELD: "midichlorian_count",
            },
            "chosen_one": {
                TYPE_FIELD: "str",
                VALUE_FIELD: "chosen_one",
            },
            "(1, 2)": {
                TYPE_FIELD: "tuple",
                VALUE_FIELD: [
                    {TYPE_FIELD: "int", VALUE_FIELD: "1"},
                    {TYPE_FIELD: "int", VALUE_FIELD: "2"},
                ],
            },
        },
    }

    mock_calls = [mock.call(v) for v in serialized.values()]
    for v in test_schema[KEYS_FIELD].values():
        mock_calls.append(mock.call(v))
    result = DictProcessor(mock_deserializer).deserialize(serialized, **kwargs)

    assert isinstance(result, dict)
    assert len(result) == len(serialized)
    for k in serialized:
        try:
            if kwargs[KEYS_FIELD][k][TYPE_FIELD] == "tuple":
                k = ast.literal_eval(k)
            result[k]
        except KeyError:
            raise KeyError(f"Key {k} not found in deserialized result.")

    mock_deserializer.serialize.assert_not_called()
    mock_deserializer.deserialize.assert_has_calls(mock_calls, any_order=True)


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

    assert isinstance(result, dict)
    assert len(result) == len(fields)
    for f in fields:
        try:
            result[f]
        except KeyError:
            raise KeyError(
                f"Key {f} representing a dataclass field"
                " not found in result's value field section."
            )

    mock_serializer.serialize.assert_has_calls(mock_calls, any_order=True)
    mock_serializer.deserialize.assert_not_called()


def test_deserialize_dataclass(mock_deserializer):
    """Test deserialization to a dataclass."""
    test_serialized = {
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
    }

    mock_calls = [mock.call(v) for v in test_serialized.values()]
    result = DataclassProcessor(mock_deserializer).deserialize(test_serialized)

    assert is_dataclass(result)
    assert len(fields(result)) == len(test_serialized)
    for f in test_serialized:
        try:
            getattr(result, f)
        except AttributeError:
            raise AttributeError(f"Field {f} not found in deserialized result.")

    mock_deserializer.serialize.assert_not_called()
    mock_deserializer.deserialize.assert_has_calls(mock_calls, any_order=False)


def test_nested_structure():
    """Integration test for a nested structure without tensors."""

    @dataclass
    class TestDC:
        name: str
        midichlorian_count: int
        chosen_one: bool

    test_input = [
        (42, "answer to everything", {"origin": "Deep Thought"}),
        {
            "name": "obi-wan",
            "midichlorian_count": 13000,
            "allies": ["anakin", "ahsoka"],
        },
        TestDC(name="anakin", midichlorian_count=27000, chosen_one=True),
        {8, "red", 13, "black"},
        [
            complex(1.618, 3.14),
            -123.56789101112,
            "abc123",
            (
                "shii-CHO",
                "makashi",
                "SORESU",
                "ataru",
                ("Shien", "Djem So"),
                "Niman",
                ("Juyo", "Vaapad"),
            ),
        ],
        {
            "grandmaster": TestDC(
                name="yoda", midichlorian_count=18000, chosen_one=False
            ),
            "form": 4,
            66: "survived",
        },
    ]

    expected_schema = {
        TYPE_FIELD: "list",
        VALUE_FIELD: [
            {
                TYPE_FIELD: "tuple",
                VALUE_FIELD: [
                    {TYPE_FIELD: "int", VALUE_FIELD: "42"},
                    {TYPE_FIELD: "str", VALUE_FIELD: "answer to everything"},
                    {
                        TYPE_FIELD: "dict",
                        VALUE_FIELD: {
                            "origin": {TYPE_FIELD: "str", VALUE_FIELD: "Deep Thought"}
                        },
                        KEYS_FIELD: {
                            "origin": {TYPE_FIELD: "str", VALUE_FIELD: "origin"}
                        },
                    },
                ],
            },
            {
                TYPE_FIELD: "dict",
                VALUE_FIELD: {
                    "name": {
                        TYPE_FIELD: "str",
                        VALUE_FIELD: "obi-wan",
                    },
                    "midichlorian_count": {
                        TYPE_FIELD: "int",
                        VALUE_FIELD: "13000",
                    },
                    "allies": {
                        TYPE_FIELD: "list",
                        VALUE_FIELD: [
                            {TYPE_FIELD: "str", VALUE_FIELD: "anakin"},
                            {TYPE_FIELD: "str", VALUE_FIELD: "ahsoka"},
                        ],
                    },
                },
                KEYS_FIELD: {
                    "name": {TYPE_FIELD: "str", VALUE_FIELD: "name"},
                    "midichlorian_count": {
                        TYPE_FIELD: "str",
                        VALUE_FIELD: "midichlorian_count",
                    },
                    "allies": {TYPE_FIELD: "str", VALUE_FIELD: "allies"},
                },
            },
            {
                TYPE_FIELD: "dataclass",
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
            },
            {
                TYPE_FIELD: "set",
                VALUE_FIELD: [
                    {TYPE_FIELD: "int", VALUE_FIELD: "8"},
                    {TYPE_FIELD: "str", VALUE_FIELD: "black"},
                    {TYPE_FIELD: "int", VALUE_FIELD: "13"},
                    {TYPE_FIELD: "str", VALUE_FIELD: "red"},
                ],
            },
            {
                TYPE_FIELD: "list",
                VALUE_FIELD: [
                    {TYPE_FIELD: "complex", VALUE_FIELD: "(1.618+3.14j)"},
                    {TYPE_FIELD: "float", VALUE_FIELD: "-123.56789101112"},
                    {TYPE_FIELD: "str", VALUE_FIELD: "abc123"},
                    {
                        TYPE_FIELD: "tuple",
                        VALUE_FIELD: [
                            {TYPE_FIELD: "str", VALUE_FIELD: "shii-CHO"},
                            {TYPE_FIELD: "str", VALUE_FIELD: "makashi"},
                            {TYPE_FIELD: "str", VALUE_FIELD: "SORESU"},
                            {TYPE_FIELD: "str", VALUE_FIELD: "ataru"},
                            {
                                TYPE_FIELD: "tuple",
                                VALUE_FIELD: [
                                    {TYPE_FIELD: "str", VALUE_FIELD: "Shien"},
                                    {TYPE_FIELD: "str", VALUE_FIELD: "Djem So"},
                                ],
                            },
                            {TYPE_FIELD: "str", VALUE_FIELD: "Niman"},
                            {
                                TYPE_FIELD: "tuple",
                                VALUE_FIELD: [
                                    {TYPE_FIELD: "str", VALUE_FIELD: "Juyo"},
                                    {TYPE_FIELD: "str", VALUE_FIELD: "Vaapad"},
                                ],
                            },
                        ],
                    },
                ],
            },
            {
                TYPE_FIELD: "dict",
                VALUE_FIELD: {
                    "grandmaster": {
                        TYPE_FIELD: "dataclass",
                        VALUE_FIELD: {
                            "name": {TYPE_FIELD: "str", VALUE_FIELD: "yoda"},
                            "midichlorian_count": {
                                TYPE_FIELD: "int",
                                VALUE_FIELD: "18000",
                            },
                            "chosen_one": {
                                TYPE_FIELD: "bool",
                                VALUE_FIELD: False,
                            },
                        },
                    },
                    "form": {TYPE_FIELD: "int", VALUE_FIELD: "4"},
                    "66": {TYPE_FIELD: "str", VALUE_FIELD: "survived"},
                },
                KEYS_FIELD: {
                    "grandmaster": {TYPE_FIELD: "str", VALUE_FIELD: "grandmaster"},
                    "form": {TYPE_FIELD: "str", VALUE_FIELD: "form"},
                    "66": {TYPE_FIELD: "int", VALUE_FIELD: "66"},
                },
            },
        ],
    }

    # Serialization
    serializer = Serializer()
    serializer.mode = Mode.SAVE
    schema = serializer.serialize(test_input)
    assert compare_nested_schemas(expected_schema, schema)

    # Deserialization
    serializer.mode = Mode.LOAD
    value = serializer.deserialize(schema)
    assert compare_values(test_input, value)
