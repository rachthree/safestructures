"""Test nested structures."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Union

import numpy as np
import pytest
import tensorflow as tf
import torch
from tensorflow.python.framework.ops import EagerTensor
from utils import compare_nested_schemas, compare_values

from safestructures import load_file, save_file
from safestructures.constants import KEYS_FIELD, Mode, TYPE_FIELD, VALUE_FIELD
from safestructures.serializer import Serializer

FRAMEWORKS = ["np", "pt", "tf"]


@dataclass
class _TestDC:
    name: str
    midichlorian_count: int
    chosen_one: bool
    force_tensor: Union[np.array, torch.Tensor, EagerTensor] = None


TEST_INPUT = [
    (42, "answer to everything", {"origin": "Deep Thought"}),
    {
        "name": "obi-wan",
        "midichlorian_count": 13000,
        "allies": ["anakin", "ahsoka"],
    },
    _TestDC(name="anakin", midichlorian_count=27000, chosen_one=True),
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
        "grandmaster": _TestDC(name="yoda", midichlorian_count=18000, chosen_one=False),
        "form": 4,
        66: "survived",
    },
]

EXPECTED_SCHEMA = {
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
                    KEYS_FIELD: {"origin": {TYPE_FIELD: "str", VALUE_FIELD: "origin"}},
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
            TYPE_FIELD: "SafestructuresDataclass",
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
                "force_tensor": {
                    TYPE_FIELD: "NoneType",
                    VALUE_FIELD: None,
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
                    TYPE_FIELD: "SafestructuresDataclass",
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
                        "force_tensor": {
                            TYPE_FIELD: "NoneType",
                            VALUE_FIELD: None,
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


def generate_test_with_tensors(
    tensor_fn: Callable, tensor_type_string: str, native_tensor_fn: Callable
):
    """Generate the test case."""
    test_input = deepcopy(TEST_INPUT)
    expected_schema = deepcopy(EXPECTED_SCHEMA)

    test_input[0] = (
        42,
        "answer to everything",
        tensor_fn(),
        {"origin": "Deep Thought"},
    )
    expected_schema[VALUE_FIELD][0][VALUE_FIELD].insert(
        2, {TYPE_FIELD: tensor_type_string, VALUE_FIELD: "0"}
    )

    test_input[1]["high_ground_tensor"] = tensor_fn()
    expected_schema[VALUE_FIELD][1][VALUE_FIELD]["high_ground_tensor"] = {
        TYPE_FIELD: tensor_type_string,
        VALUE_FIELD: "1",
    }
    expected_schema[VALUE_FIELD][1][KEYS_FIELD]["high_ground_tensor"] = {
        TYPE_FIELD: "str",
        VALUE_FIELD: "high_ground_tensor",
    }

    new_dc1 = _TestDC(
        name="anakin",
        midichlorian_count=27000,
        chosen_one=True,
        force_tensor=tensor_fn(),
    )
    test_input[2] = new_dc1
    expected_schema[VALUE_FIELD][2][VALUE_FIELD]["force_tensor"] = {
        TYPE_FIELD: tensor_type_string,
        VALUE_FIELD: "2",
    }

    test_input[4].insert(0, tensor_fn())
    expected_schema[VALUE_FIELD][4][VALUE_FIELD].insert(
        0, {TYPE_FIELD: tensor_type_string, VALUE_FIELD: "3"}
    )

    test_input[4].insert(2, tensor_fn())
    expected_schema[VALUE_FIELD][4][VALUE_FIELD].insert(
        2, {TYPE_FIELD: tensor_type_string, VALUE_FIELD: "4"}
    )

    new_dc2 = _TestDC(
        name="yoda",
        midichlorian_count=18000,
        chosen_one=False,
        force_tensor=tensor_fn(),
    )
    test_input[5]["grandmaster"] = new_dc2
    expected_schema[VALUE_FIELD][5][VALUE_FIELD]["grandmaster"][VALUE_FIELD][
        "force_tensor"
    ] = {TYPE_FIELD: tensor_type_string, VALUE_FIELD: "5"}

    return test_input, expected_schema, native_tensor_fn


def generate_numpy_tensor():
    """Generate a random numpy tensor."""
    return np.random.rand(4, 3, 128, 128)


def generate_torch_tensor():
    """Generate a random torch tensor."""
    return torch.randn(4, 3, 128, 128)


def generate_tf_tensor():
    """Generate a random tf tensor."""
    return tf.random.uniform(shape=(4, 3, 128, 128))


TEST_CASES = [
    (TEST_INPUT, EXPECTED_SCHEMA, lambda x: x),
    generate_test_with_tensors(generate_numpy_tensor, "numpy.ndarray", lambda x: x),
    generate_test_with_tensors(
        generate_torch_tensor, "torch.Tensor", lambda x: torch.from_numpy(x)
    ),
    generate_test_with_tensors(
        generate_tf_tensor,
        "tensorflow.python.framework.ops.EagerTensor",
        lambda x: tf.convert_to_tensor(x),
    ),
]


@pytest.mark.parametrize("test_input,expected_schema,native_tensor_fn", TEST_CASES)
def test_core_methods(test_input, expected_schema, native_tensor_fn):
    """Integration test for core serialize/deserialize methods."""
    # Serialization
    serializer = Serializer()
    serializer.mode = Mode.SAVE
    schema = serializer.serialize(test_input)
    assert compare_nested_schemas(expected_schema, schema)

    # Deserialization
    serializer.mode = Mode.LOAD

    # At load, all tensors could possibly be at a different framework
    # Mock this behavior by converting to the original framework here
    for k, tensor in serializer.tensors.items():
        serializer.tensors[k] = native_tensor_fn(tensor)

    value = serializer.deserialize(schema)
    assert compare_values(test_input, value)


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("test_input,expected_schema,native_tensor_fn", TEST_CASES)
def test_serializer_save_load(
    tmp_path, test_input, expected_schema, native_tensor_fn, framework
):
    """Integration test for `Serializer` save/load methods."""
    test_file = tmp_path / "Test.safetensors"
    test_other_metadata = {"test_field": "test_value"}
    Serializer().save(test_input, test_file, metadata=test_other_metadata)
    loaded = Serializer().load(test_file, framework=framework)
    compare_values(test_input, loaded)


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("test_input,expected_schema,native_tensor_fn", TEST_CASES)
def test_wrapper_save_load(
    tmp_path, test_input, expected_schema, native_tensor_fn, framework
):
    """Integration test for the wrapper `save_file` and `load_file` functions."""
    test_file = tmp_path / "Test.safetensors"
    test_other_metadata = {"test_field": "test_value"}
    save_file(test_input, test_file, metadata=test_other_metadata)
    loaded = load_file(test_file, framework=framework)
    compare_values(test_input, loaded)
