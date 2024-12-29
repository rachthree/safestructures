"""Testing utilities."""

from copy import deepcopy
from dataclasses import fields, is_dataclass
from typing import Union

import numpy as np
import tensorflow as tf
import torch
from tensorflow.python.framework.ops import EagerTensor

from safestructures.constants import KEYS_FIELD, TYPE_FIELD, VALUE_FIELD
from safestructures.utils.dataclass import SafestructuresDataclass


def check_schema(
    schema: dict, expected_type: str, expected_value: Union[str, None, bool]
) -> None:
    """Check the schema.

    Args:
        schema (dict): The schema.
        expected_type (str): The expected string that denotes the type in the schema.
        expected_value (Union[str, None, bool]): The expected value in the schema.
    """
    error_msg = f"{schema} is not dict"
    assert isinstance(schema, dict), error_msg

    for f in [TYPE_FIELD, VALUE_FIELD]:
        error_msg = f"Missing key {f} in schema."
        assert f in schema, error_msg

    error_msg = (
        "Schema type check failed",
        f"Expected {expected_type}, got {schema[TYPE_FIELD]}",
    )
    assert schema[TYPE_FIELD] == expected_type, error_msg
    assert schema[VALUE_FIELD] == expected_value, "Schema value check failed."


def generate_schema(type_value: str, value: Union[str, None, bool]) -> dict:
    """Generate a mock schema.

    Args:
        type_value (str): The type as a string.
        value (Union[str, None, bool]): The value.

    Returns:
        dict: The mock schema.
    """
    return {TYPE_FIELD: type_value, VALUE_FIELD: value}


def compare_nested_schemas(schema1: dict, schema2: dict):
    """Recursively compare two schemas.

    If TYPE_FIELD is 'set', the order of elements in VALUE_FIELD (as strings)
        does not matter.
    Assumes all VALUE_FIELD fields are strings or lists of strings.

    Args:
        schema1 (dict): Schema to compare.
        schema2 (dict): Other schema to compare.
    """

    def compare_serialized_values(value1, value2, value_type: str):
        """Compare values based on their type."""
        if value_type == "set":
            # Compare sets (order does not matter)
            # TODO: Account for sets with tuple items.
            return set(map(tuple, value1)) == set(map(tuple, value2))
        elif value_type in {"list", "tuple"}:
            # Compare lists/tuples (order matters)
            return len(value1) == len(value2) and all(
                compare_nested_schemas(item1, item2)
                for item1, item2 in zip(value1, value2)
            )
        elif value_type in {"dict", SafestructuresDataclass.__name__}:
            return all(
                compare_nested_schemas(value1[key], value2[key]) for key in value1
            )
        else:
            # For primitive types, compare values as strings
            return str(value1) == str(value2)

    if set(schema1.keys()) != set(schema2.keys()):
        return False

    if TYPE_FIELD in schema1:
        if schema1[TYPE_FIELD] != schema2[TYPE_FIELD]:
            return False

        if schema1[TYPE_FIELD] == "dict":
            if not compare_nested_schemas(schema1[KEYS_FIELD], schema2[KEYS_FIELD]):
                return False

    # Compare the VALUE_FIELD fields
    if VALUE_FIELD in schema1:
        if not compare_serialized_values(
            schema1[VALUE_FIELD], schema2[VALUE_FIELD], schema1[TYPE_FIELD]
        ):
            return False
    else:
        for k in schema1:
            if not compare_nested_schemas(schema1[k], schema2[k]):
                return False

    return True


def _assert_tf_equal(tensor1, tensor2):
    assert tf.math.reduce_all(tf.equal(tensor1, tensor2))


NP2F_MAP = {
    np.ndarray: lambda x: x,
    torch.Tensor: torch.from_numpy,
    tf.Tensor: tf.convert_to_tensor,
    EagerTensor: tf.convert_to_tensor,
}

F2NP_MAP = {
    np.ndarray: lambda x: x,
    torch.Tensor: lambda x: x.numpy(),
    tf.Tensor: lambda x: x.numpy(),
    EagerTensor: lambda x: x.numpy(),
}

ASSERT_EQUAL_MAP = {
    np.ndarray: np.testing.assert_equal,
    torch.Tensor: torch.testing.assert_close,
    tf.Tensor: _assert_tf_equal,
    EagerTensor: _assert_tf_equal,
}

FRAMEWORK_TENSORS = tuple(NP2F_MAP.keys())


def compare_values(value1, value2):
    """Recursively compare two values."""

    def _helper(value1, value2):
        if is_dataclass(value1):
            if not is_dataclass(value2):
                return False

            fields1 = [f.name for f in fields(value1)]
            fields2 = [f.name for f in fields(value2)]
            if set(fields1) != set(fields2):
                return False

            for f in fields1:
                if not compare_values(getattr(value1, f), getattr(value2, f)):
                    return False

        else:
            if type(value1) is not type(value2):
                return False

            if isinstance(value1, (list, tuple)):
                for v1, v2 in zip(value1, value2):
                    if not compare_values(v1, v2):
                        return False

            elif isinstance(value1, dict):
                if not set(value1.keys()) == set(value2.keys()):
                    return False

                for k in value1:
                    if not compare_values(value1[k], value2[k]):
                        return False

            elif isinstance(value1, FRAMEWORK_TENSORS):
                tensor_type = type(value1)
                test_value2 = deepcopy(value2)
                if not isinstance(test_value2, tensor_type):
                    # convert to numpy, then to framework for generality
                    test_value2 = F2NP_MAP[type(test_value2)](test_value2)
                    test_value2 = NP2F_MAP[tensor_type](test_value2)

                ASSERT_EQUAL_MAP[tensor_type](value1, test_value2)

            else:
                # TODO: Handle set containing tuples
                if not value1 == value2:
                    return False

        return True

    return _helper(value1, value2)
