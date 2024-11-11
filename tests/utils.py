"""Testing utilities."""
from typing import Union

from safestructures.constants import TYPE_FIELD, VALUE_FIELD


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
