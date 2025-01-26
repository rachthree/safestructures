# Plugins

!!! note "WARNING"
    The usage of plugins **can execute arbitrary code**. Be careful using plugins from a 3rd party.

`safestructures` utilizes a plugin-based architecture to process different data types.
These are based on two main classes: `safestructures.DataProcessor` and `safestructures.TensorProcessor`.

`DataProcessor` serves as an abstract base class that you can extend to handle serialization and deserialization of specific data types. By subclassing `DataProcessor`, you can define how your custom data type is converted into a format that `safetensors` can store, and how to load it from `safetensors` metadata. This is subclassed directly for [basic data types](#basic-types) and [data containers](#containers).

`TensorProcessor` is a special subclass of `DataProcessor`, utilizing `safetensors`'s capabilities to serialize and deserialize [tensors](#tensors).

## Basic types
Basic types, such as `str`, `int`, `float`, etc. uses subclasses of `DataProcessor`.
If you have a custom atomic data type, especially one that is not covered by core `safestructures` capabilities,
then you would subclass `DataProcessor` (called `MyCustomTypeProcessor` below for example) and follow 3 main steps:

1. Define `MyCustomTypeProcessor.data_type`, a class attribute.
2. Implement `MyCustomTypeProcessor.serialize`, the serialization method.
    * Input: A value of your custom type.
    * Returns: A string. Strings are required to be compatible with `safetensors` metadata.
3. Implement `MyCustomTypeProcessor.deserialize`, the deserialization method.
    * Input: The string representation of the value.
    * Returns: the original value as your custom type.

For example, for the custom class:
```python
class MyCustomType:
    def __init__(self, value: int):
        self.value = value
```

The data processor would be:
```python
from safestructures import DataProcessor

class MyCustomTypeProcessor(DataProcessor):
    data_type = MyCustomType  # Set this to your custom data type

    def serialize(self, data: MyCustomType) -> str:
        # Convert MyCustomType to a format Safetensors can store
        return str(data.value)

    def deserialize(self, serialized: str) -> MyCustomType:
        # Reconstruct MyCustomType from the serialized form
        return MyCustomType(int(serialized))
```

You can then serialize your object or a data container with your object by using the `plugins` keyword argument with `save_file` and `load_file`:

```python
from safestructures import save_file, load_file

list_obj = [MyCustomType(42), MyCustomType(88)]
file_path = "my_custom_objs.safestructures"

save_file(list_obj, file_path, plugins=[MyCustomTypeProcessor])

loaded_obj = load_file(file_path, plugins=[MyCustomTypeProcessor])
```

However, if the custom object you want to serialize is a container or even a nested container that would house atomic data types,
then see the [Containers section](#containers).

### Optional: storing other metadata to aid in deserialization
It may be helpful to store other data that cannot be captured in a single string representation by `DataProcessor.serialize`, but would be needed to properly deserialize your custom data type.
`DataProcessor.serialize_extra` helps with this by giving the option to provide extra metadata.

It accepts the value you want to serialize, and your implementation would need to return a dictionary of only string types
to be used as keyword arguments for `MyCustomTypeProcessor.deserialize`.
Note that `MyCustomTypeProcessor.deserialize` would need to accept these keyword arguments.

## Tensors

## Containers
