# Plugins

!!! warning "WARNING"
    The usage of plugins **can execute arbitrary code**. Be careful using plugins from a 3rd party.

`safestructures` utilizes a plugin-based architecture to process different data types.
These are based on two main classes: `safestructures.DataProcessor` and `safestructures.TensorProcessor`.

`DataProcessor` serves as an abstract base class that you can extend to handle serialization and deserialization of specific data types. By subclassing `DataProcessor`, you can define how your custom data type is converted into a format that `safetensors` can store, and how to load it from `safetensors` metadata. This is subclassed directly for [basic data types](#basic-types) and [data containers](#containers).

`TensorProcessor` is a special subclass of `DataProcessor`, utilizing `safetensors`'s capabilities to serialize and deserialize [tensors](#tensors).

## Basic types
Basic types, such as `str`, `int`, `float`, etc. use subclasses of `DataProcessor`.
These are considered as "atomic" data types, the base case where no further serialization is needed.
If you have a custom atomic data type, especially one that is not covered by core `safestructures` capabilities,
then you would subclass `DataProcessor` (called `MyTypeProcessor` below for example) and follow 3 main steps:

1. Define `MyTypeProcessor.data_type`, a class attribute.
2. Implement `MyTypeProcessor.serialize`, the serialization method.
    * Input: A value of your custom type.
    * Returns: A string. Strings are required to be compatible with `safetensors` metadata.
3. Implement `MyTypeProcessor.deserialize`, the deserialization method.
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

class MyTypeProcessor(DataProcessor):
    data_type = MyCustomType  # Set this to your custom data type

    def serialize(self, data: MyCustomType) -> str:
        # Convert MyCustomType to a format Safetensors can store
        return str(data.value)

    def deserialize(self, serialized: str) -> MyCustomType:
        # Reconstruct MyCustomType from the serialized form
        return MyCustomType(int(serialized))
```

You can then serialize your object or a data container containing your object by using the `plugins` keyword argument with `save_file` and `load_file`:

```python
from safestructures import save_file, load_file

list_obj = [MyCustomType(42), MyCustomType(88)]
file_path = "my_custom_objs.safestructures"

save_file(list_obj, file_path, plugins=[MyTypeProcessor])

loaded_obj = load_file(file_path, plugins=[MyTypeProcessor])
```

However, if the custom object you want to serialize is a container or even a nested container that would house atomic data types,
then see the [Containers section](#containers).

### Optional: storing other metadata to aid in deserialization
It may be helpful to store other data that cannot be captured in a single string representation by `DataProcessor.serialize`, but would be needed to properly deserialize your custom data type.
`DataProcessor.serialize_extra` helps with this by giving the option to provide extra metadata.

It accepts the value you want to serialize, and your implementation would need to return a dictionary of only string types
to be used as keyword arguments for `MyTypeProcessor.deserialize`.
Note that `MyTypeProcessor.deserialize` would need to accept these keyword arguments.

An example would be the core `DictProcessor`:
::: safestructures.processors.iterable.DictProcessor
    handler: python
    options:
        show_bases: false
        show_source: true
        show_docstring_description: false
        members: false

!!! info "Note"
    `DictProcessor` is a container plugin. See the [Containers section](#containers) for details on container plugins.

## Tensors
`TensorProcessor` is a special `DataProcessor`.
The `serialize` and `deserialize` methods do not need to be overloaded for a subclass / plugin,
but there are still 2 main steps:

1. Define `MyTensorProcessor.data_type`, a class attribute.
    * This would be the tensor class of the ML framework.
2. Implement `MyTensorProcessor.to_numpy`, a processing method to convert to NumPy.
    * Input: The ML framework tensor.
    * Returns: The tensor as a `numpy.ndarray`. The implementation should:
        * Provide a contiguous array.
        * Be casted to FP32 for float tensors for maximum compatibility.


An example would be the core `TorchProcessor`:
::: safestructures.processors.tensor.TorchProcessor
    handler: python
    options:
        show_bases: false
        show_source: true
        show_docstring_description: false
        members: false

## Containers
Processors for containers such as lists and dictionaries are still `DataProcessor` subclasses but are recursive in nature.
`safestructures` uses recursion to traverse a data structure.
Unless there are values that are needed to be serialized at the container level, such as dictionary keys,
no actual values are serialized/deserialized once the container object is reached.
A `DataProcessor` for a container merely iterates through the object and uses `self.serializer.serialize` or `self.serializer.deserialize` to further serialize or deserialize child values, respectively.

Implementing a `DataProcessor` subclass to handle your custom container class follows the same steps as
for [basic types](#basic-types), but with the following extra considerations:

1. The `DataProcessor.serialize` method must use `self.serializer.serialize` to further serialize as you iterate through the values in the container.
    * Input: The data container to serialize.
    * Output: Must be a `builtin` container. Consider what `builtin` container (such as `dict` or `list`) best fits your custom container class.
    For example, `safestructures` uses `dict` to help serialize `dataclasses.dataclass` objects.
2. The `DataProcessor.deserialize` method must use `self.serializer.deserialize` to further deserialize as you iterate through the values in a `builtin` serialized container.
    * Input: A `builtin` container with serialized values.
    For example, an object that used a dictionary to serialize would have a `dict` provided to `DataProcessor.deserialize`, and a list/tuple/set-like object would have a `list` provided to `DataProcessor.deserialize`.
    * Output: The deserialized custom object.

The `DictProcessor` implementation [above](#optional-storing-other-metadata-to-aid-in-deserialization) is a good example of these concepts, while also exercising `DataProcessor.serialize_extra` to handle other values that require serialization at the container level.

In `safestructures`, the core container processors are iterable in nature, so all are in the `safestructures.processor.iterable` submodule:

* `ListProcessor`
* `SetProcessor`
* `TupleProcessor`
* `DictProcessor`
* `DataclassProcessor`

For example, let's create a plugin for `transformers.modeling_outputs.ModelOutput` objects.
Since `ModelOutput` objects are similar to `dataclasses.dataclass` objects, we'll subclass `safestructures.processors.iterable.DataclassProcessor`

```python
from safestructures.processors.iterable import DataclassProcessor

class ModelOutputProcessor(DataclassProcessor):
    """Processor for `transformers`'s ModelOutput."""

    data_type = ModelOutput

    def deserialize(self, serialized: dict, **kwargs) -> ModelOutput:
        """Overload DataclassProcessor.deserialize.

        This is so the proper ModelOutput is provided.
        """
        mo_kwargs = {}
        for k, v in serialized.items():
            mo_kwargs[k] = self.serializer.deserialize(v)

        model_output_instance = self.data_type(**mo_kwargs)

        return model_output_instance
```

Since most `ModelOutput` objects from a `transformers` model are subclasses, we can just subclass `ModelOutputProcessor` like below:
```python
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions BaseModelOutputWithPoolingAndCrossAttentions

class BertOutputProcessor(ModelOutputProcessor):
    """Processor for BERT model outputs."""

    data_type = BaseModelOutputWithPoolingAndCrossAttentions


class BertEncoderOutputProcessor(ModelOutputProcessor):
    """Processor for BERT encoder outputs."""

    data_type = BaseModelOutputWithPastAndCrossAttentions
```

We can then serialize outputs of the model. Below is a PyTorch-based example:
```python
from transformers import BertConfig, BertModel

config = BertConfig()
model = BertModel(config)
test_plugins = [BertOutputProcessor, BertEncoderOutputProcessor]

results = {}

def _store_encoder_output(module, args, kwargs, output):
    results["encoder_output"] = output
    return output

model.encoder.register_forward_hook(_store_encoder_output, with_kwargs=True)

test_input_ids = torch.tensor([[0] * 128])
test_output = model(test_input_ids)

results["model_output"] = test_output

test_filepath = tmp_path / "test.safestructures"
save_file(results, test_filepath, plugins=test_plugins)

deserialized = load_file(test_filepath, plugins=test_plugins)
```
