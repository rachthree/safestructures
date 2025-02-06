# Usage
To use the core functionalities of `safestructures`, only two functions are needed: `save_file` and `load_file`.

## Saving / serializing
To save a Python object, simply use:

```python
from safestructures import save_file

data: Object = None # any value or data container
save_path = "path/to/save/obj.safestructures"

save_file(data, save_path)
```

!!! note "Note about Dataclasses"
    `safestructures` treats `dataclass` objects as a special case. The original class will be used when deserializing, but by default a dataclass specific to `safestructures` will be used restore the object
    unless a plugin is used.

### Saving additional metadata
Metadata can be saved using the `metadata` keyword argument with `save_file`. It accepts a flattened dictionary of key type `str`, value type `str`, as that is what `safetensors` accepts.

To load metadata, load as you would with a normal `safetensors` file, namely:

```python
from safetensors import safe_open

file_path = "path/to/load/obj.safestructures"

with safe_open(load_path) as f:
    metadata = f.metadata()
```

Go [here](./examples.md) to see more examples.

## Loading / deserializing
To load a Python object, simply use:

```python
from safestructures import save_file

file_path = "path/to/load/obj.safestructures"

obj = load_file(file_path)
```

`obj` will be the object saved from `save_file`.

Go [here](./examples.md) to see more examples.

## What if `safestructures` does not handle a certain type?
`safestructures` is extensible through plugins. See [plugins](./plugins_guide.md) for more details.
