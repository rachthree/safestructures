# safestructures

## What is safestructures?
`safestructures` is a Python package that extends `safetensors` to serialize general data structures containing tensors and other data types. It uses `safetensors` to store tensors and uses its metadata to store the schema of the original data structure as well as other basic data types.

`safetensors` can be more preferred than `pickle` when saving tensors, especially for safety reasons. See [huggingface/safetensors](https://github.com/huggingface/safetensors) for more details.

`safetensors` only stores tensors. `safestructures` extends `safetensors` by storing information of the original data structure containing tensors as well as other data types. It utilizes a [plugin architecture](./plugins_guide.md) so that projects that contain custom types can serialize and deserialize their custom data.

## Why safestructures?
ML models in practice deal with more than just tensors. Tensors can be passed between layers or outputted using data containers. Other data types are used as well to modify layer behavior.

To have better reproducibility and tracking, we may need to store and load model inputs and outputs, as well as intermediate layer inputs and output in a safe manner. `safetensors` greatly helps on the tensor side, while `safestructures` helps with the rest.

## Support
Currently, `safestructures` supports out of the box:

* All of Python's built-in data containers and data types.
* `dataclasses.dataclass` objects.
* PyTorch tensors via `safetensors`.
* TensorFlow 2 tensors via `safetensors`.
* JAX tensors via `safetensors`.

For maximum compatibility with NumPy, **all float tensors are stored in FP32**.

`safestructures` aims to cover the same frameworks as `safetensors`, but for initial release it does not support:

* `collections` - Please comment on [this issue](https://github.com/rachthree/safestructures/issues/12) to request support.
* PaddlePaddle - Please comment on [this issue](https://github.com/rachthree/safestructures/issues/10) to request support.
* MLX - Please comment on [this issue](https://github.com/rachthree/safestructures/issues/9) to request support.

However, `safestructures` supports [plugins](./plugins_guide.md) in case there is a type it does not support.
