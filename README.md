# safestructures

## What is safestructures?
`safestructures` is a Python package based on `safetensors` to serialize general data structures. It uses `safetensors` to store tensors and uses its metadata to store the schema of the original data structure as well as other basic data types.

`safetensors` can be more preferred than `pickle` when saving tensors, especially for safety reasons. See [huggingface/safetensors](https://github.com/huggingface/safetensors) for more details.

`safetensors` only stores tensors. `safestructures` extends `safetensors` by storing information of the original data structure containing tensors as well as other data types. It utilities a [plugin architecture](./docs/plugins_guide.md) so that projects that contain custom types can serialize and deserialize their custom data.

## Why safestructures?
ML models in practice deal with more than just tensors. Tensors can be passed between layers or outputted using data containers. Other data types are used as well to modify layer behavior.

To have better reproducibility and tracking, we may need to store and load model inputs and outputs, as well as intermediate layer inputs and output in a safe manner. `safetensors` greatly helps on the tensor side, while `safestructures` helps with the rest.


## Installation and requirements
`safestructures` requires Python 3.10 and above.

To install, simply run
```
pip install safestructures
```
 This will also install the minimum dependencies, namely `safetensors` and `numpy`.

As for ML / tensor frameworks, `safestructures` expects `numpy` at minimum. To use `safestructures`'s PyTorch, TensorFlow, and JAX capabilities, the respective framework must be installed separately.


## Usage and plugins
See the [usage](./docs/usage.md) and [plugins guide](./docs/plugins_guide.md) readmes, or refer to [the docs](https://rachthree.github.io/safestructures)


## Support
Currently, `safestructures` supports all of Python's core data containers and data types, including `dataclass`. Additionally, it supports saving PyTorch, TensorFlow 2, and JAX tensors via `safetensors`. It aims to cover the same frameworks as `safetensors`, but out of the box for initial release it does not support:

* `collections` - Please comment on [this issue](https://github.com/rachthree/safestructures/issues/12) to request support.
* PaddlePaddle - Please comment on [this issue](https://github.com/rachthree/safestructures/issues/10) to request support.
* MLX - Please comment on [this issue](https://github.com/rachthree/safestructures/issues/9) to request support.
