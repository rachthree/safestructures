# Examples

The examples here use forward hooks to record layer inputs and outputs.
They subclass `IOHook` below to save inputs and outputs.

```python
from pathlib import PosixPath, Path

from safestructures import save_file

class IOHook:
    framework: str

    def __init__(self, layer_name: str, save_dir: PosixPath):
        self.layer_name = layer_name
        self.save_dir = save_dir

    def __call__(self, module, inputs, outputs):
        input_filename = f"{self.layer_name}_inputs_{self.times_called}.safestructures"
        output_filename = f"{self.layer_name}_outputs_{self.times_called}.safestructures"
        input_save_file = self.save_dir / input_filename
        output_save_file = self.save_dir / output_filename
        save_file(inputs, input_save_file)
        save_file(outputs, output_save_file)
```

## PyTorch intermediate input and outputs
```python
import torch
from torchvision.models.resnet import resnet50, ResNet50_Weights


class TorchIOHook(IOHook):
    framework = "pt"

save_dir = Path(".").expanduser().resolve()

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.eval()

for n, m in model.named_modules():
    layer_name = n if n else "model"

    io_hook = TorchIOHook(layer_name, save_dir)
    m.register_forward_hook(io_hook)

test_input = torch.randn(8, 3, 224, 224)

with torch.no_grad():
    model(test_input)
```


## TensorFlow intermediate input and outputs
!!! note "Note"
    This uses [tensorflow-hooks](https://github.com/rachthree/tensorflow-hooks).

```python
import tensorflow as tf
from tf_hooks import register_forward_hook

class TFIOHook(IOHook):
    framework = "tf"

    def __call__(self, layer, args, kwargs, outputs):
        # No kwargs for this example
        super().__call__(layer, args, outputs)
        return

save_dir = Path(".").expanduser().resolve()

model = tf.keras.applications.ResNet50(weights="imagenet")

for layer in model.layers:
    io_hook = TFIOHook(layer.name, save_dir)
    register_forward_hook(layer, io_hook)

test_input = tf.random.uniform((8, 224, 224, 3), maxval=1)
model(test_input)
```
