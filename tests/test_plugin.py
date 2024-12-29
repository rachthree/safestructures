"""Test plugin support."""

import torch
from transformers import BertConfig, BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from utils import compare_values

from safestructures import load_file, save_file
from safestructures.processors.iterable import DataclassProcessor


class BertModelOutputProcessor(DataclassProcessor):
    """Processor for transformer's ModelOutput class."""

    data_type = BaseModelOutputWithPoolingAndCrossAttentions

    def deserialize(
        self, serialized: dict, **kwargs
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
        """Overload DataclassProcessor.deserialize.

        This is so the proper BertModelOutput is provided.
        """
        mo_kwargs = {}
        for k, v in serialized.items():
            mo_kwargs[k] = self.serializer.deserialize(v)

        model_output_instance = self.data_type(**mo_kwargs)

        return model_output_instance


def test_transformers_plugin(tmp_path):
    """Test an example plugin."""
    config = BertConfig()
    model = BertModel(config)
    test_input_ids = torch.tensor([[0] * 128])
    test_output = model(test_input_ids)

    test_filepath = tmp_path / "test.safestructures"
    save_file(test_output, test_filepath, plugins=[BertModelOutputProcessor])

    deserialized = load_file(test_filepath, plugins=[BertModelOutputProcessor])
    assert isinstance(deserialized, BaseModelOutputWithPoolingAndCrossAttentions)

    compare_values(test_output, deserialized)
