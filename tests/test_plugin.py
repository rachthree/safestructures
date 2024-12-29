"""Test plugin support."""

import torch
from transformers import BertConfig, BertModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    ModelOutput,
)
from utils import compare_values

from safestructures import load_file, save_file
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


class BertOutputProcessor(ModelOutputProcessor):
    """Processor for BERT model outputs."""

    data_type = BaseModelOutputWithPoolingAndCrossAttentions


class BertEncoderOutputProcessor(ModelOutputProcessor):
    """Processor for BERT encoder outputs."""

    data_type = BaseModelOutputWithPastAndCrossAttentions


def test_transformers_plugin(tmp_path):
    """Test an example plugin."""
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
    assert isinstance(
        deserialized["encoder_output"], BaseModelOutputWithPastAndCrossAttentions
    )
    assert isinstance(
        deserialized["model_output"], BaseModelOutputWithPoolingAndCrossAttentions
    )

    compare_values(results["encoder_output"], deserialized["encoder_output"])
    compare_values(results["model_output"], deserialized["model_output"])
