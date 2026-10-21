# Qwen3 wrapper — requires transformers>=4.51 (model_type qwen3).

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

try:
    from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3Model
except ImportError:
    try:
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3Model
    except ImportError as exc:
        raise ImportError(
            "Qwen3 support requires transformers>=4.51.0. "
            "Install with: pip install -r requirements-backbones-llama3-qwen3.txt"
        ) from exc

from ..utils import CausalLMOutputWithPast
from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaQwen3Config(Qwen3Config):
    model_type = "llava_qwen3"


class LlavaQwen3Model(LlavaMetaModel, Qwen3Model):
    config_class = LlavaQwen3Config

    def __init__(self, config: Qwen3Config):
        super(LlavaQwen3Model, self).__init__(config)


class LlavaQwen3ForCausalLM(Qwen3ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwen3Config

    def __init__(self, config):
        super(Qwen3ForCausalLM, self).__init__(config)
        self.model = LlavaQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
            )

        out = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if not return_dict:
            return out
        return CausalLMOutputWithPast(
            loss=out.loss,
            logits=out.logits,
            labels=labels,
            past_key_values=out.past_key_values,
            hidden_states=out.hidden_states,
            attentions=out.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs


AutoConfig.register("llava_qwen3", LlavaQwen3Config)
AutoModelForCausalLM.register(LlavaQwen3Config, LlavaQwen3ForCausalLM)
