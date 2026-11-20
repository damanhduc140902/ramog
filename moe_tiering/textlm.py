"""A text-only causal LM built from transformers' base text model plus an lm_head, for models that have no CausalLM
class (glm5_next: modeling_auto 5.19 :1130 maps glm5_next_text to Glm5NextTextModel only).

TextLM(AutoModel.from_config(text_config)) has the layout of every other CausalLM the engine builds: `.model` is
the base model (its `.layers`, `.embed_tokens`, `.norm`), `.lm_head` the head, `.config` the text config, and the
state-dict keys are `model.*` and `lm_head.weight`, which is the VLM checkpoint's layout once `model.language_model.`
becomes `model.` (arch prefix rule).

Its forward is the VLM's text-only path, step by step (5.19 Glm5NextForConditionalGeneration.forward :2240-2257 and
Glm5NextModel.forward :2007-2056): embed the ids with the base model's input embeddings, run the base model with
input_ids=None and those embeddings, then apply lm_head to the last hidden states (sliced by logits_to_keep). The
VLM adds nothing else for text, so the logits are the same (tests/test_arch_sites.py checks it bit for bit on a tiny
model). It returns a CausalLMOutputWithPast, so callers read `.logits` as for any CausalLM.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast


class TextLM(nn.Module):
    base_model_prefix = "model"  # transformers scopes the base model's weight transforms under this prefix

    def __init__(self, model: nn.Module, lm_head: nn.Linear | None = None):
        super().__init__()
        self.model = model
        self.config = model.config
        if lm_head is None:
            emb = model.get_input_embeddings().weight
            lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False, device=emb.device,
                                dtype=emb.dtype)
        self.lm_head = lm_head
        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head.weight = model.get_input_embeddings().weight

    @property
    def use_kernels(self) -> bool:
        return bool(getattr(self.model, "use_kernels", False))

    @property
    def device(self) -> torch.device:
        return self.lm_head.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.lm_head.weight.dtype

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def initialize_weights(self) -> None:
        """The base model's initialize_weights(); lm_head is always loaded from the checkpoint (or tied)."""
        self.model.initialize_weights()

    def _get_dtype_plan(self, dtype: torch.dtype) -> dict:
        return self.model._get_dtype_plan(dtype)

    def forward(self, input_ids: torch.LongTensor | None = None, attention_mask: torch.Tensor | None = None,
                position_ids: torch.LongTensor | None = None, past_key_values=None,
                inputs_embeds: torch.FloatTensor | None = None, use_cache: bool | None = None,
                logits_to_keep: int | torch.Tensor = 0, **kwargs) -> CausalLMOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        out = self.model(input_ids=None, attention_mask=attention_mask, position_ids=position_ids,
                         past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        hidden = out.last_hidden_state
        keep = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden[:, keep, :])
        return CausalLMOutputWithPast(logits=logits, past_key_values=out.past_key_values)
