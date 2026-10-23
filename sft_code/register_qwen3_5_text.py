"""Register the text-only Qwen3.5 policy checkpoint for ms-swift.

``Output/SFT/sft_output`` has ``model_type: qwen3_5_text`` in
``config.json`` while its legacy ``args.json`` still points to the
multimodal ``qwen3_5`` model type. ms-swift (this fork) does not register
``qwen3_5_text`` and only ships a multimodal ``qwen3_5`` template whose
collator needs ``model.get_rope_index``, so SFT/export would otherwise fail
on the text-only policy checkpoint.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from transformers import PreTrainedModel, PretrainedConfig

from swift.model import Model, ModelGroup, ModelLoader, ModelMeta, register_model
from swift.template import Template, TemplateMeta, register_template
from swift.template.templates.qwen import Qwen3_5Template, QwenTemplateMeta
from swift.utils import Processor


class Qwen3_5TextLoader(ModelLoader):

    def get_model(
        self,
        model_dir: str,
        config: PretrainedConfig,
        processor: Processor,
        model_kwargs: Dict[str, Any],
    ) -> Optional[PreTrainedModel]:
        from transformers import Qwen3_5ForCausalLM
        from swift.model.models.qwen import _patch_qwen3_5_linear_attention_sequence_parallel

        self.auto_model_cls = self.auto_model_cls or Qwen3_5ForCausalLM
        _patch_qwen3_5_linear_attention_sequence_parallel()
        return super().get_model(model_dir, config, processor, model_kwargs)


class Qwen3_5TextTemplate(Qwen3_5Template):

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        # Text-only Qwen3.5 has no multimodal RoPE/position-id machinery.
        return Template._data_collator(self, batch, padding_to=padding_to)


register_template(
    QwenTemplateMeta(
        template_type='qwen3_5_text',
        template_cls=Qwen3_5TextTemplate,
        default_system=None,
        agent_template='qwen3_5',
        is_thinking=True,
        thinking_prefix='<think>\n',
        non_thinking_prefix='<think>\n\n</think>\n\n',
    ))

register_model(
    ModelMeta(
        model_type='qwen3_5_text',
        model_groups=[],
        loader=Qwen3_5TextLoader,
        template='qwen3_5_text',
        architectures=['Qwen3_5ForCausalLM'],
        is_multimodal=False,
    ))
