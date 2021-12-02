from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, Wav2Vec2PreTrainedModel, Wav2Vec2ForCTC, Wav2Vec2Model, Trainer
from transformers.file_utils import ModelOutput


class DistillTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        # if self.control.should_log:
        #     self.log({
        #         'ctc_loss': outputs.ctc_loss.item(),
        #         'feat_loss': outputs.feat_loss.item(),
        #     })
        return (loss, {'logits': outputs.logits}) if return_outputs else loss


@dataclass
class DistillLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    ctc_loss: Optional[torch.FloatTensor] = None
    feat_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Wav2Vec2ForDistill(Wav2Vec2ForCTC):
    def __init__(self, config):
        super(Wav2Vec2PreTrainedModel, self).__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        cfg = config.task_specific_params
        self._train_feat_loss = cfg['feat_loss'] > 0.0
        self._train_attn_loss = cfg['attn_loss'] > 0.0
        self._feat_loss_weight = cfg['feat_loss']
        self._attn_loss_weight = cfg['attn_loss']

        self.lm = None
        if self._train_feat_loss or self._train_attn_loss:
            self.lm = AutoModelForMaskedLM.from_pretrained(cfg['lm_name']).eval()

        if self._train_feat_loss:
            self.temporal_adapter = nn.Parameter(torch.empty(
                cfg['lm_attn_size'], cfg['sm_attn_size']
            ))
            self.feat_adapters = nn.ParameterList([
                nn.Parameter(torch.empty(
                    cfg['lm_feat_size'], cfg['sm_feat_size']))
                for _ in cfg['feat_target']
            ])
            nn.init.kaiming_uniform_(self.temporal_adapter, a=math.sqrt(5))
            for adapter in self.feat_adapters:
                nn.init.kaiming_uniform_(adapter, a=math.sqrt(5))

            self.feat_adapters_configs = [
                {'lm_index': item['lm_index'], 'sm_index': item['sm_index']}
                for item in cfg['feat_target']
            ]

        if self._train_attn_loss:
            raise NotImplemented


    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        lm_input_ids=None,
        lm_attention_mask=None,
    ):
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions or self._train_attn_loss,
            output_hidden_states=output_hidden_states or self._train_feat_loss,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        ctc_loss = None
        feat_loss = None

        if labels is not None:
            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = ctc_loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

            if self.lm:
                lm_outputs = self.lm(
                    input_ids=lm_input_ids,
                    attention_mask=lm_attention_mask,
                    output_attentions=self._train_attn_loss,
                    output_hidden_states=self._train_feat_loss,
                )

                if self._train_feat_loss:
                    feat_loss = 0.0
                    for adapter, cfg in zip(self.feat_adapters, self.feat_adapters_configs):
                        feat = lm_outputs['hidden_states'][cfg['lm_index']]
                        temporal_adapted_feat = torch.tensordot(feat, self.temporal_adapter, dims=([1], [0]))
                        adapted_feat = torch.tensordot(temporal_adapted_feat, adapter, dims=([1], [0]))

                        feat_loss += nn.functional.mse_loss(
                            outputs['hidden_states'][cfg['sm_index']], adapted_feat, reduction='mean',
                        ) / len(self.feat_adapters_configs)
                    loss += self._feat_loss_weight * feat_loss

        return DistillLMOutput(
            loss=loss,
            ctc_loss=ctc_loss,
            feat_loss=feat_loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )
