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

        log_data = {'ctc_loss': outputs.ctc_loss.mean().item()}
        if outputs.feat_loss is not None:
            log_data['feat_loss'] = outputs.feat_loss.mean().item()
        self.log(log_data)

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
        self._vocab_size = config.vocab_size

        cfg = config.task_specific_params
        self._train_feat_loss = cfg['feat_loss'] > 0.0
        self._train_attn_loss = cfg['attn_loss'] > 0.0
        self._feat_loss_weight = cfg['feat_loss']
        self._attn_loss_weight = cfg['attn_loss']

        self.lm = None
        if self._train_feat_loss or self._train_attn_loss:
            self.lm = AutoModelForMaskedLM.from_pretrained(cfg['lm_name']).eval()

        if self._train_feat_loss:
            self._interpolation_do_filter = cfg['interpolation']['filter_out_pad']
            self._interpolation_do_shrink = cfg['interpolation']['shrink']
            self.temporal_adapter_kwargs = dict(
                mode=cfg['interpolation']['mode'],
                align_corners=True if cfg['interpolation']['mode'] == 'linear' else None)
            self.feat_adapter = nn.Parameter(torch.empty(
                    cfg['sm_feat_size'], cfg['lm_feat_size']))
            nn.init.kaiming_uniform_(self.feat_adapter, a=math.sqrt(5))

            assert len(cfg['feat_target']) == 1
            self.feat_adapter_config = {
                'lm_index': cfg['feat_target'][0]['lm_index'], 'sm_index': cfg['feat_target'][0]['sm_index']}

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
                    sm_feat = outputs['hidden_states'][self.feat_adapter_config['sm_index']]
                    lm_feat = lm_outputs['hidden_states'][self.feat_adapter_config['lm_index']]

                    for sm_logit, sm_length, sm_f, lm_mask, lm_f in \
                        zip(logits, input_lengths, sm_feat, lm_attention_mask, lm_feat):

                        # Generate sm_mask to filter out speech features
                        logit_mask = torch.ones(sm_logit.shape[0], dtype=bool, device=sm_length.device)
                        logit_mask[sm_length:] = False

                        if self._interpolation_do_filter:
                            sm_mask = ((sm_logit.argmax(1) < (self._vocab_size - 2)) & logit_mask)
                            sm_mask = sm_mask if sm_mask.sum() > 1 else logit_mask
                        else:
                            sm_mask = logit_mask

                        sm_f = sm_f[sm_mask]
                        sm_logit = sm_logit[sm_mask]
                        lm_f = lm_f[lm_mask.bool()]

                        if self._interpolation_do_shrink:
                            sm_f = self._shrink(sm_logit.argmax(1), sm_f)

                        # Feature interpolation (SM -> LM)
                        feature_adapted_sm_f = torch.tensordot(sm_f, self.feat_adapter, dims=([1], [0]))

                        # Time interpolation (LM -> SM)
                        time_adapted_lm_f = nn.functional.interpolate(
                            input=torch.unsqueeze(lm_f, 0).permute(0, 2, 1),
                            size=sm_f.shape[0],
                            **self.temporal_adapter_kwargs)
                        time_adapted_lm_f = time_adapted_lm_f.squeeze().permute(1, 0)

                        # MSE Loss
                        feat_loss += nn.functional.mse_loss(
                            feature_adapted_sm_f, time_adapted_lm_f, reduction='mean',
                        )

                    loss += self._feat_loss_weight * feat_loss / logits.shape[0]

        return DistillLMOutput(
            loss=loss,
            ctc_loss=ctc_loss,
            feat_loss=feat_loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )

    @staticmethod
    def _shrink(logit_max, feats):
        aligned_feats = []

        i = 0
        while i < len(logit_max):
            j = 1
            while (i + j) < len(logit_max) and logit_max[i + j].item() == logit_max[i].item():
                j += 1
            aligned_feats.append(feats[i:i + j].mean(0))
            i += j

        return torch.stack(aligned_feats)
