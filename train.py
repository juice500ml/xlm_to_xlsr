import json
from functools import partial

import hydra
import numpy as np
from datasets import load_metric
from omegaconf import OmegaConf
from transformers import (AutoTokenizer, Trainer, TrainingArguments)

from data_utils import (get_output_dir, get_processor, load_datasets)
from model_utils import Wav2Vec2ForDistill, DistillTrainer


def get_compute_metrics(processor):
    wer_metric = load_metric("wer")
    cer_metric = load_metric("cer")

    def _compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)

        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        return {
            "wer": wer_metric.compute(predictions=pred_str, references=label_str),
            "cer": cer_metric.compute(predictions=pred_str, references=label_str),
        }
    return _compute_metrics


@hydra.main(config_path="config")
def main(cfg):
    output_dir = get_output_dir(cfg)
    (train_ds, eval_ds, test_ds), cleanser, collator = load_datasets(**cfg.dataset)

    (output_dir / "processor").mkdir(exist_ok=False, parents=False)
    processor = get_processor(output_dir / "processor", train_ds, eval_ds)
    lm_tokenizer = AutoTokenizer.from_pretrained(cfg.distill.lm_name)

    _cleanse_ds = partial(cleanser, processor=processor, lm_tokenizer=lm_tokenizer)
    train_ds, eval_ds, test_ds = _cleanse_ds(train_ds), _cleanse_ds(eval_ds), _cleanse_ds(test_ds)
    print(f"Preparing done: {len(train_ds)}, {len(eval_ds)}, {len(test_ds)}")

    data_collator = collator(processor=processor, lm_tokenizer=lm_tokenizer)

    model = Wav2Vec2ForDistill.from_pretrained(
        **cfg.xlsr,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        task_specific_params=OmegaConf.to_container(cfg.distill, resolve=True)
    )
    if cfg.distill.random_init:
        model.apply(model._init_weights)
    else:
        model.freeze_feature_extractor()

    training_args = TrainingArguments(
        **cfg.train,
        output_dir=output_dir,
        load_best_model_at_end=True,
    )
    trainer = DistillTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=get_compute_metrics(processor),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=processor.feature_extractor,
    )
    trainer.train()

    preds = trainer.predict(test_dataset=test_ds)
    trainer.log(preds.metrics)

    (output_dir / "outputs").mkdir(exist_ok=True, parents=True)
    with open(output_dir / "outputs" / "metrics.json", "w") as f:
        json.dump(preds.metrics, f)
    np.save(output_dir / "outputs" / "preds.pkl", preds.predictions)
    np.save(output_dir / "outputs" / "label_ids.pkl", preds.label_ids)


if __name__ == "__main__":
    main()
