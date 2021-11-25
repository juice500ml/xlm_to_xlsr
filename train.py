import argparse
import json
import random
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import hydra
import numpy as np
import pandas as pd
import torch
from datasets import Audio, load_dataset, load_metric
from tqdm import tqdm
from transformers import (Trainer, TrainingArguments, Wav2Vec2CTCTokenizer,
                          Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC,
                          Wav2Vec2Processor)


from data_utils import load_datasets, get_processor, get_output_dir, cleanse_dataset, DataCollatorCTCWithPadding



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
    train_ds, eval_ds, test_ds = load_datasets(**cfg.dataset)

    (output_dir / "processor").mkdir(exist_ok=False, parents=False)
    processor = get_processor(output_dir / "processor", train_ds, eval_ds)

    train_ds, eval_ds, test_ds = cleanse_dataset(train_ds, processor), cleanse_dataset(eval_ds, processor), cleanse_dataset(test_ds, processor)
    print(f"Preparing done: {len(train_ds)}, {len(eval_ds)}, {len(test_ds)}")

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    model = Wav2Vec2ForCTC.from_pretrained(
        **cfg.xlsr,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    model.freeze_feature_extractor()

    training_args = TrainingArguments(
        **cfg.train,
        output_dir=output_dir,
        load_best_model_at_end=True,
    )
    trainer = Trainer(
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
    (output_dir / "outputs").mkdir(exist_ok=True, parents=True)
    with open(output_dir / "outputs" / "metrics.json", "w") as f:
        json.dump(preds.metrics, f)
    np.save(output_dir / "outputs" / "preds.pkl", preds.predictions)
    np.save(output_dir / "outputs" / "labels.pkl", preds.labels)


if __name__ == "__main__":
    main()
