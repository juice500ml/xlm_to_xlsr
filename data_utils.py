from dataclasses import dataclass
import json
import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from functools import partial
from typing import Optional, Union

import pandas as pd
from datasets import Audio, load_dataset
from transformers import (Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor,
                          Wav2Vec2Processor)


def get_output_dir(cfg):
    p = Path(f"./runs/{cfg.dataset.name}-{cfg.dataset.language}")
    assert not p.exists()
    p.mkdir(parents=True)
    return p


def load_datasets(name, language):
    def _common_voice_process(sp, lang):
        ds = load_dataset("common_voice", lang, split=sp, cache_dir="/data/dataset/public/huggingface_datasets")
        ds = ds.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
        ds = ds.map(remove_special_characters)
        show_random_elements(ds.remove_columns(["path", "audio"]))
        return ds

    _process = {"common_voice": _common_voice_process}
    assert name in _process.keys()

    return tuple(_process[name](split, language) for split in ("train", "validation", "test"))


def get_processor(save_dir, train_ds, eval_ds):
    vocab_dict = get_vocab(train_ds, eval_ds)
    with open(save_dir / "vocab.json", "w") as vocab_file:
        json.dump(vocab_dict, vocab_file)

    tokenizer = Wav2Vec2CTCTokenizer(save_dir / "vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(save_dir)

    return processor


def cleanse_dataset(ds, processor):
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    # ds = ds.filter(filter_too_long_audio, num_proc=1)

    rand_int = random.randint(0, len(ds) - 1)

    print("Target text:", ds[rand_int]["sentence"])
    print("Input array shape:", ds[rand_int]["audio"]["array"].shape)
    print("Sampling rate:", ds[rand_int]["audio"]["sampling_rate"])

    ds = ds.map(partial(prepare_each_batch, processor=processor), remove_columns=ds.column_names, num_proc=1)
    return ds


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features):
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        # print(batch)

        return batch


def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    print(df)


def remove_special_characters(
    batch,
    punctuation_table=dict.fromkeys(i for i in range(sys.maxunicode) if (not unicodedata.category(chr(i)).startswith("L")) and (chr(i) != ' '))
):
    batch["sentence"] = unicodedata.normalize("NFKC", batch["sentence"])
    batch["sentence"] = batch["sentence"].translate(punctuation_table).lower() + " "
    return batch


def filter_too_long_audio(batch):
    return batch["audio"]["array"].shape[0] * batch["audio"]["sampling_rate"] >= 30


def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


def get_vocab(train_dataset, test_dataset, threshold=0.9999):
    counter = Counter()
    for dataset in (train_dataset, test_dataset):
        for row in dataset:
            counter.update(row['sentence'])

    sum_count = 0
    total_count = sum(counter.values())
    vocab_dict = {}

    for i, (char, count) in enumerate(counter.most_common()):
        sum_count += count
        print(f"[{char}]: {count} ({sum_count / total_count * 100:.6f}%)")
        if sum_count / total_count < threshold:
            vocab_dict[char] = i

    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]

    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    print("== Vocabulary ==")
    print(vocab_dict)

    return vocab_dict


def prepare_each_batch(batch, processor):
    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]

    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch
