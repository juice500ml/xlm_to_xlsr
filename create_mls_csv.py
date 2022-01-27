import os
from pathlib import Path

import pandas as pd


def _read_transcription(foldername):
    df = pd.read_csv(foldername / 'transcripts.txt', sep='\t', names=['audio', 'sentence'])
    df.audio = df.audio.apply(lambda s: f'{foldername}/audio/{"/".join(s.split("_")[:2])}/{s}.wav')
    return df


def _prepare_handles(mls_root, hour):
    assert hour in (1, 10)
    one_handles = mls_root / 'train' / 'limited_supervision' / '1hr'
    s = set()

    for f in one_handles.glob('*/handles.txt'):
        s.update([line.strip() for line in open(f).readlines()])

    if hour == 10:
        nine_handle = mls_root / 'train' / 'limited_supervision' / '9hr' / 'handles.txt'
        s.update([line.strip() for line in open(nine_handle).readlines()])

    return s


def _remove_handle_mask(df, handles):
    return df.apply(
        lambda x: x.audio.split('/')[-1].split('.')[0] in handles, axis=1
    )


if __name__ == '__main__':
    mls_root = Path(os.environ['MLS_OPUS_ROOT'])
    assert mls_root.exists()

    dataset_name = mls_root.name.split('_opus')[0]
    print(f'Create dataset: {dataset_name}')

    dataset_csv_root = Path('dataset_csv')
    dataset_csv_root.mkdir(exist_ok=True)

    for split in ('train', 'dev', 'test'):
        df = _read_transcription(mls_root / split)
        df.to_csv(dataset_csv_root / f'{dataset_name}_{split}.csv', index=False)
        print(f'{split}: {len(df)}')
        if split == 'train':
            for hour in (1, 10):
                hour_handles = _prepare_handles(mls_root, hour)
                hour_df = df[_remove_handle_mask(df, hour_handles)]
                assert len(hour_handles) == len(hour_df)
                hour_df.to_csv(dataset_csv_root / f'{dataset_name}_{split}_{hour}h.csv', index=False)
                print(f'{split}_{hour}h: {len(hour_df)}')
