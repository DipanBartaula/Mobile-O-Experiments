import argparse
import os
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from mobileo.train.post_train import DataArguments, DataCollatorForSupervisedDataset, LazySupervisedMixDataset


DEFAULT_DATASET_ROOT = "/iopsstor/scratch/cscs/dbartaula/Mobile-O-Datasets"


class DummyTokenizer:
    def __init__(self, model_max_length: int = 512):
        self.model_max_length = model_max_length
        self.pad_token_id = 0
        self.bos_token_id = 1

    def __call__(self, text):
        return SimpleNamespace(input_ids=self.encode(text, add_special_tokens=True))

    def encode(self, text, add_special_tokens=True):
        ids = [((ord(ch) % 251) + 2) for ch in text]
        if add_special_tokens:
            return [self.bos_token_id] + ids
        return ids

    def convert_tokens_to_ids(self, token):
        if token == "<|im_end|>":
            return 2
        return 3


def _find_first_dir_with_direct_tar(root_dir):
    for cur_root, _, files in os.walk(root_dir):
        if any(name.endswith(".tar") for name in files):
            return cur_root
    return None


def _summarize_batch(batch_idx, batch):
    print(f"\n[post_train.py loader] batch={batch_idx}")
    print(f"keys={list(batch.keys())}")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={tuple(value.shape)} dtype={value.dtype}")
        elif isinstance(value, list):
            print(f"  {key}: list_len={len(value)}")
        else:
            print(f"  {key}: type={type(value).__name__}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=2)
    args = parser.parse_args()

    candidate_roots = [
        os.path.join(args.dataset_root, "Mobile-O-Post-Train"),
        os.path.join(args.dataset_root, "Mobile-O-SFT"),
        os.path.join(args.dataset_root, "Mobile-O-Pre-Train"),
    ]
    existing_roots = [p for p in candidate_roots if os.path.isdir(p)]
    if not existing_roots:
        raise FileNotFoundError(f"No expected dataset folders found under: {args.dataset_root}")

    direct_tar_dir = None
    for root in existing_roots:
        direct_tar_dir = _find_first_dir_with_direct_tar(root)
        if direct_tar_dir is not None:
            break
    if direct_tar_dir is None:
        raise FileNotFoundError(f"No .tar files found under expected folders in: {args.dataset_root}")

    data_args = DataArguments(
        data_path=None,
        image_folder=direct_tar_dir,
        is_multimodal=True,
        aspect_ratio_size_und=[512.0, 512.0],
        aspect_ratio_size_gen=[512.0, 512.0],
    )
    tokenizer = DummyTokenizer(model_max_length=512)

    dataset = LazySupervisedMixDataset(data_path=data_args.data_path, tokenizer=tokenizer, data_args=data_args)
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    print(f"[post_train.py loader] dataset_root={args.dataset_root}")
    print(f"[post_train.py loader] image_folder={data_args.image_folder}")
    print(f"[post_train.py loader] dataset_len={len(dataset)}")

    for idx, batch in enumerate(loader):
        _summarize_batch(idx + 1, batch)
        if idx + 1 >= args.max_batches:
            break


if __name__ == "__main__":
    main()
