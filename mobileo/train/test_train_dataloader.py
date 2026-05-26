import argparse
import os
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from mobileo.train.train import DataArguments, DataCollatorForSupervisedDataset, LazySupervisedMixDataset


DEFAULT_DATASET_ROOT = "/iopsstor/scratch/cscs/dbartaula/Mobile-O-Datasets"


class DummyTokenizer:
    def __init__(self, model_max_length: int = 512):
        self.model_max_length = model_max_length
        self.pad_token_id = 0
        self.bos_token_id = 1

    def encode(self, text, add_special_tokens=True):
        ids = [((ord(ch) % 251) + 2) for ch in text]
        if add_special_tokens:
            return [self.bos_token_id] + ids
        return ids

    def apply_chat_template(self, messages):
        all_ids = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            chunk = f"<|im_start|>{role}\n{content}<|im_end|>\n"
            all_ids.extend(self.encode(chunk, add_special_tokens=False))
        return all_ids


def _existing_dirs(paths):
    return [p for p in paths if os.path.isdir(p)]


def _summarize_batch(batch_idx, batch):
    print(f"\n[train.py loader] batch={batch_idx}")
    print(f"keys={list(batch.keys())}")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={tuple(value.shape)} dtype={value.dtype}")
        elif isinstance(value, list):
            print(f"  {key}: list_len={len(value)}")
        else:
            print(f"  {key}: type={type(value).__name__}")


def _run_loader_for_dirs(args, tokenizer, dirs, label):
    data_args = DataArguments(
        data_path=None,
        image_folder=",".join(dirs),
        tokenizer_max_length=512,
        is_multimodal=True,
    )

    dataset = LazySupervisedMixDataset(data_path=data_args.data_path, tokenizer=tokenizer, data_args=data_args)
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    print(f"\n{'=' * 80}")
    print(f"[train.py loader] split={label}")
    print(f"[train.py loader] image_folder={data_args.image_folder}")
    print(f"[train.py loader] dataset_len={len(dataset)}")
    print(f"{'=' * 80}")

    for idx, batch in enumerate(loader):
        _summarize_batch(idx + 1, batch)
        if idx + 1 >= args.max_batches:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument(
        "--split-mode",
        choices=["separate", "combined"],
        default="separate",
        help="Print batches per dataset folder separately, or use a single combined loader.",
    )
    args = parser.parse_args()

    expected_dirs = [
        ("Mobile-O-Pre-Train", os.path.join(args.dataset_root, "Mobile-O-Pre-Train")),
        ("Mobile-O-SFT", os.path.join(args.dataset_root, "Mobile-O-SFT")),
        ("Mobile-O-Post-Train", os.path.join(args.dataset_root, "Mobile-O-Post-Train")),
    ]
    candidate_dirs = _existing_dirs([p for _, p in expected_dirs])
    if not candidate_dirs:
        raise FileNotFoundError(f"No expected dataset folders found under: {args.dataset_root}")

    print(f"[train.py loader] dataset_root={args.dataset_root}")
    tokenizer = DummyTokenizer(model_max_length=512)

    if args.split_mode == "combined":
        _run_loader_for_dirs(args, tokenizer, candidate_dirs, label="combined")
        return

    for name, path in expected_dirs:
        if not os.path.isdir(path):
            print(f"\n[train.py loader] split={name} missing, skipping path={path}")
            continue
        _run_loader_for_dirs(args, tokenizer, [path], label=name)


if __name__ == "__main__":
    main()
