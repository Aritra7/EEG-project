import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional


# ─── Category Mapping ────────────────────────────────────────────────────────
CATEGORIES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
CAT_TO_IDX = {cat: i for i, cat in enumerate(CATEGORIES)}
CAT_TO_IDX["flower"] = CAT_TO_IDX["pottedplant"]
IDX_TO_CAT = {i: cat for i, cat in enumerate(CATEGORIES)}

SUBJECT_IDS = [
    "sub-02", "sub-03", "sub-05", "sub-09", "sub-14", "sub-15",
    "sub-17", "sub-19", "sub-20", "sub-23", "sub-24", "sub-28", "sub-29"
]
SUB_TO_IDX = {s: i for i, s in enumerate(SUBJECT_IDS)}


def parse_csv_filepath(filepath: str) -> Tuple[str, str]:
    """
    Extract category and image_name from CSV FilePath column.

    Example input:
        C:\\Users\\Huawei\\...\\pic_10000_resized\\diningtable\\n03201208_14446_resized.jpg
    Returns:
        ("diningtable", "n03201208_14446")
    """
    # Normalize Windows path separators
    filepath = filepath.replace("\\", "/")
    parts = filepath.rstrip("/").split("/")

    # Category is second-to-last component, filename is last
    category = parts[-2] if len(parts) >= 2 else "unknown"
    filename = parts[-1]

    # Strip _resized suffix and extension to get image_name
    name_no_ext = os.path.splitext(filename)[0]  # "n03201208_14446_resized"
    if name_no_ext.endswith("_resized"):
        name_no_ext = name_no_ext[: -len("_resized")]  # "n03201208_14446"

    return category, name_no_ext


def load_captions(captions_file: str) -> Dict[str, dict]:
    """
    Load captions.txt into a dict keyed by image_name.

    File format (tab-separated, no header row based on actual data):
        dataset  category  image_name  abstracted(caption)
        ImageNet bicycle   n02835271_1031  Tandem bicycle parked beside a wooden fence

    Returns:
        { "n02835271_1031": {"category": "bicycle", "caption": "Tandem bicycle...", "dataset": "ImageNet"}, ... }
    """
    captions = {}

    with open(captions_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 4:
                continue

            dataset = parts[0].strip()
            category = parts[1].strip()
            image_name = parts[2].strip()
            caption = parts[3].strip()

            # Skip if this looks like a header row
            if dataset.lower() == "dataset" or category.lower() == "category":
                continue

            captions[image_name] = {
                "category": category,
                "caption": caption,
                "dataset": dataset,
            }

    print(f"Loaded {len(captions)} captions from {captions_file}")
    return captions


class EEGDataset(Dataset):
    """
    Dataset for EEG-Image classification and retrieval.

    Each item returns:
        eeg:        (122, 500) float tensor
        label:      int — category index [0, 19]
        subject_idx: int — subject index [0, 12]
        image_name: str — image identifier (e.g. "n03201208_14446")
        caption:    str — associated caption text
        category:   str — category name
    """

    def __init__(
        self,
        eeg_root: str,
        subjects: List[str],
        sessions: List[str],
        captions: Dict[str, dict],
        transform=None,
        normalize: bool = True,
    ):
        self.eeg_root = eeg_root
        self.transform = transform
        self.normalize = normalize
        self.captions = captions

        self.trials = []
        self._load_all_trials(subjects, sessions)

        print(f"  → {len(self.trials)} trials "
              f"({len(subjects)} subjects × {len(sessions)} sessions)")

    def _load_all_trials(self, subjects: List[str], sessions: List[str]):
        skipped_no_caption = 0
        skipped_bad_category = 0

        for sub in subjects:
            sub_idx = SUB_TO_IDX[sub]
            for ses in sessions:
                for run_id in range(1, 5):
                    run_str = f"run-{run_id:02d}"
                    nc, bc = self._load_run(sub, sub_idx, ses, run_str)
                    skipped_no_caption += nc
                    skipped_bad_category += bc

        if skipped_no_caption > 0:
            print(f"  ⚠ Skipped {skipped_no_caption} trials (no caption match)")
        if skipped_bad_category > 0:
            print(f"  ⚠ Skipped {skipped_bad_category} trials (unknown category)")

    def _load_run(self, sub: str, sub_idx: int, ses: str, run_str: str):
        base_dir = os.path.join(self.eeg_root, sub, ses)

        npy_name = f"{sub}_{ses}_task-lowSpeed_{run_str}_1000Hz.npy"
        csv_name = f"{sub}_{ses}_task-lowSpeed_{run_str}_image.csv"

        npy_path = os.path.join(base_dir, npy_name)
        csv_path = os.path.join(base_dir, csv_name)

        if not os.path.exists(npy_path) or not os.path.exists(csv_path):
            return 0, 0

        # Load EEG: shape (N, 122, T)
        eeg_data = np.load(npy_path, mmap_mode="r")

        # Load trial CSV (only has FilePath column)
        trial_df = pd.read_csv(csv_path)

        assert len(eeg_data) == len(trial_df), (
            f"Mismatch: {npy_path} has {len(eeg_data)} trials, "
            f"but {csv_path} has {len(trial_df)} rows"
        )

        skipped_no_caption = 0
        skipped_bad_category = 0

        for i in range(len(eeg_data)):
            filepath = trial_df.iloc[i]["FilePath"]

            # Parse category and image_name from Windows path
            category_from_path, image_name = parse_csv_filepath(filepath)

            # Look up caption
            if image_name in self.captions:
                cap_info = self.captions[image_name]
                category = cap_info["category"]
                caption = cap_info["caption"]
            else:
                # Fallback: use category from path, no caption
                category = category_from_path
                caption = ""
                skipped_no_caption += 1

            if category not in CAT_TO_IDX:
                skipped_bad_category += 1
                continue

            self.trials.append({
                "npy_path": npy_path,
                "trial_idx": i,
                "label": CAT_TO_IDX[category],
                "subject_idx": sub_idx,
                "image_name": image_name,
                "caption": caption,
                "category": category,
            })

        return skipped_no_caption, skipped_bad_category

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        trial = self.trials[idx]

        # Load single EEG trial via memory mapping (efficient)
        eeg_data = np.load(trial["npy_path"], mmap_mode="r")
        eeg = eeg_data[trial["trial_idx"]].copy().astype(np.float32)
        # Data is (500, 122) on disk, we need (122, 500)
        if eeg.shape[0] != 122:
            eeg = eeg.T

        # Z-score normalize per trial
        if self.normalize:
            mean = eeg.mean()
            std = eeg.std() + 1e-8
            eeg = (eeg - mean) / std

        eeg = torch.from_numpy(eeg)  # (122, 500)

        if self.transform:
            eeg = self.transform(eeg)

        return {
            "eeg": eeg,
            "label": trial["label"],
            "subject_idx": trial["subject_idx"],
            "image_name": trial["image_name"],
            "caption": trial["caption"],
        }


def collate_fn(batch):
    """Custom collate that handles string fields."""
    return {
        "eeg": torch.stack([b["eeg"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "subject_idx": torch.tensor([b["subject_idx"] for b in batch], dtype=torch.long),
        "image_name": [b["image_name"] for b in batch],
        "caption": [b["caption"] for b in batch],
    }


def get_dataloaders(
    eeg_root: str,
    captions_file: str,
    subjects: List[str],
    train_sessions: List[str],
    val_sessions: List[str],
    test_sessions: List[str],
    batch_size: int = 64,
    num_workers: int = 4,
    normalize: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders with session-based splits."""

    print("Loading captions...")
    captions = load_captions(captions_file)

    print("Building train set...")
    train_ds = EEGDataset(eeg_root, subjects, train_sessions, captions, normalize=normalize)
    print("Building val set...")
    val_ds = EEGDataset(eeg_root, subjects, val_sessions, captions, normalize=normalize)
    print("Building test set...")
    test_ds = EEGDataset(eeg_root, subjects, test_sessions, captions, normalize=normalize)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader

