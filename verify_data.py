import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from utils.dataset import (
    load_captions, parse_csv_filepath, get_dataloaders,
    CATEGORIES, CAT_TO_IDX, SUBJECT_IDS,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str,
                        default="/ocean/projects/cis250019p/gandotra/11785-gp-eeg")
    args = parser.parse_args()

    eeg_root = os.path.join(args.project_root, "ds005589")
    images_dir = os.path.join(args.project_root, "images")
    captions_file = os.path.join(args.project_root, "captions.txt")

    print("=" * 60)
    print("DATA VERIFICATION")
    print("=" * 60)

    # 1. Check paths exist
    print("\n[1] Checking paths...")
    for name, path in [("EEG root", eeg_root), ("Images dir", images_dir),
                        ("Captions file", captions_file)]:
        exists = os.path.exists(path)
        print(f"  {'✓' if exists else '✗'} {name}: {path}")
        if not exists:
            print(f"    ERROR: Path not found!")
            return

    # 2. Check captions
    print("\n[2] Loading captions...")
    captions = load_captions(captions_file)
    categories_found = set()
    for img_name, info in captions.items():
        categories_found.add(info["category"])
    print(f"  Categories in captions: {sorted(categories_found)}")
    print(f"  Categories matched to known list: "
          f"{len(categories_found & set(CATEGORIES))}/{len(CATEGORIES)}")

    # Show a few samples
    samples = list(captions.items())[:3]
    print("  Sample entries:")
    for img_name, info in samples:
        print(f"    {img_name} → {info['category']}: {info['caption'][:60]}...")

    # 3. Check EEG files
    print("\n[3] Checking EEG data...")
    sub = "sub-02"
    ses = "ses-01"
    run = "run-01"
    npy_path = os.path.join(eeg_root, sub, ses,
                             f"{sub}_{ses}_task-lowSpeed_{run}_1000Hz.npy")
    csv_path = os.path.join(eeg_root, sub, ses,
                             f"{sub}_{ses}_task-lowSpeed_{run}_image.csv")

    if os.path.exists(npy_path):
        data = np.load(npy_path)
        print(f"  ✓ EEG shape: {data.shape} (expected: ~(100, 122, 500))")
    else:
        print(f"  ✗ EEG file not found: {npy_path}")

    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"  ✓ CSV columns: {list(df.columns)}, rows: {len(df)}")
        # Parse first entry
        cat, img_name = parse_csv_filepath(df.iloc[0]["FilePath"])
        print(f"  Sample: FilePath → category='{cat}', image_name='{img_name}'")
        # Check if it matches captions
        if img_name in captions:
            print(f"  ✓ Caption match found for '{img_name}'")
        else:
            print(f"  ⚠ No caption match for '{img_name}' — checking similar...")
            # Show closest matches
            similar = [k for k in list(captions.keys())[:1000] if img_name[:5] in k]
            print(f"    Similar keys: {similar[:5]}")
    else:
        print(f"  ✗ CSV file not found: {csv_path}")

    # 4. Check images
    print("\n[4] Checking images directory...")
    img_files = os.listdir(images_dir)[:20]
    print(f"  Total files: {len(os.listdir(images_dir))}")
    print(f"  Sample filenames: {img_files[:10]}")

    # Check if captions image_names match files in images/
    matched = 0
    total_checked = min(100, len(captions))
    for img_name in list(captions.keys())[:total_checked]:
        for ext in [".jpg", ".jpeg", ".png", ".JPEG"]:
            if os.path.exists(os.path.join(images_dir, img_name + ext)):
                matched += 1
                break
    print(f"  Image file match rate: {matched}/{total_checked} "
          f"({matched/total_checked*100:.0f}%)")

    # 5. Test full data loader
    print("\n[5] Testing DataLoader (small subset)...")
    try:
        train_loader, val_loader, test_loader = get_dataloaders(
            eeg_root=eeg_root,
            captions_file=captions_file,
            subjects=["sub-02"],  # just one subject for speed
            train_sessions=["ses-01"],
            val_sessions=["ses-04"],
            test_sessions=["ses-05"],
            batch_size=8,
            num_workers=0,
        )

        # Grab one batch
        batch = next(iter(train_loader))
        print(f"  ✓ Batch loaded successfully!")
        print(f"    EEG shape:    {batch['eeg'].shape}")
        print(f"    Labels:       {batch['label']}")
        print(f"    Subject IDs:  {batch['subject_idx']}")
        print(f"    Image names:  {batch['image_name'][:3]}")
        print(f"    Captions:     {batch['caption'][0][:60]}...")

        print(f"\n  Train batches: {len(train_loader)}")
        print(f"  Val batches:   {len(val_loader)}")
        print(f"  Test batches:  {len(test_loader)}")

    except Exception as e:
        print(f"  ✗ DataLoader failed: {e}")
        import traceback
        traceback.print_exc()

    # 6. Count total expected trials
    print("\n[6] Expected data size (full dataset)...")
    total_subjects = 13
    sessions = 5
    runs = 4
    trials = 100
    total = total_subjects * sessions * runs * trials
    print(f"  Total trials: {total_subjects} × {sessions} × {runs} × {trials} = {total:,}")
    print(f"  Train (3 sessions): ~{total_subjects * 3 * runs * trials:,}")
    print(f"  Val (1 session):    ~{total_subjects * 1 * runs * trials:,}")
    print(f"  Test (1 session):   ~{total_subjects * 1 * runs * trials:,}")

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
