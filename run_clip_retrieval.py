#!/usr/bin/env python3
"""
Task 2A: Image-Caption Retrieval with CLIP
============================================
Usage on PSC:
    python run_clip_retrieval.py
    python run_clip_retrieval.py --clip_model openai/clip-vit-large-patch14
"""

import argparse
import os
import sys
import glob
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from utils.clip_retrieval import (
    load_clip_model,
    compute_image_embeddings,
    compute_text_embeddings,
    run_full_retrieval_eval,
    compute_map,
)
from utils.dataset import (
    load_captions, parse_csv_filepath,
    CATEGORIES, CAT_TO_IDX, SUBJECT_IDS,
)


def find_image_path(image_name: str, images_dir: str) -> str:
    """
    Try to find the actual image file for a given image_name.
    Tries multiple extensions and naming conventions.
    """
    for ext in [".jpg", ".jpeg", ".png", ".JPEG", ".JPG"]:
        # Direct match
        path = os.path.join(images_dir, image_name + ext)
        if os.path.exists(path):
            return path
        # With _resized suffix
        path = os.path.join(images_dir, image_name + "_resized" + ext)
        if os.path.exists(path):
            return path
    return None


def collect_test_images(
    eeg_root: str,
    images_dir: str,
    captions: dict,
    test_sessions=["ses-05"],
):
    """
    Collect unique test images that have: caption + image file on disk.
    Returns list of dicts with image_path, caption, category_idx.
    """
    seen = set()
    records = []
    missing_image = 0

    for sub in SUBJECT_IDS:
        for ses in test_sessions:
            for run_id in range(1, 5):
                csv_name = f"{sub}_{ses}_task-lowSpeed_run-{run_id:02d}_image.csv"
                csv_path = os.path.join(eeg_root, sub, ses, csv_name)
                if not os.path.exists(csv_path):
                    continue

                import pandas as pd
                trial_df = pd.read_csv(csv_path)

                for _, row in trial_df.iterrows():
                    category, image_name = parse_csv_filepath(row["FilePath"])

                    if image_name in seen:
                        continue
                    seen.add(image_name)

                    # Get caption
                    if image_name not in captions:
                        continue
                    cap_info = captions[image_name]

                    if cap_info["category"] not in CAT_TO_IDX:
                        continue

                    # Find image file
                    img_path = find_image_path(image_name, images_dir)
                    if img_path is None:
                        missing_image += 1
                        continue

                    records.append({
                        "image_name": image_name,
                        "image_path": img_path,
                        "caption": cap_info["caption"],
                        "category": cap_info["category"],
                        "category_idx": CAT_TO_IDX[cap_info["category"]],
                    })

    print(f"Collected {len(records)} unique test images with captions and files")
    if missing_image > 0:
        print(f"  ⚠ {missing_image} images not found on disk (will be skipped)")

    return records


def main():
    parser = argparse.ArgumentParser(description="Task 2A: CLIP Image-Caption Retrieval")
    parser.add_argument("--project_root", type=str,
                        default="/ocean/projects/cis250019p/gandotra/11785-gp-eeg")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_dir", type=str, default="./results")
    args = parser.parse_args()

    eeg_root = os.path.join(args.project_root, "ds005589")
    images_dir = os.path.join(args.project_root, "images")
    captions_file = os.path.join(args.project_root, "captions.txt")

    os.makedirs(args.save_dir, exist_ok=True)

    # ─── Load Captions & CLIP ────────────────────────────────────────────
    captions = load_captions(captions_file)
    clip_model, processor = load_clip_model(args.clip_model, args.device)

    # ─── Collect Test Images ─────────────────────────────────────────────
    records = collect_test_images(eeg_root, images_dir, captions)

    if len(records) == 0:
        print("ERROR: No test images found. Check paths and image directory.")
        print(f"  Images dir: {images_dir}")
        print(f"  Sample contents: {os.listdir(images_dir)[:10]}")
        return

    image_paths = [r["image_path"] for r in records]
    caption_texts = [r["caption"] for r in records]
    labels = torch.tensor([r["category_idx"] for r in records])

    # ─── Compute Embeddings ──────────────────────────────────────────────
    print("\nComputing image embeddings...")
    image_embeds = compute_image_embeddings(
        clip_model, processor, image_paths, args.device, args.batch_size)

    print("Computing text embeddings...")
    text_embeds = compute_text_embeddings(
        clip_model, processor, caption_texts, args.device, args.batch_size)

    # Image i → caption i (1:1 mapping)
    gt_indices = torch.arange(len(records))

    # ─── Full Evaluation ─────────────────────────────────────────────────
    results = run_full_retrieval_eval(
        query_embeds=image_embeds,
        candidate_embeds=text_embeds,
        ground_truth_indices=gt_indices,
        query_labels=labels,
        candidate_labels=labels,
        image_embeds=image_embeds,
    )

    # ─── Per-Class MAP ───────────────────────────────────────────────────
    print("\nPer-Class MAP:")
    print("-" * 40)
    for cat_idx, cat_name in enumerate(CATEGORIES):
        mask = labels == cat_idx
        if mask.sum() == 0:
            continue
        class_query = image_embeds[mask]
        class_gt = gt_indices[mask]
        class_map = compute_map(class_query, text_embeds, class_gt) * 100
        print(f"  {cat_name:15s}: {class_map:.2f}%")

    # ─── Save ────────────────────────────────────────────────────────────
    save_path = os.path.join(args.save_dir, "clip_retrieval_results.pt")
    torch.save({
        "image_embeds": image_embeds,
        "text_embeds": text_embeds,
        "labels": labels,
        "results": results,
        "records": records,
    }, save_path)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
