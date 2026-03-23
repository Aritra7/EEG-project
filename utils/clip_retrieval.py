"""
CLIP Retrieval Pipeline (Task 2A)
==================================
Image-to-caption retrieval using pretrained CLIP.
Computes Recall@K, BERTScore, CLIPScore, and MAP.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm


def load_clip_model(model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
    """Load pretrained CLIP model and processor from HuggingFace."""
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    print(f"Loaded CLIP model: {model_name}")
    print(f"  Image embed dim: {model.config.projection_dim}")
    print(f"  Text embed dim:  {model.config.projection_dim}")

    return model, processor


@torch.no_grad()
def compute_image_embeddings(
    model, processor, image_paths: List[str], device: str = "cuda", batch_size: int = 32
) -> torch.Tensor:
    """Compute normalized CLIP image embeddings."""
    from PIL import Image

    all_embeds = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Image embeddings"):
        batch_paths = image_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        embeds = model.get_image_features(**inputs)
        embeds = F.normalize(embeds, dim=-1)
        all_embeds.append(embeds.cpu())

    return torch.cat(all_embeds, dim=0)


@torch.no_grad()
def compute_text_embeddings(
    model, processor, captions: List[str], device: str = "cuda", batch_size: int = 64
) -> torch.Tensor:
    """Compute normalized CLIP text embeddings."""
    all_embeds = []
    for i in tqdm(range(0, len(captions), batch_size), desc="Text embeddings"):
        batch_captions = captions[i : i + batch_size]
        inputs = processor(text=batch_captions, return_tensors="pt",
                           padding=True, truncation=True, max_length=77).to(device)
        embeds = model.get_text_features(**inputs)
        embeds = F.normalize(embeds, dim=-1)
        all_embeds.append(embeds.cpu())

    return torch.cat(all_embeds, dim=0)


# ─── Retrieval ───────────────────────────────────────────────────────────────

def retrieve_top_k(
    query_embeds: torch.Tensor,
    candidate_embeds: torch.Tensor,
    k: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Retrieve top-K candidates for each query via cosine similarity.

    Returns:
        scores: (N, K) — similarity scores
        indices: (N, K) — candidate indices
    """
    # Similarity matrix: (N_query, N_candidates)
    sim = query_embeds @ candidate_embeds.T
    scores, indices = sim.topk(k, dim=1)
    return scores, indices


# ─── Evaluation Metrics ──────────────────────────────────────────────────────

def recall_at_k(
    query_embeds: torch.Tensor,
    candidate_embeds: torch.Tensor,
    ground_truth_indices: torch.Tensor,
    k_values: List[int] = [1, 3, 5],
) -> Dict[str, float]:
    """
    Instance-level Recall@K.
    ground_truth_indices[i] = index of correct caption for query i.
    """
    sim = query_embeds @ candidate_embeds.T
    results = {}

    for k in k_values:
        _, top_k_idx = sim.topk(k, dim=1)
        hits = (top_k_idx == ground_truth_indices.unsqueeze(1)).any(dim=1)
        results[f"R@{k}"] = hits.float().mean().item() * 100

    return results


def class_aware_recall_at_k(
    query_embeds: torch.Tensor,
    candidate_embeds: torch.Tensor,
    query_labels: torch.Tensor,
    candidate_labels: torch.Tensor,
    k_values: List[int] = [1, 3, 5],
) -> Dict[str, float]:
    """
    Class-aware Recall@K: correct if ANY caption from the same class is in top-K.
    """
    sim = query_embeds @ candidate_embeds.T
    results = {}

    for k in k_values:
        _, top_k_idx = sim.topk(k, dim=1)
        hits = 0
        for i in range(len(query_embeds)):
            retrieved_labels = candidate_labels[top_k_idx[i]]
            if query_labels[i] in retrieved_labels:
                hits += 1
        results[f"Class-R@{k}"] = hits / len(query_embeds) * 100

    return results


def compute_map(
    query_embeds: torch.Tensor,
    candidate_embeds: torch.Tensor,
    ground_truth_indices: torch.Tensor,
) -> float:
    """
    Caption-level Mean Average Precision.
    Each query has exactly one relevant caption.
    """
    sim = query_embeds @ candidate_embeds.T
    # Sort by similarity descending
    sorted_indices = sim.argsort(dim=1, descending=True)

    aps = []
    for i in range(len(query_embeds)):
        gt = ground_truth_indices[i].item()
        rank = (sorted_indices[i] == gt).nonzero(as_tuple=True)[0].item()
        ap = 1.0 / (rank + 1)
        aps.append(ap)

    return np.mean(aps)


def compute_class_aware_map(
    query_embeds: torch.Tensor,
    candidate_embeds: torch.Tensor,
    query_labels: torch.Tensor,
    candidate_labels: torch.Tensor,
) -> float:
    """
    Class-aware MAP: any caption from the same class is relevant.
    """
    sim = query_embeds @ candidate_embeds.T
    sorted_indices = sim.argsort(dim=1, descending=True)

    aps = []
    for i in range(len(query_embeds)):
        relevant = (candidate_labels == query_labels[i])
        num_relevant = relevant.sum().item()
        if num_relevant == 0:
            continue

        precision_sum = 0.0
        hits = 0
        for rank, idx in enumerate(sorted_indices[i]):
            if relevant[idx]:
                hits += 1
                precision_sum += hits / (rank + 1)
        aps.append(precision_sum / num_relevant)

    return np.mean(aps)


def compute_clip_scores(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    ground_truth_indices: torch.Tensor,
) -> Dict[str, float]:
    """Compute CLIPScore for matched and mismatched pairs."""
    sim = image_embeds @ text_embeds.T

    # Matched pairs
    matched_scores = sim[torch.arange(len(sim)), ground_truth_indices].numpy()

    # Mismatched: sample random negatives
    n = len(sim)
    random_indices = torch.randint(0, text_embeds.size(0), (n,))
    # Ensure they differ from ground truth
    same = random_indices == ground_truth_indices
    random_indices[same] = (random_indices[same] + 1) % text_embeds.size(0)
    mismatched_scores = sim[torch.arange(n), random_indices].numpy()

    return {
        "matched_mean": float(matched_scores.mean()),
        "matched_std": float(matched_scores.std()),
        "mismatched_mean": float(mismatched_scores.mean()),
        "mismatched_std": float(mismatched_scores.std()),
    }


def run_full_retrieval_eval(
    query_embeds: torch.Tensor,
    candidate_embeds: torch.Tensor,
    ground_truth_indices: torch.Tensor,
    query_labels: torch.Tensor,
    candidate_labels: torch.Tensor,
    image_embeds: Optional[torch.Tensor] = None,
) -> Dict:
    """Run all retrieval metrics and return combined results."""
    results = {}

    # Recall@K
    r_at_k = recall_at_k(query_embeds, candidate_embeds, ground_truth_indices)
    results.update(r_at_k)

    # Class-aware Recall@K
    class_r = class_aware_recall_at_k(
        query_embeds, candidate_embeds, query_labels, candidate_labels)
    results.update(class_r)

    # MAP
    results["MAP"] = compute_map(query_embeds, candidate_embeds, ground_truth_indices) * 100
    results["Class-MAP"] = compute_class_aware_map(
        query_embeds, candidate_embeds, query_labels, candidate_labels) * 100

    # CLIPScore (if image embeds provided)
    if image_embeds is not None:
        clip_scores = compute_clip_scores(image_embeds, candidate_embeds, ground_truth_indices)
        results["CLIPScore"] = clip_scores

    print("\n" + "=" * 50)
    print("Retrieval Results")
    print("=" * 50)
    for k, v in results.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv:.4f}")
        else:
            print(f"  {k}: {v:.2f}%")

    return results
