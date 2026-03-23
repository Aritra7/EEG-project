import argparse
import os
import sys
import torch
import numpy as np
import random

sys.path.insert(0, os.path.dirname(__file__))

from configs.config import Config
from utils.dataset import get_dataloaders, SUBJECT_IDS
from utils.train import train, evaluate
from utils.evaluation import (
    compute_confusion_matrix, print_confusion_analysis,
    print_per_subject_accuracy, generate_kaggle_submission,
    plot_confusion_matrix_text,
)
from models.baseline_mlp import BaselineMLP
from models.cnn_transformer import EEGCNNTransformer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(model_type, config):
    if model_type == "baseline":
        model = BaselineMLP(
            input_dim=config.data.num_channels * config.data.num_timepoints,
            hidden_dims=config.baseline.hidden_dims,
            num_classes=config.data.num_classes,
            num_subjects=config.data.num_subjects,
            dropout=config.baseline.dropout,
        )
    elif model_type == "cnn_transformer":
        model = EEGCNNTransformer(
            cnn_embed_dim=config.cnn_transformer.embed_dim,
            cnn_num_layers=config.cnn_transformer.cnn_num_layers,
            cnn_kernel_size=config.cnn_transformer.cnn_kernel_size,
            num_heads=config.cnn_transformer.num_heads,
            num_transformer_layers=config.cnn_transformer.num_transformer_layers,
            feedforward_dim=config.cnn_transformer.feedforward_dim,
            transformer_dropout=config.cnn_transformer.transformer_dropout,
            head_hidden_dim=config.cnn_transformer.head_hidden_dim,
            output_dim=config.data.num_classes,
            num_subjects=config.data.num_subjects,
            head_dropout=config.cnn_transformer.dropout,
            num_channels=config.data.num_channels,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    total = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {model_type} | Parameters: {total:,}")
    if hasattr(model, "count_parameters"):
        model.count_parameters()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline",
                        choices=["baseline", "cnn_transformer"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--kaggle_submission", action="store_true")
    args = parser.parse_args()

    config = Config()
    set_seed(args.seed)

    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        eeg_root=config.data.eeg_root,
        captions_file=config.data.captions_file,
        subjects=config.data.subjects,
        train_sessions=config.data.train_sessions,
        val_sessions=config.data.val_sessions,
        test_sessions=config.data.test_sessions,
        batch_size=args.batch_size,
        num_workers=config.train.num_workers,
    )

    model = build_model(args.model, config)

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=config.train.weight_decay,
        device=args.device,
        save_dir=args.save_dir,
        patience=config.train.patience,
        scheduler_type=config.train.scheduler,
        model_name=args.model,
    )

    print("\n" + "=" * 60)
    print("Final Test Evaluation")
    print("=" * 60)

    ckpt_path = os.path.join(args.save_dir, f"{args.model}_best.pt")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")

    criterion = torch.nn.CrossEntropyLoss()
    test_metrics = evaluate(model, test_loader, criterion, args.device)

    print(f"\nTest Accuracy: {test_metrics['accuracy']:.2f}%")
    print_per_subject_accuracy(test_metrics["per_subject_acc"])

    cm = compute_confusion_matrix(test_metrics["all_preds"], test_metrics["all_labels"])
    plot_confusion_matrix_text(cm)
    print_confusion_analysis(cm)

    if args.kaggle_submission:
        generate_kaggle_submission(
            model, test_loader, args.device,
            output_path=os.path.join(args.save_dir, f"{args.model}_submission.csv"))

    print("\nDone!")


if __name__ == "__main__":
    main()
