import os
from dataclasses import dataclass, field
from typing import List, Optional


# ─── Dataset Paths ───────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    # Root paths on PSC
    project_root: str = "/ocean/projects/cis250019p/gandotra/11785-gp-eeg"
    eeg_root: str = ""      # ds005589/ dir containing sub-XX folders
    images_dir: str = ""     # images/ dir with .jpg files
    captions_file: str = ""  # captions.txt

    # Subjects in the dataset
    subjects: List[str] = field(default_factory=lambda: [
        "sub-02", "sub-03", "sub-05", "sub-09", "sub-14", "sub-15",
        "sub-17", "sub-19", "sub-20", "sub-23", "sub-24", "sub-28", "sub-29"
    ])
    num_subjects: int = 13
    num_sessions: int = 5
    num_runs: int = 4
    trials_per_run: int = 100

    # EEG dimensions
    num_channels: int = 122
    num_timepoints: int = 500
    num_classes: int = 20

    # Session-based split (per subject): 3 train, 1 val, 1 test
    train_sessions: List[str] = field(default_factory=lambda: ["ses-01", "ses-02", "ses-03"])
    val_sessions: List[str] = field(default_factory=lambda: ["ses-04"])
    test_sessions: List[str] = field(default_factory=lambda: ["ses-05"])

    def __post_init__(self):
        self.eeg_root = os.path.join(self.project_root, "ds005589")
        self.images_dir = os.path.join(self.project_root, "images")
        self.captions_file = os.path.join(self.project_root, "captions.txt")


# ─── Model Hyperparameters ───────────────────────────────────────────────────
@dataclass
class BaselineMLPConfig:
    input_dim: int = 122 * 500
    hidden_dims: List[int] = field(default_factory=lambda: [1024, 512, 256])
    num_classes: int = 20
    dropout: float = 0.5


@dataclass
class CNNTransformerConfig:
    cnn_out_channels: int = 64
    cnn_kernel_size: int = 25
    cnn_stride: int = 1
    cnn_num_layers: int = 2
    embed_dim: int = 64
    num_heads: int = 8
    num_transformer_layers: int = 4
    feedforward_dim: int = 256
    transformer_dropout: float = 0.1
    head_hidden_dim: int = 128
    num_classes: int = 20
    num_subjects: int = 13
    dropout: float = 0.3


@dataclass
class TrainConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    num_epochs: int = 50
    scheduler: str = "cosine"
    patience: int = 10
    min_delta: float = 1e-4
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    save_dir: str = "./checkpoints"
    log_interval: int = 50
    seed: int = 42


@dataclass
class CLIPConfig:
    model_name: str = "openai/clip-vit-base-patch32"
    embed_dim: int = 512
    freeze_clip: bool = True
    projection_hidden_dim: int = 256
    projection_layers: int = 2
    temperature: float = 0.07


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    baseline: BaselineMLPConfig = field(default_factory=BaselineMLPConfig)
    cnn_transformer: CNNTransformerConfig = field(default_factory=CNNTransformerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    clip: CLIPConfig = field(default_factory=CLIPConfig)
