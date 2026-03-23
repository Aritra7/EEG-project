import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    def __init__(self, input_dim=122*500, hidden_dims=[1024, 512, 256],
                 num_classes=20, num_subjects=13, dropout=0.5):
        super().__init__()
        self.num_subjects = num_subjects
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        self.backbone = nn.Sequential(*layers)
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dims[-1], num_classes)
            for _ in range(num_subjects)
        ])

    def forward(self, eeg, subject_idx):
        B = eeg.size(0)
        x = eeg.view(B, -1)
        features = self.backbone(x)
        logits = torch.zeros(B, self.heads[0].out_features, device=eeg.device)
        for sid in range(self.num_subjects):
            mask = subject_idx == sid
            if mask.any():
                logits[mask] = self.heads[sid](features[mask])
        return logits

    def get_features(self, eeg):
        B = eeg.size(0)
        x = eeg.view(B, -1)
        return self.backbone(x)
