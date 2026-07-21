import torch
import torch.nn as nn
import torch.nn.functional as F

class ECG1DResNetEncoder(nn.Module):
    """
    1D-ResNet34 Waveform Encoder extracting spatial-temporal representations from 12-lead ECG signals.
    """
    def __init__(self, in_leads=12, embed_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_leads, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.proj = nn.Linear(128, embed_dim)
        
    def forward(self, x):
        h = self.conv(x).squeeze(-1)
        return F.relu(self.proj(h))

class EHRGatedEncoder(nn.Module):
    """
    Gated-MLP tabular EHR encoder processing concatenated features and missingness indicator masks.
    """
    def __init__(self, in_feats=10, embed_dim=256):
        super().__init__()
        # Accepts [raw_features || missingness_mask]
        self.fc = nn.Linear(in_feats * 2, 128)
        self.gate = nn.Linear(128, embed_dim)
        self.val = nn.Linear(128, embed_dim)
        
    def forward(self, x, mask):
        h = F.relu(self.fc(torch.cat([x, mask], dim=-1)))
        return torch.sigmoid(self.gate(h)) * F.relu(self.val(h))

class CrossModalGatedFusion(nn.Module):
    """
    Cross-Modal Gated Projection Layer dynamically weighting ECG waveform and tabular EHR contributions.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.gate = nn.Linear(embed_dim * 2, embed_dim)
        self.proj_e = nn.Linear(embed_dim, embed_dim)
        self.proj_t = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, z_ecg, z_ehr):
        g = torch.sigmoid(self.gate(torch.cat([z_ecg, z_ehr], dim=-1)))
        return g * self.proj_e(z_ecg) + (1.0 - g) * self.proj_t(z_ehr)
