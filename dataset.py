import numpy as np
import torch
from torch.utils.data import Dataset

class SyntheticCVDFusionDataset(Dataset):
    """
    Multimodal Cardiovascular Dataset generating paired 12-lead ECG signals,
    structured EHR features with missingness masks, and multi-class diagnostic labels.
    """
    def __init__(self, num_samples=1000, num_leads=12, seq_len=5000, num_ehr_feats=10, num_classes=5):
        super().__init__()
        time = np.linspace(0, 4 * np.pi, seq_len)
        ecg_data, ehr_data, mask_data, labels = [], [], [], []
        
        for _ in range(num_samples):
            # Synthetic 12-lead ECG waveform generation with phase offsets & noise
            freq = np.random.uniform(0.8, 1.2)
            noise = np.random.uniform(0.05, 0.20)
            leads = [
                np.sin(2 * np.pi * freq * time + l * np.pi / 6) + np.random.normal(0, noise, seq_len)
                for l in range(num_leads)
            ]
            ecg_data.append(np.array(leads))
            
            # Tabular EHR variables (Age, Troponin, BNP, etc.)
            ehr = np.random.randn(num_ehr_feats)
            ehr[0] = np.random.uniform(30, 85) # Age
            ehr[4] = np.random.exponential(1.5) # Biomarker concentration
            
            # Simulate 15% random missingness mask
            mask = (np.random.rand(num_ehr_feats) > 0.15).astype(np.float32)
            ehr = ehr * mask
            
            ehr_data.append(ehr)
            mask_data.append(mask)
            
            # Clinical diagnostic superclass rules
            if ehr[4] > 3.0:
                labels.append(1) # Myocardial Infarction
            elif ehr[4] > 1.5 and ehr[0] > 60:
                labels.append(2) # ST/T Changes
            elif ehr[0] > 70:
                labels.append(3) # Conduction Disturbance
            elif ehr[2] > 1.5:
                labels.append(4) # Hypertrophy
            else:
                labels.append(0) # Normal (NORM)
            
        self.ecg = torch.tensor(np.array(ecg_data), dtype=torch.float32)
        self.ehr = torch.tensor(np.array(ehr_data), dtype=torch.float32)
        self.mask = torch.tensor(np.array(mask_data), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ecg[idx], self.ehr[idx], self.mask[idx], self.labels[idx]
