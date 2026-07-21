import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders import ECG1DResNetEncoder, EHRGatedEncoder, CrossModalGatedFusion
from graph_gat import MemoryBankDynamicGAT

class UMGF_CVD(nn.Module):
    """
    UMGF-CVD: Uncertainty-Aware Multimodal Graph Fusion for Cardiovascular Disease Risk.
    Unifies 12-lead ECG waveforms, tabular EHR variables with missingness masks,
    dynamic population GAT memory bank, and a Subjective Logic Evidential Head.
    """
    def __init__(self, num_classes=5, embed_dim=256, k_neighbors=10, memory_bank_size=2000):
        super().__init__()
        self.K = num_classes
        self.ecg_enc = ECG1DResNetEncoder(embed_dim=embed_dim)
        self.ehr_enc = EHRGatedEncoder(embed_dim=embed_dim)
        self.fusion = CrossModalGatedFusion(embed_dim=embed_dim)
        self.gat = MemoryBankDynamicGAT(dim=embed_dim, k=k_neighbors, mem_size=memory_bank_size)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, ecg, ehr, mask):
        # 1. Encode individual modalities
        z_ecg = self.ecg_enc(ecg)
        z_ehr = self.ehr_enc(ehr, mask)
        
        # 2. Cross-modal gated projection
        z_fused = self.fusion(z_ecg, z_ehr)
        
        # 3. Dynamic population GAT propagation over memory bank
        h_gat = F.relu(self.gat(z_fused))
        
        # 4. Evidential Neural Head (Softplus activation for non-negative evidence e_k >= 0)
        evidence = F.softplus(self.head(h_gat))
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=-1, keepdim=True)
        
        # Returns expected probability vector p_k, vacuity uncertainty u_i, and concentration parameters alpha
        expected_probs = alpha / S
        vacuity_uncertainty = self.K / S.squeeze(-1)
        
        return expected_probs, vacuity_uncertainty, alpha
