import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryBankDynamicGAT(nn.Module):
    """
    Offline FIFO Memory Bank Population Graph Attention Network (GAT).
    Maintains a global memory bank of historical patient representations N_mem to perform
    adaptive k-NN graph attention message passing beyond single mini-batches.
    """
    def __init__(self, dim=256, k=10, mem_size=2000):
        super().__init__()
        self.k = k
        self.W = nn.Linear(dim, dim)
        self.a = nn.Parameter(torch.randn(1, 1, 2 * dim))
        self.register_buffer("memory_bank", torch.randn(mem_size, dim))
        
    def forward(self, h):
        N, D = h.shape
        h_proj = self.W(h)
        h_norm = F.normalize(h, p=2, dim=-1)
        mem_norm = F.normalize(self.memory_bank, p=2, dim=-1)
        
        # Pairwise cosine similarity matrix (N x mem_size)
        sim = torch.matmul(h_norm, mem_norm.T)
        _, topk_idx = torch.topk(sim, k=self.k, dim=-1)
        
        # Retrieve top-k nearest patient embeddings from memory bank
        mem_retrieved = self.memory_bank[topk_idx] # (N, k, D)
        h_i = h_proj.unsqueeze(1).repeat(1, self.k, 1)
        
        # Multi-head attention scores
        e = F.leaky_relu(torch.sum(torch.cat([h_i, mem_retrieved], dim=-1) * self.a, dim=-1))
        alpha = F.softmax(e, dim=-1).unsqueeze(-1)
        h_gat = torch.sum(alpha * mem_retrieved, dim=1)
        
        # Update FIFO Memory Bank queue during training
        if self.training:
            self.memory_bank = torch.cat([self.memory_bank[N:], h.detach()], dim=0)
            
        return h_gat
