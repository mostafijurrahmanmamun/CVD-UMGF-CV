import torch
import torch.nn as nn

class SubjectiveLogicLoss(nn.Module):
    """
    Subjective Logic Evidential Neural Loss.
    Combines Expected Cross-Entropy over Dirichlet density distributions with an
    annealed Kullback-Leibler (KL) divergence penalty to prevent uninformative evidence assignment.
    """
    def __init__(self, num_classes=5, warm_up_epochs=5.0):
        super().__init__()
        self.K = num_classes
        self.warm_up_epochs = warm_up_epochs
        
    def forward(self, alpha, y_onehot, epoch):
        # Dirichlet Total Strength S = sum(alpha)
        S = torch.sum(alpha, dim=-1, keepdim=True)
        
        # Expected Cross-Entropy via Digamma function psi(x)
        A = torch.sum(y_onehot * (torch.digamma(S) - torch.digamma(alpha)), dim=-1, keepdim=True)
        
        # KL Divergence Regularization toward uniform Dirichlet prior D(p | 1)
        alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha
        sum_a = torch.sum(alpha_tilde, dim=-1, keepdim=True)
        kl = (
            torch.lgamma(sum_a)
            - torch.sum(torch.lgamma(alpha_tilde), dim=-1, keepdim=True)
            + torch.sum((alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(sum_a)), dim=-1, keepdim=True)
        )
        
        annealing_coef = min(1.0, float(epoch) / self.warm_up_epochs)
        return torch.mean(A + annealing_coef * kl)
