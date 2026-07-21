import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np

from dataset import SyntheticCVDFusionDataset
from model import UMGF_CVD
from loss import SubjectiveLogicLoss
from metrics import compute_ece, evaluate_selective_prediction

def main():
    print("=" * 75)
    print("  UMGF-CVD: Uncertainty-Aware Multimodal Graph Fusion Model Training  ")
    print("=" * 75)
    
    # 1. Device and Random Seed
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution Device: {device}")
    
    # 2. Dataset Preparation
    dataset = SyntheticCVDFusionDataset(num_samples=1200, num_leads=12, seq_len=5000, num_ehr_feats=10, num_classes=5)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    print(f"[*] Dataset Split: {train_size} Train Samples, {val_size} Validation Samples")
    
    # 3. Model, Loss, Optimizer Initialization
    model = UMGF_CVD(num_classes=5, embed_dim=256, k_neighbors=10, memory_bank_size=2000).to(device)
    criterion = SubjectiveLogicLoss(num_classes=5, warm_up_epochs=5.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    
    # 4. Model Training Loop
    epochs = 5
    print("\n[*] Starting End-to-End Optimization Loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for ecg_b, ehr_b, mask_b, y_b in train_loader:
            ecg_b = ecg_b.to(device)
            ehr_b = ehr_b.to(device)
            mask_b = mask_b.to(device)
            y_b = y_b.to(device)
            
            optimizer.zero_grad()
            probs, u, alpha = model(ecg_b, ehr_b, mask_b)
            
            y_onehot = F.one_hot(y_b, num_classes=5).float()
            loss = criterion(alpha, y_onehot, epoch)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # 5. Validation & Evaluation Protocol
    print("\n[*] Evaluating Validation Cohort Performance...")
    model.eval()
    all_probs, all_u, all_labels = [], [], []
    
    with torch.no_grad():
        for ecg_b, ehr_b, mask_b, y_b in val_loader:
            ecg_b = ecg_b.to(device)
            ehr_b = ehr_b.to(device)
            mask_b = mask_b.to(device)
            
            probs, u, alpha = model(ecg_b, ehr_b, mask_b)
            all_probs.append(probs.cpu().numpy())
            all_u.append(u.cpu().numpy())
            all_labels.append(y_b.numpy())
            
    all_probs = np.concatenate(all_probs, axis=0)
    all_u = np.concatenate(all_u, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    val_preds = np.argmax(all_probs, axis=1)
    val_acc = np.mean(val_preds == all_labels)
    val_ece = compute_ece(all_probs, all_labels)
    mean_vacuity = np.mean(all_u)
    
    print("-" * 55)
    print(f" Validation Accuracy    : {val_acc * 100:.2f}%")
    print(f" Expected Calibration ECE: {val_ece:.4f}")
    print(f" Mean Vacuity Uncertainty: {mean_vacuity:.4f}")
    print("-" * 55)
    
    # 6. Selective Prediction & Clinical Abstention Assessment
    print("\n[*] Selective Prediction Performance under Clinical Abstention:")
    sel_results = evaluate_selective_prediction(all_probs, all_u, all_labels)
    
    print(f"{'Rejection (%)':<15} {'Coverage (%)':<15} {'Accuracy (%)':<15} {'ECE':<10} {'Deferred Cases':<15}")
    print("-" * 70)
    for res in sel_results:
        print(f"{res['rejection_rate']:<15.1f} {res['coverage']:<15.1f} {res['accuracy']*100:<15.2f} {res['ece']:<10.4f} {res['deferred_count']:<15}")
    print("=" * 75)
    print("  UMGF-CVD Benchmark Execution Completed Successfully.  ")
    print("=" * 75)

if __name__ == "__main__":
    main()
