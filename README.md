# UMGF-CVD: Uncertainty-Aware Multimodal Graph Fusion Architecture

Production-grade PyTorch implementation of **UMGF-CVD** for cardiovascular disease risk prediction, dynamic population graph attention reasoning, and evidential uncertainty quantification.

## 📁 Repository Structure
```
code/
├── dataset.py                          # Multimodal 12-lead ECG & EHR dataset loader with missingness masks
├── encoders.py                         # 1D-ResNet34 ECG encoder, Gated-MLP EHR encoder, Cross-Modal Gated Fusion
├── graph_gat.py                        # Offline FIFO Memory Bank Population Graph Attention Network (N_mem=2,000)
├── loss.py                             # Subjective Logic Evidential Neural Loss with annealed KL penalty
├── metrics.py                          # Expected Calibration Error (ECE) & Selective Prediction evaluation
├── model.py                            # Master UMGF-CVD architecture unifying all components
├── train.py                            # Executable training, evaluation, and calibration benchmark pipeline
├── plot_calibration.py                 # Calibration reliability diagram generator script
├── generate_calibration_diagram.ipynb  # Interactive calibration diagram notebook
├── requirements.txt                    # Python package dependencies
├── environment.yml                     # Conda environment definition
├── Dockerfile                          # Production GPU CUDA container specification
└── Singularity.def                     # HPC Singularity/Apptainer container recipe
```

## 🚀 Execution & Benchmarking

### 1. Install Dependencies & Setup Environment
```bash
# Option A: via Pip
pip install -r requirements.txt

# Option B: via Conda
conda env create -f environment.yml
conda activate umgf_cvd

# Option C: via Docker
docker build -t umgf-cvd:latest .
docker run --gpus all umgf-cvd:latest
```

### 2. Run End-to-End Benchmark Training
```bash
python train.py
```

## 💡 Key Architectural Features
1. **Multimodal Gated Fusion:** Jointly integrates continuous 12-lead ECG waveforms ($500\text{ Hz}$) with tabular EHR variables ($[x_{\text{EHR}} \parallel m_{\text{EHR}}]$) using dynamic cross-modal gating.
2. **Offline Memory Bank GAT:** Maintains a FIFO queue of $N_{\text{mem}}=2,000$ historical patient representations to perform population graph reasoning beyond mini-batch limits.
3. **Evidential Uncertainty Head:** Replaces Softmax with Subjective Logic Dirichlet concentration parameters ($\alpha_k = e_k + 1$), providing expected probabilities ($\hat{p}_k$) and explicit vacuity uncertainty ($u_i$) in a single deterministic pass.
