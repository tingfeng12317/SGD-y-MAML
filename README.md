# SGD-y-MAML

## 1. Overview
We propose SGD-y, an optimizer based on Stochastic Gradient Descent (SGD). It reduces gradient noise in the inner loop of Model-Agnostic Meta-Learning (MAML) and dynamically adjusts the inner-loop steps. We introduce the SGD-y-MAML framework based on it. The framework retains SGD’s stability and adds two mechanisms, a dual convergence mechanism and an inter-layer adaptive learning rate. The former gets adaptive inner-loop steps by gradient residual detection and parameter drift verification. The latter balances the updates between shallow textural features and deep semantic features. Both mechanisms reduce gradient noise and remove the need for validation-based early stopping. Snapshot rollback conflicts with second-order derivatives during calculation, so we keep the snapshot mechanism in SGD-y-FOMAML and remove it from SGD-y-MAML. We still report accuracy on the validation set for direct comparison with baselines. On Omniglot (5-way 1-shot), SGD-y-MAML reaches 2.52% higher test accuracy than MAML, and SGD-y-FOMAML exceeds First-Order MAML (FOMAML) by 5.6%. On CIFAR-FS (5-way 5-shot), the absence of snapshot rollback leads to slight overfitting. There is a 2.84% gain in validation accuracy and a 2.71% drop in test accuracy relative to MAML. This finding reveals that adaptive step counts will drive the meta-initialization away from the cross-task common intersection in the absence of overfitting safeguards.

## 2. Environment

- **OS**: Windows / Linux
- **Python**: 3.10+
- **PyTorch**: 2.10.0 (CUDA 12.8)
- **GPU**: NVIDIA GPU with CUDA support

```bash
# PyTorch with CUDA 12.8
pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# PyTorch CPU-only
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
```
## 3. Project Structure
```text
SGD-y-MAML/
├── algorithms/              # Core meta-learning algorithms
│   ├── __init__.py
│   ├── maml.py
│   ├── sgd_y_fomaml.py
│   ├── sgd_y_maml.py
│   └── taming_maml.py
├── data/                    # Dataset preprocessing
│   ├── cifar_fs_dataset.py
│   └── omniglot_dataset.py
├── figures/                 # Visualization scripts & experimental data
│   ├── result/              # Ablation and comparison logs
│   ├── update_tracking/     # Gradient tracking data
│   ├── ablation_image.py    # Ablation visualization
│   ├── compare_image.py     # 4-algorithm comparison plots
│   ├── update_2.py          # Update magnitude comparison
│   └── update_image.py      # Gradient tracking visualization
├── models/                  # CNN architectures
│   ├── __init__.py
│   ├── cifar_fs_net.py
│   └── omniglot_net.py
├── ablation_analysis.py     # Ablation study analysis
├── ablation_config.py       # Ablation configuration
├── ablation_runner.py       # Ablation experiment runner
├── cifar_fs_train.py        # CIFAR-FS training script
├── config.py                # Global configuration
├── omniglot_train.py        # Omniglot training script
├── requirements.txt         # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```
## 4. Datasets

Download and place the following datasets in the root directory:

- Omniglot: https://github.com/brendenlake/omniglot
  Place in: omniglot_data/

- CIFAR-FS: https://www.kaggle.com/datasets/keywhere/cifarfs?resource=download
  Place in: CIFAR-FS/

Expected structure:

```text
omniglot_data/
├── images_background/
└── images_evaluation/

CIFAR-FS/
├── cifar_fs_train.pickle
├── cifar_fs_val.pickle
└── cifar_fs_test.pickle
```

## 5. Installation

```bash
git clone https://github.com/tingfeng12317/SGD-y-MAML.git
cd SGD-y-MAML
pip install -r requirements.txt
```

## 6. Usage
### 6.1 Training
```bash
# Omniglot 5-way 1-shot (4-algorithm comparison for compare_image.py)
python omniglot_train.py --n_way 5 --k_shot 1 --algorithms maml sgd_y_maml maml_fo sgd_y_maml_fo taming_maml --seeds 42 123 456 789 1024 --episodes 1000 --save_dir ./figures/result/full_comparison

# Omniglot 5-way 1-shot (MAML 10 steps vs SGD-y)
python omniglot_train.py --n_way 5 --k_shot 1 --algorithms maml sgd_y_maml --inner_steps 10 --seeds 42 123 456 789 1024 --episodes 1000 --save_dir ./figures/result/5way1shot_10steps

# Omniglot 5-way 5-shot
python omniglot_train.py --n_way 5 --k_shot 5 --algorithms maml sgd_y_maml --seeds 42 123 456 789 1024 --episodes 1000 --save_dir ./figures/result/5way5shot

# CIFAR-FS 5-way 1-shot
python cifar_fs_train.py --algorithms maml sgd_y_maml --n_way 5 --k_shot 1 --episodes 60000 --seeds 42 --save_dir ./figures/result/cifar_fs_5w1s

# CIFAR-FS 5-way 5-shot
python cifar_fs_train.py --algorithms maml sgd_y_maml --n_way 5 --k_shot 5 --episodes 20000 --seeds 42 --save_dir ./figures/result/cifar_fs_5w5s
```
### 6.2 Ablation Study
```bash
# Run ablation (ensure output path is ./figures/result/ablation_b3_drift_0.015/detailed_logs)
python ablation_runner.py --experiment B
```
### 6.3 Data Tracking
```bash
# Smax adaptive steps
python omniglot_train.py --algorithms sgd_y_maml --track_updates --track_dir ./figures/update_tracking/Smax --save_dir ./figures/update_tracking/Smax --seeds 42

# Fixed 5-step
python omniglot_train.py --algorithms sgd_y_maml --sgdy_fixed_steps 5 --track_updates --track_dir ./figures/update_tracking/fixed5_vs_adaptive --save_dir ./figures/update_tracking/fixed5_vs_adaptive --seeds 42
```
### 6.4 Plotting

```bash
python compare_image.py      # Figure 3: 4-algorithm comparison (2 subfigures)
python ablation_image.py     # Figure 4: Ablation study (4 subfigures)
python update_image.py       # Figure 5 & 6: Gradient trajectory + Layer-wise convergence
python update_2.py           # Figure 7: Fixed vs adaptive update magnitudes (2 subfigures)
```
## 7. Key Features

- **SGD-y optimizer** — An SGD-based adaptive inner-loop optimizer that reduces gradient noise and dynamically adjusts inner-loop steps for MAML.
- **Dual convergence mechanism** — Employs gradient residual detection and parameter drift verification to enable adaptive step counts without validation-set dependency.
- **Inter-layer adaptive learning rate** — Balances parameter updates between shallow textural features and deep semantic features.
- **Empirical validation** — Achieves 2.52% higher test accuracy than MAML on Omniglot (5-way 1-shot); experiments on CIFAR-FS (5-way 5-shot) further reveal the overfitting risk without overfitting safeguards.
