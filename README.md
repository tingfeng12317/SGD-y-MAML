# SGD-y-MAML

## 1. Overview
We propose SGD-y, an optimizer based on Stochastic Gradient Descent (SGD). It reduces gradient noise in the inner loop of Model-Agnostic Meta-Learning (MAML) and dynamically adjusts the inner-loop steps. We introduce the SGD-y-MAML framework based on it. The framework retains SGD’s stability and adds two mechanisms, a dual convergence mechanism and an inter-layer adaptive learning rate. The former gets adaptive inner-loop steps by gradient residual detection and parameter drift verification. The latter balances the updates between shallow textural features and deep semantic features. Both mechanisms reduce gradient noise and remove the need for validation-based early stopping. Snapshot rollback conflicts with second-order derivatives during calculation, so we keep the snapshot mechanism in SGD-y-FOMAML and remove it from SGD-y-MAML. We still report accuracy on the validation set for direct comparison with baselines. On Omniglot (5-way 1-shot), SGD-y-MAML reaches 2.52% higher test accuracy than MAML, and SGD-y-FOMAML exceeds First-Order MAML (FOMAML) by 5.6%. On CIFAR-FS (5-way 5-shot), the absence of snapshot rollback leads to slight overfitting. There is a 2.84% gain in validation accuracy and a 2.71% drop in test accuracy relative to MAML. This finding reveals that adaptive step counts will drive the meta-initialization away from the cross-task common intersection in the absence of overfitting safeguards.

## 2. Environment
[Python/PyTorch 版本 + 安装命令]

## 3. Project Structure
[目录树说明]

## 4. Datasets
[下载链接 + 放置路径]

## 5. Installation
[clone + pip install]

## 6. Usage
### 6.1 Training
[训练命令]
### 6.2 Ablation Study
[消融实验命令]
### 6.3 Visualization
[绘图命令]

## 7. Key Features
[算法特点，3-4条]

## 8. Results
[可选：主要实验结论]

## 9. Citation
[论文发表后补 BibTeX]
