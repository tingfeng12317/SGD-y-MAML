# config.py
import torch
import numpy as np
import random
import os


class Config:
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========== 任务配置 ==========
    N_WAY = 5
    K_SHOT = 1
    K_QUERY = 15

    # ========== MAML 配置（对应 B1 Baseline）==========
    INNER_LR = 0.01
    INNER_STEPS = 5

    # ========== SGD-Y-MAML 配置（对应 B5 Full）==========
    SGDY_INNER_LR = 0.01
    SGDY_INNER_STEPS_MAX = 20
    SGDY_MOMENTUM = 0.9
    SGDY_EPS_MIN = 0.03
    SGDY_DECAY_RATE = 0.9
    SGDY_MIN_LR = 1e-4
    SGDY_K = 3
    SGDY_NOISE_FLOOR_THRESHOLD = 0.6
    SGDY_DUAL_CONVERGENCE = True
    SGDY_ADJACENT_DRIFT_THRESHOLD = 0.015
    SGDY_STATISTICAL_WEIGHT = 0.01
    SGDY_GRADIENT_SMOOTH_WEIGHT = 0.001

    # ========== TamingMAML 配置（自适应正则化MAML）==========
    TAMING_ALPHA_REG = 1e-3  # 内循环L2正则化系数
    TAMING_GRAD_CLIP = 1.0  # 内循环梯度裁剪阈值
    TAMING_LR_DECAY = 0.9  # 自适应学习率衰减率
    TAMING_META_WEIGHT_DECAY = 1e-4  # AdamW权重衰减

    # ========== ES-MAML 配置（进化策略MAML）==========
    # 核心参数
    ES_SIGMA = 0.15  # 噪声标准差σ（探索强度）
    ES_N_PERTURBATIONS = 128  # 种群大小（扰动数量）
    ES_ANTITHETIC = True  # 是否使用对偶采样（减少方差）

    # 元优化器配置
    ES_META_LR = 0.001  # 元学习率（外循环）
    ES_MOMENTUM = 0.9  # SGD动量（仅ES_USE_ADAM=False时有效）
    ES_USE_ADAM = False  # 是否使用Adam（False则使用SGD）
    ES_GRAD_CLIP = 10.0  # 元梯度裁剪阈值

    # ES专用内循环配置（可与标准MAML不同）
    ES_INNER_LR = 0.03  # ES内循环学习率（通常比MAML大）
    ES_INNER_STEPS = 1  # ES内循环步数（通常比MAML少）

    # 可选：自适应噪声衰减（进阶功能）
    ES_ADAPTIVE_NOISE = False  # 是否启用自适应噪声衰减
    ES_NOISE_DECAY_RATE = 0.995  # 噪声衰减率
    ES_MIN_SIGMA = 0.01  # 最小噪声水平

    # ========== 外循环配置 ==========
    META_LR = 0.001
    META_BATCH_SIZE = 4

    # ========== 训练配置 ==========
    MAX_EPISODES = 1000
    EVAL_INTERVAL = 10

    # ========== 数据集路径 ==========
    DATA_ROOT = "./omniglot_data"

    # ========== 消融实验相关 ==========
    USE_PURE_MAML = False
    USE_LR_DECAY = True
    USE_REGULARIZATION = True
    USE_DUAL_CONVERGENCE = True

    @classmethod
    def set_seed(cls, seed=None):
        """设置随机种子，确保可复现性"""
        if seed is not None:
            cls.SEED = seed

        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        os.environ['PYTHONHASHSEED'] = str(cls.SEED)

        print(f"✅ 随机种子已设置: {cls.SEED}")
        print(f"✅ 使用设备: {cls.DEVICE}")
        if torch.cuda.is_available():
            print(
                f"   确定性模式: deterministic={torch.backends.cudnn.deterministic}, benchmark={torch.backends.cudnn.benchmark}")

        return cls.SEED

    @classmethod
    def get_maml_config(cls):
        """获取MAML配置"""
        return {
            'algorithm': 'maml',
            'inner_steps': cls.INNER_STEPS,
            'inner_lr': cls.INNER_LR,
            'first_order': False,
            'description': 'MAML: 固定步数，无正则化'
        }

    @classmethod
    def get_taming_config(cls):
        """获取TamingMAML配置"""
        return {
            'algorithm': 'taming_maml',
            'inner_steps': cls.INNER_STEPS,
            'inner_lr': cls.INNER_LR,
            'alpha_reg': cls.TAMING_ALPHA_REG,
            'grad_clip': cls.TAMING_GRAD_CLIP,
            'lr_decay': cls.TAMING_LR_DECAY,
            'meta_weight_decay': cls.TAMING_META_WEIGHT_DECAY,
            'description': 'TamingMAML: 自适应正则化，梯度裁剪，学习率衰减'
        }

    @classmethod
    def get_es_config(cls):
        """获取ES-MAML配置"""
        return {
            'algorithm': 'es_maml',
            'inner_steps': cls.ES_INNER_STEPS,
            'inner_lr': cls.ES_INNER_LR,
            'sigma': cls.ES_SIGMA,
            'n_perturbations': cls.ES_N_PERTURBATIONS,
            'antithetic': cls.ES_ANTITHETIC,
            'meta_lr': cls.ES_META_LR,
            'use_adam': cls.ES_USE_ADAM,
            'momentum': cls.ES_MOMENTUM,
            'grad_clip': cls.ES_GRAD_CLIP,
            'description': 'ES-MAML: 进化策略，无梯度元优化，对偶采样'
        }

    @classmethod
    def get_sgdy_config(cls):
        """获取SGD-Y-MAML配置"""
        return {
            'algorithm': 'sgd_y_maml',
            'max_steps': cls.SGDY_INNER_STEPS_MAX,
            'initial_lr': cls.SGDY_INNER_LR,
            'momentum': cls.SGDY_MOMENTUM,
            'eps_min': cls.SGDY_EPS_MIN,
            'noise_floor_threshold': cls.SGDY_NOISE_FLOOR_THRESHOLD,
            'adjacent_drift_threshold': cls.SGDY_ADJACENT_DRIFT_THRESHOLD,
            'use_lr_decay': cls.USE_LR_DECAY,
            'use_regularization': cls.USE_REGULARIZATION,
            'lambda_stat': cls.SGDY_STATISTICAL_WEIGHT,
            'lambda_smooth': cls.SGDY_GRADIENT_SMOOTH_WEIGHT,
            'description': 'SGD-Y-MAML: 自适应步数，双重收敛'
        }

    @classmethod
    def print_config(cls, config_type='es'):
        """打印当前配置"""
        print(f"\n{'=' * 50}")
        print(f"当前配置: {config_type.upper()}")
        print(f"{'=' * 50}")

        if config_type == 'es':
            cfg = cls.get_es_config()
            for k, v in cfg.items():
                print(f"{k:20s}: {v}")
        elif config_type == 'maml':
            cfg = cls.get_maml_config()
            for k, v in cfg.items():
                print(f"{k:20s}: {v}")
        # ... 其他类型

        print(f"{'=' * 50}\n")