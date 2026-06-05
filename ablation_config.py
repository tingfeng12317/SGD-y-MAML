# ablation_config.py
"""
消融实验配置系统 - 最终版 (B1-B5)
递进关系：Baseline → Momentum → 双重收敛(无衰减) → +LR衰减 → Full(加正则)
"""

import copy
import random
import numpy as np
import torch
from config import Config


class AblationConfig(Config):
    """
    消融实验专用配置类 - 新B组设计
    """

    # ========== A组：内部参数敏感性分析 ==========
    ABLATION_SEEDS = [42, 123, 456]
    A1_STEPS_LIST = [20, 30, 40, 60]  # S_max
    A2_K_LIST = [3, 5, 10]  # 衰减触发步数
    A3_EPS_LIST = [0.02, 0.03, 0.05]  # 噪声检测阈值
    A4_STAT_WEIGHTS = [0, 0.001, 0.01]  # 统计正则权重
    A4_SMOOTH_WEIGHTS = [0, 0.0001, 0.001]  # 平滑正则权重

    # ========== B组配置 ==========
    B1_INNER_STEPS = 5  # B1固定步数
    B_SMAX = 20  # B3-B5统一最大步数

    # B5正则权重
    B5_LAMBDA_STAT = 0.01
    B5_LAMBDA_SMOOTH = 0.001

    # ========== 实验控制 ==========
    ABLATION_EPISODES = 1000
    ABLATION_EVAL_INTERVAL = 10  # 每10个ep记录
    ABLATION_SAVE_DIR = "./results/ablation"

    @classmethod
    def get_ablation_config(cls, experiment: str, variant_idx: int, seed: int = 42):
        """获取实验配置"""
        config = cls._create_copy()
        config.SEED = seed

        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        config.MAX_EPISODES = cls.ABLATION_EPISODES
        config.EVAL_INTERVAL = cls.ABLATION_EVAL_INTERVAL

        if experiment.startswith('A'):
            config = cls._setup_A_group(config, experiment, variant_idx)
        elif experiment.startswith('B'):
            config = cls._setup_B_group(config, experiment)
        else:
            raise ValueError(f"未知实验: {experiment}，可选A1-A4, B1-B5")

        return config

    @classmethod
    def _setup_A_group(cls, config, experiment: str, variant_idx: int):
        """A组：参数敏感性分析"""
        if experiment == 'A1':
            if variant_idx >= len(cls.A1_STEPS_LIST):
                raise ValueError(f"A1变体索引{variant_idx}越界")
            config.SGDY_INNER_STEPS_MAX = cls.A1_STEPS_LIST[variant_idx]
            config.EXPERIMENT_NAME = f"A1_steps_{config.SGDY_INNER_STEPS_MAX}"
            config.EXPERIMENT_DESC = f"最大内循环步数={config.SGDY_INNER_STEPS_MAX}"

        elif experiment == 'A2':
            if variant_idx >= len(cls.A2_K_LIST):
                raise ValueError(f"A2变体索引{variant_idx}越界")
            config.SGDY_K = cls.A2_K_LIST[variant_idx]
            config.EXPERIMENT_NAME = f"A2_K_{config.SGDY_K}"
            config.EXPERIMENT_DESC = f"触发衰减步数K={config.SGDY_K}"

        elif experiment == 'A3':
            if variant_idx >= len(cls.A3_EPS_LIST):
                raise ValueError(f"A3变体索引{variant_idx}越界")
            config.SGDY_EPS_MIN = cls.A3_EPS_LIST[variant_idx]
            config.EXPERIMENT_NAME = f"A3_eps_{config.SGDY_EPS_MIN}"
            config.EXPERIMENT_DESC = f"噪声地板阈值={config.SGDY_EPS_MIN}"

        elif experiment == 'A4':
            if variant_idx >= len(cls.A4_STAT_WEIGHTS):
                raise ValueError(f"A4变体索引{variant_idx}越界")
            config.SGDY_STATISTICAL_WEIGHT = cls.A4_STAT_WEIGHTS[variant_idx]
            config.SGDY_GRADIENT_SMOOTH_WEIGHT = cls.A4_SMOOTH_WEIGHTS[variant_idx]
            config.EXPERIMENT_NAME = f"A4_stat{config.SGDY_STATISTICAL_WEIGHT}_smooth{config.SGDY_GRADIENT_SMOOTH_WEIGHT}"
            config.EXPERIMENT_DESC = f"统计权重={config.SGDY_STATISTICAL_WEIGHT}, 平滑权重={config.SGDY_GRADIENT_SMOOTH_WEIGHT}"

        else:
            raise ValueError(f"未知A组实验: {experiment}")

        # A组通用设置（完整SGD-Y）
        config.EXPERIMENT_GROUP = "A"
        config.VARIANT_IDX = variant_idx
        config.USE_PURE_MAML = False
        config.EXPERIMENT_TYPE = "internal_param"

        # A组使用完整配置作为基准
        config.SGDY_INNER_STEPS_MAX = getattr(config, 'SGDY_INNER_STEPS_MAX', cls.B_SMAX)
        config.SGDY_MOMENTUM = 0.9
        config.SGDY_K = getattr(config, 'SGDY_K', 3)
        config.SGDY_EPS_MIN = getattr(config, 'SGDY_EPS_MIN', 0.03)
        config.SGDY_DECAY_RATE = 0.9
        config.SGDY_MIN_LR = 1e-4
        config.SGDY_NOISE_FLOOR_THRESHOLD = 0.6
        config.SGDY_ADJACENT_DRIFT_THRESHOLD = 0.015
        config.USE_LR_DECAY = True  # A组默认启用
        config.USE_REGULARIZATION = True

        return config

    @classmethod
    def _setup_B_group(cls, config, experiment: str):
        """
        B组：逐步叠加组件（B1→B5）
        B1: Baseline (纯MAML，固定5步)
        B2: +Momentum (5步，无衰减)
        B3: +双重收敛 (S_max=20，无衰减，启用早停)
        B4: +LR衰减 (S_max=20，启用衰减，启用早停)
        B5: +正则化 (完整系统)
        """

        if experiment == 'B1':
            # B1: 纯MAML基线
            config.USE_PURE_MAML = True
            config.INNER_STEPS = cls.B1_INNER_STEPS
            config.INNER_LR = 0.01

            # 禁用所有SGD-Y特性
            config.SGDY_INNER_STEPS_MAX = cls.B1_INNER_STEPS
            config.SGDY_MOMENTUM = 0.0
            config.SGDY_K = 1000  # 无效值
            config.SGDY_EPS_MIN = 1.0  # 永不触发
            config.SGDY_NOISE_FLOOR_THRESHOLD = 1.0  # 永不触发
            config.SGDY_ADJACENT_DRIFT_THRESHOLD = 1e-8  # 永不触发
            config.USE_LR_DECAY = False
            config.USE_REGULARIZATION = False
            config.SGDY_STATISTICAL_WEIGHT = 0.0
            config.SGDY_GRADIENT_SMOOTH_WEIGHT = 0.0

            config.EXPERIMENT_NAME = "B1_Baseline"
            config.EXPERIMENT_DESC = "Baseline（固定5步，无momentum，固定lr，无早停）"

        elif experiment == 'B2':
            # B2: +Momentum
            config.USE_PURE_MAML = False
            config.SGDY_INNER_STEPS_MAX = cls.B1_INNER_STEPS  # 仍固定5步
            config.SGDY_MOMENTUM = 0.9
            config.SGDY_K = 1000  # 大值，但衰减关闭
            config.SGDY_EPS_MIN = 0.03
            config.SGDY_DECAY_RATE = 0.9
            config.SGDY_MIN_LR = 1e-4
            config.SGDY_NOISE_FLOOR_THRESHOLD = 1.0  # 禁用早停
            config.SGDY_ADJACENT_DRIFT_THRESHOLD = 1e-8  # 禁用早停
            config.USE_LR_DECAY = False  # 关键：禁用衰减
            config.USE_REGULARIZATION = False
            config.SGDY_STATISTICAL_WEIGHT = 0.0
            config.SGDY_GRADIENT_SMOOTH_WEIGHT = 0.0

            config.EXPERIMENT_NAME = "B2_Momentum"
            config.EXPERIMENT_DESC = "B1 + Momentum（固定5步，momentum=0.9，无衰减）"

        elif experiment == 'B3':
            # B3: +双重收敛（无LR衰减）
            config.USE_PURE_MAML = False
            config.SGDY_INNER_STEPS_MAX = cls.B_SMAX  # 放开到20
            config.SGDY_MOMENTUM = 0.9
            config.SGDY_K = 3  # 保留参数
            config.SGDY_EPS_MIN = 0.03
            config.SGDY_DECAY_RATE = 0.9
            config.SGDY_MIN_LR = 1e-4
            config.SGDY_NOISE_FLOOR_THRESHOLD = 0.6  # 启用第一重收敛
            config.SGDY_ADJACENT_DRIFT_THRESHOLD = 0.015  # 启用第二重收敛
            config.USE_LR_DECAY = False  # 关键：禁用衰减（与B4对比）
            config.USE_REGULARIZATION = False
            config.SGDY_STATISTICAL_WEIGHT = 0.0
            config.SGDY_GRADIENT_SMOOTH_WEIGHT = 0.0

            config.EXPERIMENT_NAME = "B3_Dual_Convergence"
            config.EXPERIMENT_DESC = f"B2 + 双重收敛（S_max={cls.B_SMAX}，无LR衰减，启用早停）"

        elif experiment == 'B4':
            # B4: +LR衰减
            config.USE_PURE_MAML = False
            config.SGDY_INNER_STEPS_MAX = cls.B_SMAX
            config.SGDY_MOMENTUM = 0.9
            config.SGDY_K = 3
            config.SGDY_EPS_MIN = 0.03
            config.SGDY_DECAY_RATE = 0.9
            config.SGDY_MIN_LR = 1e-4
            config.SGDY_NOISE_FLOOR_THRESHOLD = 0.6
            config.SGDY_ADJACENT_DRIFT_THRESHOLD = 0.015
            config.USE_LR_DECAY = True  # 关键：启用衰减（与B3对比）
            config.USE_REGULARIZATION = False
            config.SGDY_STATISTICAL_WEIGHT = 0.0
            config.SGDY_GRADIENT_SMOOTH_WEIGHT = 0.0

            config.EXPERIMENT_NAME = "B4_LR_Decay"
            config.EXPERIMENT_DESC = f"B3 + LR衰减（S_max={cls.B_SMAX}，K=3，启用衰减）"

        elif experiment == 'B5':
            # B5: Full（+正则化）
            config.USE_PURE_MAML = False
            config.SGDY_INNER_STEPS_MAX = cls.B_SMAX
            config.SGDY_MOMENTUM = 0.9
            config.SGDY_K = 3
            config.SGDY_EPS_MIN = 0.03
            config.SGDY_DECAY_RATE = 0.9
            config.SGDY_MIN_LR = 1e-4
            config.SGDY_NOISE_FLOOR_THRESHOLD = 0.6
            config.SGDY_ADJACENT_DRIFT_THRESHOLD = 0.015
            config.USE_LR_DECAY = True
            config.USE_REGULARIZATION = True  # 启用正则
            config.SGDY_STATISTICAL_WEIGHT = cls.B5_LAMBDA_STAT
            config.SGDY_GRADIENT_SMOOTH_WEIGHT = cls.B5_LAMBDA_SMOOTH

            config.EXPERIMENT_NAME = "B5_Full"
            config.EXPERIMENT_DESC = f"B4 + 正则化（完整系统，λ_stat={cls.B5_LAMBDA_STAT}, λ_smooth={cls.B5_LAMBDA_SMOOTH}）"

        else:
            raise ValueError(f"未知B组实验: {experiment}，支持B1-B5")

        config.EXPERIMENT_GROUP = "B"
        config.VARIANT_IDX = 0
        config.EXPERIMENT_TYPE = "component"
        return config

    @classmethod
    def _create_copy(cls):
        """深拷贝配置"""
        config = type('AblationConfigInstance', (), {})()

        for attr in dir(Config):
            if not attr.startswith('_'):
                try:
                    value = getattr(Config, attr)
                    if not callable(value):
                        setattr(config, attr, copy.deepcopy(value))
                except:
                    pass

        # 复制AblationConfig特有属性
        for attr in ['ABLATION_SEEDS', 'A1_STEPS_LIST', 'A2_K_LIST',
                     'A3_EPS_LIST', 'A4_STAT_WEIGHTS', 'A4_SMOOTH_WEIGHTS',
                     'ABLATION_EPISODES', 'ABLATION_EVAL_INTERVAL',
                     'ABLATION_SAVE_DIR', 'B1_INNER_STEPS', 'B_SMAX',
                     'B5_LAMBDA_STAT', 'B5_LAMBDA_SMOOTH']:
            setattr(config, attr, getattr(cls, attr))

        config.USE_PURE_MAML = False
        return config

    @classmethod
    def get_variant_count(cls, experiment: str) -> int:
        """获取变体数量"""
        if experiment.startswith('A'):
            counts = {'A1': 4, 'A2': 3, 'A3': 3, 'A4': 3}
            return counts.get(experiment, 0)
        elif experiment.startswith('B'):
            return 1
        return 0

    @classmethod
    def get_all_ablation_runs(cls, group: str = None) -> list:
        """生成所有运行配置"""
        runs = []

        # A组
        if group is None or group == 'A':
            for exp, variants in [('A1', cls.A1_STEPS_LIST),
                                  ('A2', cls.A2_K_LIST),
                                  ('A3', cls.A3_EPS_LIST),
                                  ('A4', cls.A4_STAT_WEIGHTS)]:
                for seed in cls.ABLATION_SEEDS:
                    for i in range(len(variants)):
                        runs.append((exp, i, seed))

        # B组（5组）
        if group is None or group == 'B':
            for exp in ['B1', 'B2', 'B3', 'B4', 'B5']:
                for seed in cls.ABLATION_SEEDS:
                    runs.append((exp, 0, seed))

        return runs

    @classmethod
    def print_experiment_plan(cls):
        """打印实验计划"""
        print("=" * 75)
        print("SGD-Y-MAML 消融实验计划（新B组：B1-B5）")
        print("=" * 75)

        print(f"\n【B组设计】递进叠加，统一S_max={cls.B_SMAX}（B3-B5）")
        print("-" * 75)

        experiments_B = [
            ('B1', 'Baseline', '固定5步，无momentum，无衰减，无早停，无正则'),
            ('B2', '+Momentum', 'momentum=0.9，仍固定5步，无衰减'),
            ('B3', '+双重收敛', f'S_max={cls.B_SMAX}，无LR衰减，启用早停（K=3无效），无正则'),
            ('B4', '+LR衰减', f'S_max={cls.B_SMAX}，启用LR衰减（K=3），启用早停，无正则'),
            ('B5', 'Full',
             f'S_max={cls.B_SMAX}，完整系统，+正则化（λ_stat={cls.B5_LAMBDA_STAT}, λ_smooth={cls.B5_LAMBDA_SMOOTH}）')
        ]

        for exp_id, name, desc in experiments_B:
            print(f"\n【{exp_id}】{name}")
            print(f"    {desc}")

        total_B = len(experiments_B) * len(cls.ABLATION_SEEDS)
        print(f"\nB组总计: {total_B} runs（5组 × 3种子）")

        print(f"\n{'=' * 75}")
        print("递进对比：")
        print("  B2 vs B1 = momentum基础增益")
        print("  B3 vs B2 = 双重收敛贡献（自适应步数+早停，无衰减）")
        print("  B4 vs B3 = LR衰减贡献（K=3启用衰减）")
        print("  B5 vs B4 = 正则化贡献（统计+平滑正则）")
        print(f"{'=' * 75}")