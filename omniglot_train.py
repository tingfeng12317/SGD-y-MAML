# omniglot_train.py
import os
import sys
import argparse
import torch
import json
import csv
import random
import numpy as np
from datetime import datetime
import time
from typing import Dict, List
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data.omniglot_dataset import OmniglotDataset


# ========== 参数更新追踪器（支持梯度统计） ==========
class UpdateTracker:
    """
    追踪MAML内循环中每层参数的更新幅度（L2 norm）
    支持SGD-Y的梯度统计（原始梯度模长 vs 残差模长）
    """

    def __init__(self, save_dir: str, track_interval: int = 10):
        self.save_dir = save_dir
        self.track_interval = track_interval
        os.makedirs(save_dir, exist_ok=True)

        # 参数更新追踪CSV
        self.csv_path = os.path.join(save_dir, "update_magnitudes.csv")
        # 梯度统计CSV（记录原始梯度 vs 残差对比）
        self.gradient_csv_path = os.path.join(save_dir, "gradient_stats.csv")

        self.episode_count = 0

        # 定义层分类（浅层 vs 深层）
        self.shallow_layers = ['layer1', 'layer2']
        self.deep_layers = ['layer3', 'layer4', 'fc']

        # 初始化CSV文件
        self._init_csv()
        self._init_gradient_csv()

    def _init_csv(self):
        """初始化参数更新CSV"""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'layer_name', 'depth_type',
                'update_magnitude', 'param_norm', 'relative_update',
                'inner_step', 'task_id'
            ])

    def _init_gradient_csv(self):
        """初始化梯度统计CSV"""
        with open(self.gradient_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'task_id', 'inner_step', 'layer_name',
                'grad_norm',  # 原始梯度模长 ||∇L||
                'residual_norm',  # 残差模长 ||∇L - buf||
                'residual_ratio',  # 残差/原始梯度 比值
                'depth_type',  # shallow/deep
                'converged',  # 是否收敛
                'total_steps'  # 总步数
            ])

    def compute_update_magnitude(self, old_params: Dict, new_params: Dict,
                                 task_id: int = 0, step: int = 0) -> List[Dict]:
        """
        计算参数更新幅度（新旧参数的L2距离）
        """
        records = []

        for name in old_params.keys():
            if name not in new_params:
                continue

            old_val = old_params[name]
            new_val = new_params[name]

            # 计算更新量
            delta = new_val - old_val
            update_norm = torch.norm(delta).item()
            param_norm = torch.norm(old_val).item()
            relative_update = update_norm / (param_norm + 1e-8)

            # 判断层深度
            depth_type = 'shallow' if any(s in name for s in self.shallow_layers) else 'deep'

            records.append({
                'layer_name': name,
                'depth_type': depth_type,
                'update_magnitude': update_norm,
                'param_norm': param_norm,
                'relative_update': relative_update,
                'task_id': task_id,
                'step': step
            })

        return records

    def log_updates(self, episode: int, update_records: List[Dict]):
        """记录参数更新到CSV"""
        self.episode_count = episode

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for rec in update_records:
                writer.writerow([
                    episode,
                    rec['layer_name'],
                    rec['depth_type'],
                    f"{rec['update_magnitude']:.6f}",
                    f"{rec['param_norm']:.6f}",
                    f"{rec['relative_update']:.6f}",
                    rec['step'],
                    rec['task_id']
                ])

    def log_gradient_stats(self, episode: int, gradient_history: List[Dict],
                           task_id: int = 0, converged: bool = False, total_steps: int = 0):
        """
        记录梯度统计（原始梯度模长 vs 梯度残差模长）
        用于分析SGD-Y的自适应收敛机制
        """
        with open(self.gradient_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for step_data in gradient_history:
                inner_step = step_data['step']
                for layer_stat in step_data['layer_stats']:
                    # 判断层深度类型
                    depth_type = 'shallow' if any(
                        s in layer_stat['layer_name'] for s in self.shallow_layers) else 'deep'

                    writer.writerow([
                        episode,
                        task_id,
                        inner_step,
                        layer_stat['layer_name'],
                        f"{layer_stat['grad_norm']:.8f}",
                        f"{layer_stat['residual_norm']:.8f}",
                        f"{layer_stat['residual_ratio']:.6f}",
                        depth_type,
                        int(converged),
                        total_steps
                    ])

    def should_track(self, episode: int) -> bool:
        """判断当前episode是否需要记录"""
        return episode % self.track_interval == 0

    def close(self):
        """保存汇总统计"""
        summary_path = os.path.join(self.save_dir, "update_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Update Tracking Summary\n")
            f.write(f"{'=' * 50}\n")
            f.write(f"Total episodes tracked: {self.episode_count}\n")
            f.write(f"Parameter Update CSV: {self.csv_path}\n")
            if os.path.exists(self.gradient_csv_path):
                # 检查文件是否有数据（不只是表头）
                with open(self.gradient_csv_path, 'r') as gf:
                    lines = len(gf.readlines())
                    if lines > 1:
                        f.write(f"Gradient Stats CSV: {self.gradient_csv_path}\n")
                        f.write(f"  - Records: original_grad_norm vs residual_norm\n")
                        f.write(f"  - Lines: {lines}\n")
        print(f"📊 追踪数据已保存至: {self.save_dir}")


# ========== Monkey Patch 包装器 ==========
def create_tracked_inner_loop(original_inner_loop, tracker: UpdateTracker):
    """
    创建带追踪的inner_loop包装器（用于标准MAML）
    """

    def tracked_inner_loop(model, support_x, support_y, inner_steps, inner_lr, first_order=False):
        # 获取初始参数快照
        initial_params = {name: param.clone().detach()
                          for name, param in model.named_parameters()}

        # 调用原始inner_loop
        adapted_params, loss_val = original_inner_loop(
            model, support_x, support_y, inner_steps, inner_lr, first_order
        )

        # 计算并记录更新幅度
        if hasattr(tracker, '_current_episode') and tracker._current_episode > 0:
            update_records = tracker.compute_update_magnitude(
                initial_params, adapted_params,
                task_id=0, step=inner_steps - 1
            )
            tracker.log_updates(tracker._current_episode, update_records)

        return adapted_params, loss_val

    return tracked_inner_loop


def create_tracked_adapt(original_adapt, tracker: UpdateTracker, model):
    """
    创建带追踪的adapt包装器（用于SGD-Y-MAML）
    固定步数：只记录参数更新幅度
    自适应步数：记录参数更新 + 梯度统计（如果可用）
    """

    def tracked_adapt(support_x, support_y, task_id=0):
        # 记录初始参数（内循环前）
        initial_params = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        # 调用原始adapt方法
        adapted_params, info = original_adapt(support_x, support_y, task_id)

        # 计算并记录更新幅度
        if hasattr(tracker, '_current_episode') and tracker._current_episode > 0:
            episode = tracker._current_episode

            # 1. 始终记录参数更新幅度
            actual_steps = info.get('steps', 0) - 1 if info.get('steps', 0) > 0 else 0
            update_records = tracker.compute_update_magnitude(
                initial_params, adapted_params,
                task_id=task_id, step=actual_steps
            )
            tracker.log_updates(episode, update_records)

            # 2. 仅在非固定步数且有梯度历史时记录梯度统计
            if not info.get('force_fixed_steps', False) and info.get('gradient_history'):
                if len(info['gradient_history']) > 0:  # 确保有数据
                    tracker.log_gradient_stats(
                        episode=episode,
                        gradient_history=info['gradient_history'],
                        task_id=task_id,
                        converged=info.get('converged', False),
                        total_steps=info.get('steps', 0)
                    )

        return adapted_params, info

    return tracked_adapt


def parse_args():
    parser = argparse.ArgumentParser(description='Few-Shot Learning: MAML vs SGD-Y-MAML vs TamingMAML vs ES-MAML')

    # 算法选择
    parser.add_argument('--algorithms', type=str, nargs='+',
                        choices=['maml', 'maml_fo', 'sgd_y_maml', 'sgd_y_maml_fo', 'taming_maml', 'es_maml'],
                        default=['maml', 'taming_maml', 'es_maml'],
                        help='要运行的算法列表')

    # 任务设置
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--k_shot', type=int, default=1)
    parser.add_argument('--k_query', type=int, default=15)

    # 通用训练参数
    parser.add_argument('--meta_lr', type=float, default=None)
    parser.add_argument('--meta_batch_size', type=int, default=None)
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--eval_interval', type=int, default=10)

    # ========== 新增：MAML 内循环步数控制 ==========
    parser.add_argument('--inner_steps', type=int, default=None,
                        help='MAML/TamingMAML 内循环固定步数（默认使用 config.py 中的 INNER_STEPS）')
    parser.add_argument('--inner_lr', type=float, default=None,
                        help='内循环学习率（默认使用 config.py 中的 INNER_LR）')

    # 路径
    parser.add_argument('--data_root', type=str, default='./omniglot_data')
    parser.add_argument('--save_dir', type=str, default='./results/all_algorithms')

    # 多种子运行
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 1024])

    # ========== 参数更新追踪配置 ==========
    parser.add_argument('--track_updates', action='store_true', default=False,
                        help='启用参数更新幅度追踪（生成CSV用于绘制深层vs浅层更新对比图）')
    parser.add_argument('--track_dir', type=str, default='./update_tracking',
                        help='参数追踪CSV保存目录')
    parser.add_argument('--track_interval', type=int, default=10,
                        help='每隔多少episode记录一次更新幅度')

    # ========== SGD-Y-MAML 特定配置 ==========
    parser.add_argument('--sgdy_no_momentum', action='store_true',
                        help='SGD-Y-MAML: 关闭动量（momentum=0）')
    parser.add_argument('--sgdy_no_lr_decay', action='store_true',
                        help='SGD-Y-MAML: 关闭自适应学习率衰减')
    parser.add_argument('--sgdy_fixed_steps', type=int, default=None,
                        help='SGD-Y-MAML: 使用固定步数（关闭自适应早停），指定步数如5')

    return parser.parse_args()


def update_config(args, seed):
    """用命令行参数更新Config，并设置特定种子"""
    Config.N_WAY = args.n_way
    Config.K_SHOT = args.k_shot
    Config.K_QUERY = args.k_query
    Config.MAX_EPISODES = args.episodes
    Config.EVAL_INTERVAL = args.eval_interval
    Config.DATA_ROOT = args.data_root
    Config.SEED = seed

    if args.meta_lr is not None:
        Config.META_LR = args.meta_lr
    if args.meta_batch_size is not None:
        Config.META_BATCH_SIZE = args.meta_batch_size

    # ========== 新增：命令行控制内循环参数 ==========
    if args.inner_steps is not None:
        Config.INNER_STEPS = args.inner_steps
    if args.inner_lr is not None:
        Config.INNER_LR = args.inner_lr

    # 设置种子并确保确定性
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    return Config.DEVICE


def get_gpu_memory():
    """获取当前GPU显存占用（MB）"""
    if not torch.cuda.is_available():
        return 0
    return round(torch.cuda.max_memory_allocated() / 1e6, 1)


def reset_gpu_memory():
    """重置显存统计"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def save_detailed_log(save_dir, algorithm, seed, history, display_name):
    """保存每10episodes的详细记录到JSON文件"""
    detailed_dir = os.path.join(save_dir, "detailed_logs")
    os.makedirs(detailed_dir, exist_ok=True)

    log_file = os.path.join(detailed_dir, f"{algorithm}_seed{seed}.json")

    detailed_record = {
        'algorithm': algorithm,
        'display_name': display_name,
        'seed': seed,
        'timestamp': datetime.now().isoformat(),
        'records': []
    }

    episodes = list(range(10, len(history['train_loss']) * 10 + 1, 10))
    for i, ep in enumerate(episodes):
        if i < len(history['train_loss']):
            record = {
                'episode': ep,
                'train_loss': history['train_loss'][i],
                'train_acc': history['train_acc'][i],
                'val_acc': history['val_acc'][i] if i < len(history['val_acc']) else None,
                'avg_steps': history.get('avg_steps', [Config.INNER_STEPS] * len(history['train_loss']))[i]
                if 'avg_steps' in history else Config.INNER_STEPS
            }
            detailed_record['records'].append(record)

    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_record, f, indent=2)

    return log_file


def run_single_experiment(algorithm, seed, args, save_dir):
    """运行单个算法-种子组合（集成追踪功能）"""

    # 解析算法名称和配置
    algo_map = {
        'maml': ('maml', False, 'MAML (Baseline)'),
        'maml_fo': ('maml', True, 'FOMAML (First-Order)'),
        'sgd_y_maml': ('sgd_y_maml', False, 'SGD-Y-MAML (Adaptive)'),
        'sgd_y_maml_fo': ('sgd_y_maml', True, 'SGD-Y-FOMAML'),
        'taming_maml': ('taming_maml', False, 'TamingMAML (Regularized)'),
        'es_maml': ('es_maml', False, 'ES-MAML (Evolution Strategy)')
    }

    algo_name, first_order, display_name = algo_map.get(algorithm, (algorithm, False, algorithm))

    print(f"\n{'=' * 80}")
    print(f"🚀 [{display_name}] | Seed {seed}")
    print(f"{'=' * 80}")

    device = update_config(args, seed)
    reset_gpu_memory()
    start_time = time.time()

    # 加载数据
    try:
        train_dataset = OmniglotDataset(Config.DATA_ROOT, background=True)
        val_dataset = OmniglotDataset(Config.DATA_ROOT, background=False)
    except FileNotFoundError:
        print(f"❌ 找不到数据集: {Config.DATA_ROOT}")
        return None

    # 初始化追踪器
    tracker = None
    original_inner_loop_maml = None
    original_adapt_sgd_y = None
    trainer = None

    # 根据算法类型配置追踪
    if args.track_updates and algo_name in ['maml', 'maml_fo', 'sgd_y_maml', 'sgd_y_maml_fo']:
        tracker_dir = os.path.join(args.track_dir, f"{algorithm}_seed{seed}")
        tracker = UpdateTracker(tracker_dir, args.track_interval)

        if algo_name in ['maml', 'maml_fo']:
            import algorithms.maml as maml_module
            original_inner_loop_maml = maml_module.inner_loop
            maml_module.inner_loop = create_tracked_inner_loop(original_inner_loop_maml, tracker)

        print(f"📊 参数更新追踪已启用 -> {tracker_dir}")

    # 定义step_callback
    def step_callback(episode, train_loss, train_acc,
                      val_loss, val_acc, avg_steps,
                      best_val_acc, model):
        if tracker:
            tracker._current_episode = episode

    history = None

    try:
        if algo_name == 'maml':
            from algorithms.maml import MAML
            from models.omniglot_net import OmniglotNet

            model = OmniglotNet(num_classes=Config.N_WAY, hidden_dim=64)
            trainer = MAML(model, Config)

            print(f"配置: 固定{Config.INNER_STEPS}步, lr={Config.INNER_LR}, "
                  f"{'一阶' if first_order else '二阶'}")

            history = trainer.train(train_dataset, val_dataset,
                                    first_order=first_order,
                                    step_callback=step_callback)

        elif algo_name == 'taming_maml':
            from algorithms.taming_maml import TamingMAML
            from models.omniglot_net import OmniglotNet

            model = OmniglotNet(num_classes=Config.N_WAY, hidden_dim=64)
            trainer = TamingMAML(model, Config)

            print(f"配置: 固定{Config.INNER_STEPS}步, lr={Config.INNER_LR}, "
                  f"α_reg={Config.TAMING_ALPHA_REG}, grad_clip={Config.TAMING_GRAD_CLIP}")

            history = trainer.train(train_dataset, val_dataset, first_order=first_order)

        elif algo_name == 'es_maml':
            from algorithms.es_maml import ESMAML
            from models.omniglot_net import OmniglotNet

            model = OmniglotNet(num_classes=Config.N_WAY, hidden_dim=64)
            original_meta_lr = Config.META_LR
            Config.META_LR = Config.ES_META_LR
            trainer = ESMAML(model, Config)
            Config.META_LR = original_meta_lr

            print(f"配置: σ={Config.ES_SIGMA}, 扰动数={Config.ES_N_PERTURBATIONS}")

            history = trainer.train(train_dataset, val_dataset)

        elif algo_name == 'sgd_y_maml':
            from algorithms.sgd_y_maml import SGD_Y_MAML
            from models.omniglot_net import OmniglotNet

            model = OmniglotNet(num_classes=Config.N_WAY, hidden_dim=64)

            # 判断是否为固定步数模式
            is_fixed_steps = args.sgdy_fixed_steps is not None

            # 构建消融配置
            ablation_cfg = {
                'USE_LR_DECAY': not args.sgdy_no_lr_decay,
                'USE_REGULARIZATION': False,
                'SGDY_STATISTICAL_WEIGHT': 0.0,
                'SGDY_GRADIENT_SMOOTH_WEIGHT': 0.0,
                'FORCE_FIXED_STEPS': is_fixed_steps,
                'FIXED_STEPS': args.sgdy_fixed_steps if is_fixed_steps else Config.INNER_STEPS,
                # 关键：梯度统计仅在非固定步数且启用追踪时开启
                'ENABLE_GRADIENT_STATS': (args.track_updates and not is_fixed_steps)
            }

            # 其他配置覆盖
            if args.sgdy_no_momentum:
                Config.SGDY_MOMENTUM = 0.0

            trainer = SGD_Y_MAML(model, Config, ablation_config=ablation_cfg)

            # 打印配置信息
            if is_fixed_steps:
                print(f"配置: 固定{args.sgdy_fixed_steps}步 | 早停:禁用 | 梯度统计:关")
            else:
                grad_stats_status = "开" if (args.track_updates and not is_fixed_steps) else "关"
                print(f"配置: 自适应步数(max={Config.SGDY_INNER_STEPS_MAX}) | 早停:启用 | 梯度统计:{grad_stats_status}")

            # 为SGD-Y-MAML实例添加adapt方法追踪
            if tracker:
                original_adapt_sgd_y = trainer.adapt
                trainer.adapt = create_tracked_adapt(original_adapt_sgd_y, tracker, trainer.model)

            # 训练
            history = trainer.train(train_dataset, val_dataset,
                                    first_order=first_order,
                                    step_callback=step_callback if tracker else None)

    finally:
        # 清理：恢复原始函数
        if original_inner_loop_maml is not None:
            import algorithms.maml as maml_module
            maml_module.inner_loop = original_inner_loop_maml

        if original_adapt_sgd_y is not None and trainer is not None:
            trainer.adapt = original_adapt_sgd_y

        if tracker:
            tracker.close()

    # 保存结果
    if history is None:
        print("❌ 训练失败，无历史记录")
        return None

    detailed_log_path = save_detailed_log(save_dir, algorithm, seed, history, display_name)
    print(f"💾 详细日志已保存: {detailed_log_path}")

    gpu_mem_peak = get_gpu_memory()
    elapsed_time = time.time() - start_time
    elapsed_minutes = elapsed_time / 60

    result = {
        'algorithm': algorithm,
        'display_name': display_name,
        'seed': seed,
        'first_order': first_order,
        'best_val_acc': max(history['val_acc']) if history.get('val_acc') else 0.0,
        'final_train_loss': history['train_loss'][-1] if history['train_loss'] else 0.0,
        'final_train_acc': history['train_acc'][-1] if history['train_acc'] else 0.0,
        'avg_steps': np.mean(history.get('avg_steps', [Config.INNER_STEPS]))
        if 'avg_steps' in history else Config.INNER_STEPS,
        'gpu_memory_mb': gpu_mem_peak,
        'elapsed_time': elapsed_minutes,
        'history': history,
        'detailed_log': detailed_log_path,
        'update_tracking_csv': tracker.csv_path if tracker else None,
        'gradient_stats_csv': tracker.gradient_csv_path if tracker and os.path.exists(
            tracker.gradient_csv_path) else None
    }

    print(f"\n✅ 完成: ValAcc={result['best_val_acc']:.2%}, "
          f"GPU={result['gpu_memory_mb']:.1f}MB, "
          f"Time={elapsed_minutes:.1f}min")

    if tracker:
        print(f"   参数追踪: {tracker.csv_path}")
        if result['gradient_stats_csv']:
            print(f"   梯度统计: {result['gradient_stats_csv']}")

    return result


def save_all_results(all_results, args, save_dir):
    """保存所有算法的聚合结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    algo_groups = {}
    for r in all_results:
        algo = r['algorithm']
        if algo not in algo_groups:
            algo_groups[algo] = []
        algo_groups[algo].append(r)

    # 计算统计量
    summary_rows = []
    supported_algos = ['maml', 'maml_fo', 'sgd_y_maml', 'sgd_y_maml_fo', 'taming_maml', 'es_maml']

    for algo in supported_algos:
        if algo not in algo_groups:
            continue

        results = algo_groups[algo]
        seeds = [r['seed'] for r in results]

        best_val_accs = [r['best_val_acc'] for r in results]
        final_train_losses = [r['final_train_loss'] for r in results]
        final_train_accs = [r['final_train_acc'] for r in results]
        avg_steps_list = [r['avg_steps'] for r in results]
        gpu_memories = [r['gpu_memory_mb'] for r in results]
        elapsed_times = [r['elapsed_time'] for r in results]

        def mean_std_str(values, fmt=".2f", is_percentage=False):
            mean_val = np.mean(values)
            std_val = np.std(values)
            if is_percentage:
                return f"{mean_val * 100:.2f}±{std_val * 100:.2f}%"
            return f"{mean_val:{fmt}}±{std_val:.2f}"

        display_name = results[0]['display_name']

        summary_rows.append({
            'algorithm': algo,
            'display_name': display_name,
            'n_seeds': len(results),
            'seeds': seeds,
            'ValAcc': mean_std_str(best_val_accs, is_percentage=True),
            'TrainLoss': mean_std_str(final_train_losses, fmt=".3f"),
            'TrainAcc': mean_std_str(final_train_accs, is_percentage=True),
            'Steps': mean_std_str(avg_steps_list, fmt=".1f"),
            'GPU_MB': mean_std_str(gpu_memories, fmt=".1f"),
            'Time': mean_std_str(elapsed_times, fmt=".1f"),
            'raw': {
                'best_val_acc': best_val_accs,
                'final_train_loss': final_train_losses,
                'final_train_acc': final_train_accs,
                'avg_steps': avg_steps_list,
                'gpu_memory_mb': gpu_memories,
                'elapsed_time': elapsed_times
            }
        })

    # 打印表格
    print(f"\n{'=' * 110}")
    print("完整对比表格（多种算法 × 多 seeds）")
    print(f"{'=' * 110}")
    print(
        f"{'实验':<25} | {'ValAcc':<12} | {'TrainLoss':<10} | {'TrainAcc':<12} | {'步数':<8} | {'显存(MB)':<12} | {'时间(min)':<10}")
    print("-" * 110)

    for row in summary_rows:
        print(f"{row['display_name']:<25} | {row['ValAcc']:<12} | "
              f"{row['TrainLoss']:<10} | {row['TrainAcc']:<12} | "
              f"{row['Steps']:<8} | {row['GPU_MB']:<12} | {row['Time']:<10}")

    print(f"{'=' * 110}")
    print(f"总实验数: {len(all_results)}")

    # 保存CSV
    csv_summary = os.path.join(save_dir, f'summary_{timestamp}.csv')
    with open(csv_summary, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['algorithm', 'display_name', 'n_seeds', 'ValAcc', 'TrainLoss',
                         'TrainAcc', 'Steps', 'GPU_MB', 'Time_min'])
        for row in summary_rows:
            writer.writerow([
                row['algorithm'],
                row['display_name'],
                row['n_seeds'],
                row['ValAcc'],
                row['TrainLoss'],
                row['TrainAcc'],
                row['Steps'],
                row['GPU_MB'],
                row['Time']
            ])

    # 保存CSV（每seed）
    csv_all = os.path.join(save_dir, f'all_runs_{timestamp}.csv')
    with open(csv_all, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['algorithm', 'display_name', 'seed', 'best_val_acc',
                         'final_train_loss', 'final_train_acc', 'avg_steps',
                         'gpu_memory_mb', 'elapsed_time_min', 'detailed_log',
                         'update_tracking_csv', 'gradient_stats_csv'])
        for r in all_results:
            writer.writerow([
                r['algorithm'],
                r['display_name'],
                r['seed'],
                f"{r['best_val_acc']:.4f}",
                f"{r['final_train_loss']:.4f}",
                f"{r['final_train_acc']:.4f}",
                f"{r['avg_steps']:.1f}",
                f"{r['gpu_memory_mb']:.1f}",
                f"{r['elapsed_time']:.2f}",
                r.get('detailed_log', ''),
                r.get('update_tracking_csv', ''),
                r.get('gradient_stats_csv', '')
            ])

    # 保存JSON
    json_path = os.path.join(save_dir, f'summary_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump({
            'config': {
                'n_way': args.n_way,
                'k_shot': args.k_shot,
                'episodes': args.episodes,
                'seeds': args.seeds,
                'algorithms': args.algorithms,
                'track_updates': args.track_updates
            },
            'results': summary_rows
        }, f, indent=2)

    print(f"\n💾 结果已保存:")
    print(f"   汇总CSV: {csv_summary}")
    print(f"   原始数据CSV: {csv_all}")
    print(f"   JSON: {json_path}")

    return summary_rows


def main():
    args = parse_args()

    total_experiments = len(args.algorithms) * len(args.seeds)

    print("=" * 80)
    print("批量运行: 多种算法 × 多 seeds 对比实验")
    print(f"算法: {args.algorithms}")
    print(f"Seeds: {args.seeds}")
    print(f"保存目录: {args.save_dir}")

    # 显示步数配置
    inner_steps_display = args.inner_steps if args.inner_steps is not None else "config默认"
    print(f"内循环步数: {inner_steps_display}")

    if args.track_updates:
        print(f"📊 参数追踪: 已启用 (间隔: {args.track_interval})")
        print(f"📊 追踪目录: {args.track_dir}")
    print("=" * 80)

    os.makedirs(args.save_dir, exist_ok=True)

    # 运行所有组合
    all_results = []
    exp_idx = 0

    for algo in args.algorithms:
        for seed in args.seeds:
            exp_idx += 1
            print(f"\n\n{'#' * 80}")
            print(f"# 进度: [{exp_idx}/{total_experiments}] {algo} | seed={seed}")
            print(f"{'#' * 80}")

            result = run_single_experiment(algo, seed, args, args.save_dir)
            if result is not None:
                all_results.append(result)

    if len(all_results) == 0:
        print("❌ 没有成功运行的实验")
        return

    # 保存聚合结果
    save_all_results(all_results, args, args.save_dir)

    print(f"\n{'=' * 80}")
    print(f"全部完成！成功: {len(all_results)}/{total_experiments}")
    if args.track_updates:
        print(f"\n📊 追踪数据已生成，运行以下命令分析:")
        print(f"   python plot_update_analysis.py --track_dir {args.track_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()