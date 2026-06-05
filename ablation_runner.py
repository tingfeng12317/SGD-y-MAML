"""
消融实验运行器 - 最终版（确定性/可复现版本）
修改：禁用 cudnn.benchmark，添加完整随机种子控制
"""

import os
import sys
import csv
import time
import torch
import argparse
import numpy as np
import json
import random
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

# ========== GPU优化设置（修改：禁用 benchmark 以确保确定性） ==========
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False  # 关键修改：禁用动态算法选择
    torch.backends.cudnn.deterministic = True  # 关键修改：强制确定性算法
    torch.backends.cudnn.enabled = True
    _default_device = torch.device("cuda:0")
else:
    _default_device = torch.device("cpu")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ablation_config import AblationConfig
from config import Config
from data.omniglot_dataset import OmniglotDataset
from algorithms.maml import MAML
from algorithms.sgd_y_maml import SGD_Y_MAML
from models.omniglot_net import OmniglotNet


class AblationRunner:
    def __init__(self, save_dir: str = None, device: str = None):
        if device:
            self.device = torch.device(device)
        else:
            self.device = _default_device

        self._print_device_info()

        self.save_dir = save_dir or AblationConfig.ABLATION_SAVE_DIR
        os.makedirs(self.save_dir, exist_ok=True)

        # 详细日志目录（每10ep记录）
        self.detailed_dir = os.path.join(self.save_dir, "detailed_logs")
        os.makedirs(self.detailed_dir, exist_ok=True)

        # CSV文件路径（汇总结果）
        self.csv_path = os.path.join(self.save_dir, "results.csv")
        self._init_csv()

        self.results = []
        self.start_time = None

    def _set_seed(self, seed: int):
        """完全可复现性设置（关键修改）"""
        # Python 内置随机
        random.seed(seed)
        # Numpy
        np.random.seed(seed)
        # PyTorch CPU
        torch.manual_seed(seed)
        # PyTorch CUDA
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # 多 GPU
        # Python hash 种子（影响字典顺序等）
        os.environ['PYTHONHASHSEED'] = str(seed)
        # 单线程避免线程调度随机性（可选，会减慢速度）
        # torch.set_num_threads(1)

    def _init_csv(self):
        """初始化CSV文件"""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'exp_group', 'experiment', 'seed', 'trainer_type',
                    'best_val_acc', 'final_train_loss', 'final_train_acc',
                    'train_time_minutes', 'avg_inner_steps', 'use_lr_decay',
                    'k_value', 'lambda_stat', 'lambda_smooth',
                    'early_stop_rate', 'gpu_memory_peak_mb', 'status', 'error_message'
                ])

    def _print_device_info(self):
        """打印GPU/CPU信息"""
        print(f"\n{'=' * 70}")
        if self.device.type == 'cuda':
            props = torch.cuda.get_device_properties(self.device)
            print(f"🚀 使用GPU: {torch.cuda.get_device_name(self.device)}")
            print(f"   显存: {props.total_memory / 1e9:.1f} GB")
            print(f"   确定性模式: benchmark={torch.backends.cudnn.benchmark}, deterministic={torch.backends.cudnn.deterministic}")
        else:
            print(f"⚠️  使用CPU")
        print(f"{'=' * 70}\n")

    def _get_gpu_memory(self):
        """获取峰值显存（MB）"""
        if self.device.type != 'cuda':
            return 0
        return round(torch.cuda.max_memory_allocated(self.device) / 1e6, 1)

    def _reset_peak_memory(self):
        """重置显存统计"""
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)

    def _save_detailed_log(self, exp_name: str, seed: int, history: Dict):
        """保存每10ep的详细记录"""
        log_file = os.path.join(self.detailed_dir, f"{exp_name}_seed{seed}.json")
        detailed_record = {
            'experiment': exp_name,
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
                    'avg_steps': history.get('avg_steps', [0] * len(history['train_loss']))[i] if 'avg_steps' in history else None
                }
                detailed_record['records'].append(record)

        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_record, f, indent=2)

        return log_file

    def run_single_experiment(self, experiment: str, variant_idx: int, seed: int) -> Dict:
        """运行单个消融实验（关键修改：重置种子）"""
        # 关键修改：每次实验前强制设置种子，确保完全可复现
        self._set_seed(seed)

        cfg = AblationConfig.get_ablation_config(experiment, variant_idx, seed)
        cfg.DEVICE = self.device

        run_id = f"{cfg.EXPERIMENT_NAME}_seed{seed}"
        self._reset_peak_memory()

        is_pure_maml = getattr(cfg, 'USE_PURE_MAML', False)
        exp_group = getattr(cfg, 'EXPERIMENT_GROUP', 'unknown')
        trainer_type = "MAML" if is_pure_maml else "SGD-Y"

        use_lr_decay = getattr(cfg, 'USE_LR_DECAY', False)
        k_value = getattr(cfg, 'SGDY_K', 'N/A')
        lambda_stat = getattr(cfg, 'SGDY_STATISTICAL_WEIGHT', 0.0)
        lambda_smooth = getattr(cfg, 'SGDY_GRADIENT_SMOOTH_WEIGHT', 0.0)

        print(f"\n{'=' * 70}")
        print(f"开始实验: {run_id}")
        print(f"实验组: {exp_group.upper()}")
        print(f"描述: {cfg.EXPERIMENT_DESC}")
        print(f"训练器: {trainer_type}")
        print(f"配置: use_lr_decay={use_lr_decay}, K={k_value}")
        print(f"确定性设置: seed={seed}, benchmark=False, deterministic=True")
        print(f"{'=' * 70}")

        try:
            train_dataset = OmniglotDataset(cfg.DATA_ROOT, background=True)
            val_dataset = OmniglotDataset(cfg.DATA_ROOT, background=False)
        except FileNotFoundError as e:
            print(f"❌ 数据集错误: {e}")
            return self._create_error_result(cfg, run_id, str(e))

        model = OmniglotNet(num_classes=cfg.N_WAY, hidden_dim=64).to(self.device)

        if not is_pure_maml:
            ablation_cfg = {
                'USE_LR_DECAY': use_lr_decay,
                'USE_REGULARIZATION': getattr(cfg, 'USE_REGULARIZATION', False),
                'SGDY_STATISTICAL_WEIGHT': lambda_stat,
                'SGDY_GRADIENT_SMOOTH_WEIGHT': lambda_smooth
            }
            trainer = SGD_Y_MAML(model, cfg, ablation_config=ablation_cfg)
        else:
            trainer = MAML(model, cfg)

        exp_start = time.time()
        try:
            history = trainer.train(train_dataset, val_dataset, first_order=False)
            exp_time = time.time() - exp_start

            best_val_acc = max(history['val_acc']) if history['val_acc'] else 0.0
            final_train_loss = history['train_loss'][-1] if history['train_loss'] else 0.0
            final_train_acc = history['train_acc'][-1] if history['train_acc'] else 0.0

            if is_pure_maml:
                avg_steps = cfg.INNER_STEPS
                early_stop_rate = 0.0
            else:
                if hasattr(trainer, 'task_stats') and trainer.task_stats:
                    recent = trainer.task_stats[-100:]
                    avg_steps = sum(s['steps'] for s in recent) / len(recent)
                    early_stops = sum(1 for s in recent if s.get('early_stop', False))
                    early_stop_rate = early_stops / len(recent)
                else:
                    avg_steps = cfg.SGDY_INNER_STEPS_MAX / 2
                    early_stop_rate = 0.0

            detailed_log_path = self._save_detailed_log(cfg.EXPERIMENT_NAME, seed, history)

            gpu_mem = self._get_gpu_memory()

            result = {
                'timestamp': datetime.now().isoformat(),
                'exp_group': exp_group,
                'experiment': cfg.EXPERIMENT_NAME,
                'seed': seed,
                'trainer_type': trainer_type,
                'best_val_acc': best_val_acc,
                'final_train_loss': final_train_loss,
                'final_train_acc': final_train_acc,
                'train_time_minutes': exp_time / 60,
                'avg_inner_steps': round(avg_steps, 1),
                'use_lr_decay': use_lr_decay,
                'k_value': k_value,
                'lambda_stat': lambda_stat,
                'lambda_smooth': lambda_smooth,
                'early_stop_rate': round(early_stop_rate, 3),
                'gpu_memory_peak_mb': gpu_mem,
                'status': 'success',
                'error_message': '',
                'detailed_log': detailed_log_path
            }

            self._append_to_csv(result)
            print(f"✅ 完成: ValAcc={best_val_acc:.2%} | "
                  f"TrainLoss={final_train_loss:.3f} | "
                  f"TrainAcc={final_train_acc:.2%} | "
                  f"Time={exp_time/60:.1f}min | "
                  f"Steps={avg_steps:.1f} | "
                  f"EarlyStopRate={early_stop_rate:.1%} | "
                  f"GPU={gpu_mem}MB")

            return result

        except Exception as e:
            print(f"\n❌ 失败: {str(e)}")
            import traceback
            traceback.print_exc()
            error_result = self._create_error_result(cfg, run_id, str(e))
            self._append_to_csv(error_result)
            return error_result

    def _append_to_csv(self, result: Dict):
        """追加到CSV文件"""
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                result['timestamp'],
                result['exp_group'],
                result['experiment'],
                result['seed'],
                result['trainer_type'],
                f"{result['best_val_acc']:.4f}",
                f"{result['final_train_loss']:.4f}",
                f"{result['final_train_acc']:.4f}",
                f"{result['train_time_minutes']:.2f}",
                result['avg_inner_steps'],
                result['use_lr_decay'],
                result['k_value'],
                result['lambda_stat'],
                result['lambda_smooth'],
                result['early_stop_rate'],
                result['gpu_memory_peak_mb'],
                result['status'],
                result['error_message']
            ])

    def _create_error_result(self, cfg, run_id: str, error_msg: str) -> Dict:
        """创建错误结果"""
        exp_group = getattr(cfg, 'EXPERIMENT_GROUP', 'unknown')
        return {
            'timestamp': datetime.now().isoformat(),
            'exp_group': exp_group,
            'experiment': cfg.EXPERIMENT_NAME,
            'seed': cfg.SEED,
            'trainer_type': 'ERROR',
            'best_val_acc': 0.0,
            'final_train_loss': 0.0,
            'final_train_acc': 0.0,
            'train_time_minutes': 0.0,
            'avg_inner_steps': 0,
            'use_lr_decay': getattr(cfg, 'USE_LR_DECAY', 'N/A'),
            'k_value': getattr(cfg, 'SGDY_K', 'N/A'),
            'lambda_stat': getattr(cfg, 'SGDY_STATISTICAL_WEIGHT', 0.0),
            'lambda_smooth': getattr(cfg, 'SGDY_GRADIENT_SMOOTH_WEIGHT', 0.0),
            'early_stop_rate': 0.0,
            'gpu_memory_peak_mb': 0,
            'status': 'error',
            'error_message': error_msg[:100],
            'detailed_log': ''
        }

    def run_all_experiments(self, experiment_filter: str = None):
        """批量运行实验"""
        runs = AblationConfig.get_all_ablation_runs()

        if experiment_filter:
            if experiment_filter in ['A', 'B']:
                runs = [r for r in runs if r[0].startswith(experiment_filter)]
            else:
                filtered = [r for r in runs if r[0] == experiment_filter]
                runs = filtered if filtered else runs

        total = len(runs)
        if total == 0:
            print(f"⚠️ 未找到匹配的实验: {experiment_filter}")
            return

        print(f"\n{'=' * 70}")
        print(f"批量消融实验 | 总实验数: {total} | 保存目录: {self.save_dir}")
        print(f"{'=' * 70}")

        self.start_time = time.time()

        for i, (exp, var_idx, seed) in enumerate(runs, 1):
            print(f"\n进度: [{i}/{total}] ({i/total*100:.1f}%)")
            result = self.run_single_experiment(exp, var_idx, seed)
            self.results.append(result)

        self._print_summary(time.time() - self.start_time)

        return self.results

    def _print_summary(self, total_time: float):
        """打印汇总表格"""
        print(f"\n{'=' * 90}")
        print("消融实验完成")
        print(f"{'=' * 90}")
        print(f"总耗时: {total_time/60:.1f} 分钟")
        print(f"CSV保存至: {self.csv_path}")
        print(f"详细日志保存至: {self.detailed_dir}")

        groups = defaultdict(list)
        for r in self.results:
            g = r.get('exp_group', 'unknown')
            groups[g].append(r)

        for group_name in sorted(groups.keys()):
            print(f"\n{'=' * 90}")
            print(f"{group_name.upper()}组结果")
            print(f"{'=' * 90}")
            print(f"{'实验':<18} | {'ValAcc':<10} | {'TrainLoss':<9} | {'TrainAcc':<9} | {'步数':<6} | {'时间(min)':<10} | {'显存(MB)':<10}")
            print("-" * 90)

            exp_results = defaultdict(list)
            for r in groups[group_name]:
                exp_name = r['experiment']
                exp_results[exp_name].append(r)

            for exp_name in sorted(exp_results.keys()):
                records = exp_results[exp_name]
                succ_records = [r for r in records if r['status'] == 'success']

                if not succ_records:
                    continue

                val_accs = [r['best_val_acc'] for r in succ_records]
                train_losses = [r['final_train_loss'] for r in succ_records]
                train_accs = [r['final_train_acc'] for r in succ_records]
                steps = [r['avg_inner_steps'] for r in succ_records]
                times = [r['train_time_minutes'] for r in succ_records]
                gpu_mems = [r['gpu_memory_peak_mb'] for r in succ_records]

                def fmt_mean_std(vals, fmt=".2f", is_percentage=False):
                    if len(vals) > 1:
                        mean_val = np.mean(vals)
                        std_val = np.std(vals)
                        if is_percentage:
                            return f"{mean_val:.2%}±{std_val:.2%}"
                        return f"{mean_val:{fmt}}±{std_val:.2f}"
                    if is_percentage:
                        return f"{vals[0]:.2%}"
                    return f"{vals[0]:{fmt}}"

                val_str = fmt_mean_std(val_accs, is_percentage=True)
                loss_str = fmt_mean_std(train_losses, ".3f")
                acc_str = fmt_mean_std(train_accs, is_percentage=True)
                step_str = fmt_mean_std(steps, ".1f")
                time_str = fmt_mean_std(times, ".1f")
                mem_str = fmt_mean_std(gpu_mems, ".1f")

                print(f"{exp_name:<18} | {val_str:<10} | {loss_str:<9} | {acc_str:<9} | {step_str:<6} | {time_str:<10} | {mem_str:<10}")

        print(f"\n{'=' * 90}")


def main():
    parser = argparse.ArgumentParser(description='SGD-Y-MAML 消融实验（确定性版本）')
    parser.add_argument('--experiment', type=str, default=None,
                       help='实验组名 (如 B, B3, A)')
    parser.add_argument('--variant', type=int, default=None,
                       help='变体索引（A组使用）')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子（单实验模式）')
    parser.add_argument('--device', type=str, default=None,
                       help='设备 (cuda:0/cpu)')
    parser.add_argument('--save_dir', type=str, default='./results/ablation',
                       help='保存目录')

    args = parser.parse_args()

    runner = AblationRunner(save_dir=args.save_dir, device=args.device)

    # 单实验模式
    if args.experiment and args.variant is not None and args.seed is not None:
        result = runner.run_single_experiment(args.experiment, args.variant, args.seed)
        print(f"\n结果: {result.get('status', 'unknown')}")

    # 批量模式
    elif args.experiment:
        runner.run_all_experiments(experiment_filter=args.experiment)

    # 全部运行
    else:
        runner.run_all_experiments()


if __name__ == "__main__":
    main()