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
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data.cifar_fs_dataset import CifarFSDataset


# ========== 参数更新追踪器 ==========
class UpdateTracker:
    def __init__(self, save_dir: str, track_interval: int = 10):
        self.save_dir = save_dir
        self.track_interval = track_interval
        os.makedirs(save_dir, exist_ok=True)
        self.csv_path = os.path.join(save_dir, "update_magnitudes.csv")
        self.episode_count = 0
        self.shallow_layers = ['layer1', 'layer2']
        self.deep_layers = ['layer3', 'layer4', 'fc']
        self._init_csv()

    def _init_csv(self):
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'layer_name', 'depth_type',
                'update_magnitude', 'param_norm', 'relative_update',
                'inner_step', 'task_id'
            ])

    def compute_update_magnitude(self, old_params: Dict, new_params: Dict,
                                 task_id: int = 0, step: int = 0) -> List[Dict]:
        records = []
        for name in old_params.keys():
            if name not in new_params:
                continue
            old_val = old_params[name]
            new_val = new_params[name]
            delta = new_val - old_val
            update_norm = torch.norm(delta).item()
            param_norm = torch.norm(old_val).item()
            relative_update = update_norm / (param_norm + 1e-8)
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
        self.episode_count = episode
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for rec in update_records:
                writer.writerow([
                    episode, rec['layer_name'], rec['depth_type'],
                    f"{rec['update_magnitude']:.6f}", f"{rec['param_norm']:.6f}",
                    f"{rec['relative_update']:.6f}", rec['step'], rec['task_id']
                ])

    def should_track(self, episode: int) -> bool:
        return episode % self.track_interval == 0

    def close(self):
        summary_path = os.path.join(self.save_dir, "update_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Update Tracking Summary\n{'=' * 50}\n")
            f.write(f"Total episodes tracked: {self.episode_count}\n")
        print(f"📊 追踪数据已保存至: {self.csv_path}")


# ========== Monkey Patch 包装器 ==========
def create_tracked_inner_loop(original_inner_loop, tracker: UpdateTracker):
    def tracked_inner_loop(model, support_x, support_y, inner_steps, inner_lr, first_order=False):
        initial_params = {name: param.clone().detach() for name, param in model.named_parameters()}
        adapted_params, loss_val = original_inner_loop(model, support_x, support_y, inner_steps, inner_lr, first_order)
        if hasattr(tracker, '_current_episode') and tracker._current_episode > 0:
            update_records = tracker.compute_update_magnitude(initial_params, adapted_params, task_id=0,
                                                              step=inner_steps - 1)
            tracker.log_updates(tracker._current_episode, update_records)
        return adapted_params, loss_val

    return tracked_inner_loop


def create_tracked_adapt(original_adapt, tracker: UpdateTracker, model):
    def tracked_adapt(support_x, support_y, task_id=0):
        initial_params = {name: param.clone().detach() for name, param in model.named_parameters() if
                          param.requires_grad}
        adapted_params, info = original_adapt(support_x, support_y, task_id)
        if hasattr(tracker, '_current_episode') and tracker._current_episode > 0:
            actual_steps = info.get('steps', 0) - 1 if info.get('steps', 0) > 0 else 0
            update_records = tracker.compute_update_magnitude(initial_params, adapted_params, task_id=task_id,
                                                              step=actual_steps)
            tracker.log_updates(tracker._current_episode, update_records)
        return adapted_params, info

    return tracked_adapt


def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-FS Few-Shot Learning')

    parser.add_argument('--algorithms', type=str, nargs='+',
                        choices=['maml', 'maml_fo', 'sgd_y_maml', 'sgd_y_maml_fo', 'taming_maml', 'es_maml'],
                        default=['maml'],
                        help='要运行的算法列表')

    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--k_shot', type=int, default=1)
    parser.add_argument('--k_query', type=int, default=15)

    parser.add_argument('--meta_lr', type=float, default=None)
    parser.add_argument('--meta_batch_size', type=int, default=8)
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--eval_interval', type=int, default=50)

    parser.add_argument('--test_episodes', type=int, default=600,
                        help='最终Test评估的episode数量（默认600）')

    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--augment', action='store_true', default=True)
    parser.add_argument('--no_augment', action='store_true')

    parser.add_argument('--data_root', type=str, default='./CIFAR-FS')
    parser.add_argument('--save_dir', type=str, default='./results/cifar_fs')

    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 1024])

    parser.add_argument('--track_updates', action='store_true', default=False)
    parser.add_argument('--track_dir', type=str, default='./update_tracking/cifar_fs')
    parser.add_argument('--track_interval', type=int, default=50)

    return parser.parse_args()


def update_config(args, seed):
    """配置更新"""
    if hasattr(Config, 'set_dataset'):
        Config.set_dataset("cifar_fs")
    else:
        Config.DATASET = "cifar_fs"
        if hasattr(Config, 'CIFAR_FS_ROOT'):
            Config.DATA_ROOT = Config.CIFAR_FS_ROOT
        else:
            Config.DATA_ROOT = args.data_root

    Config.N_WAY = args.n_way
    Config.K_SHOT = args.k_shot
    Config.K_QUERY = args.k_query
    Config.MAX_EPISODES = args.episodes
    Config.EVAL_INTERVAL = args.eval_interval

    if hasattr(Config, 'CIFAR_FS_ROOT'):
        Config.CIFAR_FS_ROOT = args.data_root

    Config.SEED = seed

    if hasattr(Config, 'CIFAR_HIDDEN_DIM'):
        Config.CIFAR_HIDDEN_DIM = args.hidden_dim

    if args.no_augment and hasattr(Config, 'CIFAR_AUGMENT'):
        Config.CIFAR_AUGMENT = False

    if args.meta_lr is not None:
        Config.META_LR = args.meta_lr
    if args.meta_batch_size is not None:
        Config.META_BATCH_SIZE = args.meta_batch_size

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
    if not torch.cuda.is_available():
        return 0
    return round(torch.cuda.max_memory_allocated() / 1e6, 1)


def reset_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def evaluate_with_adaptation(model, dataset, n_way, k_shot, k_query, num_episodes=600, inner_steps=5, inner_lr=0.01):
    """
    标准Test评估：每个task重新adapt
    返回0-1范围的小数（与ValAcc格式一致）
    """
    model.train()
    accs = []

    print(f"\n🧪 正在评估 Test Set ({num_episodes} episodes)...")

    for episode in range(num_episodes):
        support_x, support_y, query_x, query_y = dataset.get_task(n_way, k_shot, k_query)
        support_x = support_x.to(Config.DEVICE)
        support_y = support_y.to(Config.DEVICE)
        query_x = query_x.to(Config.DEVICE)
        query_y = query_y.to(Config.DEVICE)

        # 保存meta-params
        original_params = {name: param.clone() for name, param in model.named_parameters()}

        # Inner loop adaptation
        inner_opt = torch.optim.SGD(model.parameters(), lr=inner_lr)
        for _ in range(inner_steps):
            inner_opt.zero_grad()
            logits = model(support_x)
            loss = F.cross_entropy(logits, support_y)
            loss.backward()
            inner_opt.step()

        # 评估
        model.eval()
        with torch.no_grad():
            logits = model(query_x)
            acc = (logits.argmax(dim=1) == query_y).float().mean().item()
            accs.append(acc)
        model.train()

        # 恢复meta-params
        for name, param in model.named_parameters():
            param.data.copy_(original_params[name])

    mean_acc = np.mean(accs)  # 0-1范围
    std_acc = np.std(accs)  # 0-1范围

    print(f"✅ Test Set 结果: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
    return mean_acc, std_acc


def save_detailed_log(save_dir, algorithm, seed, history, display_name, test_results=None):
    """
    保存每100 episodes的详细记录到JSON文件（匹配experiment.txt格式）
    """
    detailed_dir = os.path.join(save_dir, "detailed_logs")
    os.makedirs(detailed_dir, exist_ok=True)

    log_file = os.path.join(detailed_dir, f"{algorithm}_seed{seed}.json")

    # 构建 records（每100轮一条）
    records = []
    eval_interval = getattr(Config, 'EVAL_INTERVAL', 50)

    if history and 'train_loss' in history:
        num_records = len(history['train_loss'])

        for i in range(num_records):
            episode = (i + 1) * eval_interval

            # 严格只保留100的倍数（100, 200, 300...）
            if episode % 100 != 0:
                continue

            record = {
                "episode": int(episode),
                "train_loss": float(history['train_loss'][i]),
                "train_acc": float(history['train_acc'][i]) if i < len(history.get('train_acc', [])) else None,
                "val_acc": float(history['val_acc'][i]) if i < len(history.get('val_acc', [])) else None,
                "avg_steps": float(history.get('avg_steps', [Config.INNER_STEPS] * num_records)[i])
                if 'avg_steps' in history else float(Config.INNER_STEPS)
            }
            records.append(record)

    # 构建与 experiment.txt 一致的 JSON 结构
    detailed_record = {
        "experiment": display_name,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "records": records
    }

    # 添加最终test结果
    if test_results:
        detailed_record["final_test"] = {
            "test_acc": float(test_results.get('final_test_acc', 0)),
            "test_std": float(test_results.get('final_test_std', 0))
        }

    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_record, f, indent=2)

    print(f"💾 详细日志已保存: {log_file} (共 {len(records)} 条记录)")
    return log_file


def run_single_experiment(algorithm, seed, args, save_dir):
    algo_map = {
        'maml': ('maml', False, 'MAML-CIFAR (Baseline)'),
        'maml_fo': ('maml', True, 'FOMAML-CIFAR (First-Order)'),
        'sgd_y_maml': ('sgd_y_maml', False, 'SGD-Y-MAML-CIFAR (Adaptive)'),
        'sgd_y_maml_fo': ('sgd_y_maml', True, 'SGD-Y-FOMAML-CIFAR'),
        'taming_maml': ('taming_maml', False, 'TamingMAML-CIFAR (Regularized)'),
        'es_maml': ('es_maml', False, 'ES-MAML-CIFAR (Evolution Strategy)')
    }

    algo_name, first_order, display_name = algo_map.get(algorithm, (algorithm, False, algorithm))

    print(f"\n{'=' * 80}")
    print(f"🚀 [{display_name}] | CIFAR-FS | Seed {seed}")
    print(f"{'=' * 80}")

    device = update_config(args, seed)
    reset_gpu_memory()
    start_time = time.time()

    # 加载三个数据集
    try:
        print("📂 正在加载数据集...")
        train_dataset = CifarFSDataset(args.data_root, split='train', augment=not args.no_augment)
        val_dataset = CifarFSDataset(args.data_root, split='val', augment=False)
        test_dataset = CifarFSDataset(args.data_root, split='test', augment=False)
        print(
            f"✅ 数据加载: Train{len(train_dataset.classes)}类 / Val{len(val_dataset.classes)}类 / Test{len(test_dataset.classes)}类")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 初始化追踪器相关变量（全部初始化为None）
    tracker = None
    original_inner_loop_maml = None
    original_adapt_sgd_y = None
    trainer = None

    if args.track_updates and algo_name in ['maml', 'maml_fo', 'sgd_y_maml', 'sgd_y_maml_fo']:
        tracker_dir = os.path.join(args.track_dir, f"{algorithm}_seed{seed}")
        tracker = UpdateTracker(tracker_dir, args.track_interval)

        if algo_name in ['maml', 'maml_fo']:
            import algorithms.maml as maml_module
            original_inner_loop_maml = maml_module.inner_loop
            maml_module.inner_loop = create_tracked_inner_loop(original_inner_loop_maml, tracker)

        print(f"📊 参数追踪已启用")

    # 简单的step_callback
    def step_callback(episode, train_loss, train_acc, val_loss, val_acc, avg_steps, best_val_acc, model):
        if tracker:
            tracker._current_episode = episode

    history = None

    try:
        if algo_name == 'maml':
            from algorithms.maml import MAML
            from models.omniglot_net import OmniglotNet, ConvBlock

            class CIFARWrapper(OmniglotNet):
                def __init__(self, num_classes=5, hidden_dim=128):
                    super().__init__(num_classes, hidden_dim)
                    self.layer1 = ConvBlock(3, hidden_dim)
                    self.fc = torch.nn.Linear(hidden_dim * 4 * 4, num_classes)

            model = CIFARWrapper(num_classes=Config.N_WAY, hidden_dim=args.hidden_dim)
            trainer = MAML(model, Config)

            print(f"配置: 固定{Config.INNER_STEPS}步, lr={Config.INNER_LR}, {'一阶' if first_order else '二阶'}")

            history = trainer.train(train_dataset, val_dataset, first_order=first_order, step_callback=step_callback)

            # 训练完成后，使用最佳模型进行Test评估
            print(f"\n{'=' * 60}")
            print(f"🏁 训练完成 | 最佳Val: {max(history['val_acc']):.2%}")
            print(f"🧪 正在评估 Test Set...")
            print(f"{'=' * 60}")

            # 加载验证集最佳模型状态
            if hasattr(trainer, 'best_state') and trainer.best_state is not None:
                model.load_state_dict(trainer.best_state)
                print("✅ 已加载验证集最佳模型")

            test_acc, test_std = evaluate_with_adaptation(
                model, test_dataset, Config.N_WAY, Config.K_SHOT, Config.K_QUERY,
                num_episodes=args.test_episodes,
                inner_steps=Config.INNER_STEPS,
                inner_lr=Config.INNER_LR
            )

            test_results = {
                'final_test_acc': test_acc,  # 0-1小数
                'final_test_std': test_std,  # 0-1小数
                'n_way': Config.N_WAY,
                'k_shot': Config.K_SHOT
            }

        elif algo_name == 'taming_maml':
            from algorithms.taming_maml import TamingMAML
            from models.omniglot_net import OmniglotNet, ConvBlock

            class CIFARWrapper(OmniglotNet):
                def __init__(self, num_classes=5, hidden_dim=128):
                    super().__init__(num_classes, hidden_dim)
                    self.layer1 = ConvBlock(3, hidden_dim)
                    self.fc = torch.nn.Linear(hidden_dim * 4 * 4, num_classes)

            model = CIFARWrapper(num_classes=Config.N_WAY, hidden_dim=args.hidden_dim)
            trainer = TamingMAML(model, Config)
            print(f"配置: α_reg={Config.TAMING_ALPHA_REG}, grad_clip={Config.TAMING_GRAD_CLIP}")

            history = trainer.train(train_dataset, val_dataset, first_order=first_order)

            # Test评估
            print(f"\n🏁 训练完成 | 最佳Val: {max(history['val_acc']):.2%}")
            print(f"🧪 正在评估 Test Set...")
            test_acc, test_std = evaluate_with_adaptation(
                model, test_dataset, Config.N_WAY, Config.K_SHOT, Config.K_QUERY,
                num_episodes=args.test_episodes
            )
            test_results = {'final_test_acc': test_acc, 'final_test_std': test_std}

        elif algo_name == 'es_maml':
            from algorithms.es_maml import ESMAML
            from models.omniglot_net import OmniglotNet, ConvBlock

            class CIFARWrapper(OmniglotNet):
                def __init__(self, num_classes=5, hidden_dim=128):
                    super().__init__(num_classes, hidden_dim)
                    self.layer1 = ConvBlock(3, hidden_dim)
                    self.fc = torch.nn.Linear(hidden_dim * 4 * 4, num_classes)

            model = CIFARWrapper(num_classes=Config.N_WAY, hidden_dim=args.hidden_dim)
            original_meta_lr = Config.META_LR
            Config.META_LR = Config.ES_META_LR
            trainer = ESMAML(model, Config)
            Config.META_LR = original_meta_lr

            print(f"配置: σ={Config.ES_SIGMA}, 扰动数={Config.ES_N_PERTURBATIONS}")

            history = trainer.train(train_dataset, val_dataset)

            # Test评估
            print(f"\n🏁 训练完成 | 最佳Val: {max(history['val_acc']):.2%}")
            print(f"🧪 正在评估 Test Set...")
            test_acc, test_std = evaluate_with_adaptation(
                model, test_dataset, Config.N_WAY, Config.K_SHOT, Config.K_QUERY,
                num_episodes=args.test_episodes
            )
            test_results = {'final_test_acc': test_acc, 'final_test_std': test_std}

        elif algo_name in ['sgd_y_maml', 'sgd_y_maml_fo']:
            from algorithms.sgd_y_maml import SGD_Y_MAML
            from models.omniglot_net import OmniglotNet, ConvBlock

            class CIFARWrapper(OmniglotNet):
                def __init__(self, num_classes=5, hidden_dim=128):
                    super().__init__(num_classes, hidden_dim)
                    self.layer1 = ConvBlock(3, hidden_dim)
                    self.fc = torch.nn.Linear(hidden_dim * 4 * 4, num_classes)

            model = CIFARWrapper(num_classes=Config.N_WAY, hidden_dim=args.hidden_dim)

            ablation_cfg = {
                'USE_LR_DECAY': True,
                'USE_REGULARIZATION': True,
                'SGDY_STATISTICAL_WEIGHT': Config.SGDY_STATISTICAL_WEIGHT,
                'SGDY_GRADIENT_SMOOTH_WEIGHT': Config.SGDY_GRADIENT_SMOOTH_WEIGHT
            }
            trainer = SGD_Y_MAML(model, Config, ablation_config=ablation_cfg)

            print(f"配置: 自适应步数(max={Config.SGDY_INNER_STEPS_MAX})")

            if args.track_updates and tracker:
                original_adapt_sgd_y = trainer.adapt
                trainer.adapt = create_tracked_adapt(original_adapt_sgd_y, tracker, trainer.model)

            history = trainer.train(train_dataset, val_dataset, first_order=first_order,
                                    step_callback=step_callback if tracker else None)

            # Test评估
            print(f"\n🏁 训练完成 | 最佳Val: {max(history['val_acc']):.2%}")
            print(f"🧪 正在评估 Test Set...")
            test_acc, test_std = evaluate_with_adaptation(
                model, test_dataset, Config.N_WAY, Config.K_SHOT, Config.K_QUERY,
                num_episodes=args.test_episodes
            )
            test_results = {'final_test_acc': test_acc, 'final_test_std': test_std}

        else:
            print(f"❌ 算法 {algo_name} 暂未实现")
            return None

    except Exception as e:
        print(f"❌ 训练过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        if original_inner_loop_maml is not None:
            import algorithms.maml as maml_module
            maml_module.inner_loop = original_inner_loop_maml

        if original_adapt_sgd_y is not None and trainer is not None:
            trainer.adapt = original_adapt_sgd_y

        if tracker:
            tracker.close()

    if history is None:
        print("❌ 训练失败，无历史记录")
        return None

    # 保存每seed的JSON文件（每100轮一条，匹配experiment.txt格式）
    detailed_log_path = save_detailed_log(save_dir, algorithm, seed, history, display_name, test_results)
    print(f"💾 详细日志已保存: {detailed_log_path}")

    gpu_mem_peak = get_gpu_memory()
    elapsed_time = time.time() - start_time

    result = {
        'algorithm': algorithm,
        'display_name': display_name,
        'seed': seed,
        'dataset': 'cifar_fs',
        'first_order': first_order,
        'best_val_acc': max(history['val_acc']) if history.get('val_acc') else 0.0,
        'final_test_acc': test_results['final_test_acc'],
        'final_test_std': test_results['final_test_std'],
        'final_train_loss': history['train_loss'][-1] if history['train_loss'] else 0.0,
        'final_train_acc': history['train_acc'][-1] if history['train_acc'] else 0.0,
        'avg_steps': np.mean(
            history.get('avg_steps', [Config.INNER_STEPS])) if 'avg_steps' in history else Config.INNER_STEPS,
        'gpu_memory_mb': gpu_mem_peak,
        'elapsed_time': elapsed_time / 60,
        'history': history,
        'detailed_log': detailed_log_path,
        'update_tracking_csv': tracker.csv_path if tracker else None
    }

    print(
        f"\n✅ 完成: ValAcc={result['best_val_acc']:.2%}, TestAcc={result['final_test_acc']:.2%}, Time={result['elapsed_time']:.1f}min")
    return result


def save_all_results(all_results, args, save_dir):
    """
    保存所有算法的聚合结果（格式与Omniglot_train完全一致）
    """
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
        final_test_accs = [r['final_test_acc'] for r in results]
        final_train_losses = [r['final_train_loss'] for r in results]
        final_train_accs = [r['final_train_acc'] for r in results]
        avg_steps_list = [r['avg_steps'] for r in results]
        gpu_memories = [r['gpu_memory_mb'] for r in results]
        elapsed_times = [r['elapsed_time'] for r in results]

        def mean_std_str(values, fmt=".2f", is_percentage=False):
            mean_val = np.mean(values)
            std_val = np.std(values)
            if is_percentage:
                # 统一处理：都按0-1小数处理，乘以100显示为百分比
                return f"{mean_val * 100:.2f}±{std_val * 100:.2f}%"
            return f"{mean_val:{fmt}}±{std_val:.2f}"

        display_name = results[0]['display_name']

        summary_rows.append({
            'algorithm': algo,
            'display_name': display_name,
            'n_seeds': len(results),
            'seeds': seeds,
            'ValAcc': mean_std_str(best_val_accs, is_percentage=True),
            'TestAcc': mean_std_str(final_test_accs, is_percentage=True),
            'TrainLoss': mean_std_str(final_train_losses, fmt=".3f"),
            'TrainAcc': mean_std_str(final_train_accs, is_percentage=True),
            'Steps': mean_std_str(avg_steps_list, fmt=".1f"),
            'GPU_MB': mean_std_str(gpu_memories, fmt=".1f"),
            'Time': mean_std_str(elapsed_times, fmt=".1f"),
            'raw': {
                'best_val_acc': best_val_accs,
                'final_test_acc': final_test_accs,
                'final_train_loss': final_train_losses,
                'final_train_acc': final_train_accs,
                'avg_steps': avg_steps_list,
                'gpu_memory_mb': gpu_memories,
                'elapsed_time': elapsed_times
            }
        })

    # 打印表格（与Omniglot_train完全一致）
    print(f"\n{'=' * 110}")
    print("CIFAR-FS 完整对比表格（多种算法 × 多 seeds）")
    print(f"{'=' * 110}")
    print(
        f"{'实验':<25} | {'ValAcc':<12} | {'TestAcc':<12} | {'TrainLoss':<10} | {'TrainAcc':<12} | {'步数':<8} | {'显存(MB)':<12} | {'时间(min)':<10}"
    )
    print("-" * 110)

    for row in summary_rows:
        print(f"{row['display_name']:<25} | {row['ValAcc']:<12} | {row['TestAcc']:<12} | "
              f"{row['TrainLoss']:<10} | {row['TrainAcc']:<12} | "
              f"{row['Steps']:<8} | {row['GPU_MB']:<12} | {row['Time']:<10}")

    print(f"{'=' * 110}")
    print(f"总实验数: {len(all_results)}")

    # 保存CSV（汇总）
    csv_summary = os.path.join(save_dir, f'summary_{timestamp}.csv')
    with open(csv_summary, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['algorithm', 'display_name', 'n_seeds', 'ValAcc', 'TestAcc', 'TrainLoss',
                         'TrainAcc', 'Steps', 'GPU_MB', 'Time_min'])
        for row in summary_rows:
            writer.writerow([
                row['algorithm'],
                row['display_name'],
                row['n_seeds'],
                row['ValAcc'],
                row['TestAcc'],
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
        writer.writerow(['algorithm', 'display_name', 'seed', 'best_val_acc', 'final_test_acc',
                         'final_train_loss', 'final_train_acc', 'avg_steps',
                         'gpu_memory_mb', 'elapsed_time_min', 'detailed_log', 'update_tracking_csv'])
        for r in all_results:
            writer.writerow([
                r['algorithm'],
                r['display_name'],
                r['seed'],
                f"{r['best_val_acc']:.4f}",
                f"{r['final_test_acc']:.4f}",
                f"{r['final_train_loss']:.4f}",
                f"{r['final_train_acc']:.4f}",
                f"{r['avg_steps']:.1f}",
                f"{r['gpu_memory_mb']:.1f}",
                f"{r['elapsed_time']:.2f}",
                r.get('detailed_log', ''),
                r.get('update_tracking_csv', '')
            ])

    # 保存JSON
    json_path = os.path.join(save_dir, f'summary_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump({
            'config': {
                'dataset': 'cifar_fs',
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
    print("CIFAR-FS 批量运行: 多种算法 × 多 seeds 对比实验")
    print(f"算法: {args.algorithms}")
    print(f"Seeds: {args.seeds}")
    print(f"保存目录: {args.save_dir}")
    if args.track_updates:
        print(f"📊 参数追踪: 已启用")
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

    # 关键修复：确保缩进正确
    if len(all_results) == 0:
        print("❌ 没有成功运行的实验")
        return

    # 保存聚合结果（使用 args.save_dir 而非 save_dir）
    save_all_results(all_results, args, args.save_dir)

    print(f"\n{'=' * 80}")
    print(f"全部完成！成功: {len(all_results)}/{total_experiments}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()