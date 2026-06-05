# plot_comparison_4algos.py
# 4算法对比实验：MAML vs FOMAML vs SGD-y-MAML vs SGD-y-FOMAML

import json
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300

# ==================== 路径配置 ====================
base_dir = r'result\full_comparison\detailed_logs'

# ==================== 4算法配置（匹配你的JSON文件名）====================
ALGORITHMS = {
    'maml': {
        'label': 'MAML',
        'color': '#CF221E',  # 红
        'marker': 'o',
        'linestyle': '-',
        'linewidth': 2,
        'markersize': 6,
        'fixed_steps': 5
    },
    'maml_fo': {
        'label': 'FOMAML',
        'color': '#0575BF',  # 蓝
        'marker': 's',
        'linestyle': '--',
        'linewidth': 2,
        'markersize': 6,
        'fixed_steps': 5
    },
    'sgd_y_maml': {
        'label': 'SGD-y-MAML',  # 修改：SGD-Y → SGD-y
        'color': '#74AB29',  # 绿
        'marker': '^',
        'linestyle': '-.',
        'linewidth': 2,
        'markersize': 7,
        'fixed_steps': None
    },
    'sgd_y_maml_fo': {
        'label': 'SGD-y-FOMAML',  # 修改：SGD-Y → SGD-y
        'color': '#702FA8',  # 紫
        'marker': 'v',
        'linestyle': '-',
        'linewidth': 2.5,
        'markersize': 7,
        'fixed_steps': None
    }
}


def load_comparison_data():
    """加载4算法对比数据（包括准确率和损失）"""
    data = {}

    for algo_key, cfg in ALGORITHMS.items():
        pattern = os.path.join(base_dir, f"{algo_key}_seed*.json")
        files = sorted(glob.glob(pattern))

        if not files:
            print(f"⚠️  未找到 {algo_key} 的数据")
            continue

        print(f"✅ {algo_key}: {len(files)} seeds")

        all_records = []
        for f in files:
            with open(f, 'r') as fp:
                content = json.load(fp)
                recs = content.get('records', content) if isinstance(content, dict) else content
                all_records.append(recs)

        if not all_records:
            continue

        episodes = [r['episode'] for r in all_records[0]]
        val_acc = []
        train_loss = []  # 新增：存储训练损失

        for seed_recs in all_records:
            va = []
            tl = []  # 新增：单个seed的损失列表
            for r in seed_recs:
                # 提取准确率（保持原有逻辑）
                acc = r.get('val_acc') or r.get('test_acc') or r.get('accuracy') or 0
                va.append(acc)

                # 新增：提取损失（支持常见字段名）
                loss = r.get('train_loss') or r.get('loss') or r.get('training_loss') or 0
                tl.append(loss)

            val_acc.append(va)
            train_loss.append(tl)

        data[algo_key] = {
            'episodes': episodes,
            'mean': np.mean(val_acc, axis=0),
            'std': np.std(val_acc, axis=0),
            'loss_mean': np.mean(train_loss, axis=0),  # 新增：损失均值
            'loss_std': np.std(train_loss, axis=0),  # 新增：损失标准差
            'config': cfg,
            'final_mean': np.mean([v[-10:] for v in val_acc]),
            'final_std': np.std([v[-10:] for v in val_acc])
        }

    return data


def plot_convergence(data, save_dir):
    """图1: 测试准确率收敛曲线"""
    fig, ax = plt.subplots(figsize=(9, 6))

    for algo_key, d in data.items():
        cfg = d['config']
        ax.plot(d['episodes'], d['mean'],
                label=cfg['label'],
                color=cfg['color'],
                marker=cfg['marker'],
                linestyle=cfg['linestyle'],
                linewidth=cfg['linewidth'],
                markersize=cfg['markersize'],
                markevery=len(d['episodes']) // 10,
                markerfacecolor='white' if cfg['fixed_steps'] else cfg['color'],
                markeredgewidth=1.5)

        ax.fill_between(d['episodes'],
                        d['mean'] - d['std'],
                        d['mean'] + d['std'],
                        alpha=0.15, color=cfg['color'])

    ax.set_xlabel('Episode', fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontweight='bold')
    ax.set_title('Test Accuracy Convergence: 4-Algorithm Comparison', fontweight='bold')
    ax.legend(loc='lower right', frameon=True)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_convergence.svg'), format='svg', bbox_inches='tight')
    print(f"Saved: comparison_convergence.svg")
    plt.close()


def plot_loss_convergence(data, save_dir):
    """图2: 训练损失收敛曲线（新增）"""
    fig, ax = plt.subplots(figsize=(9, 6))

    for algo_key, d in data.items():
        cfg = d['config']
        # 绘制损失曲线（注意：损失通常递减，所以不需要fill_between的上下颠倒）
        ax.plot(d['episodes'], d['loss_mean'],
                label=cfg['label'],
                color=cfg['color'],
                marker=cfg['marker'],
                linestyle=cfg['linestyle'],
                linewidth=cfg['linewidth'],
                markersize=cfg['markersize'],
                markevery=len(d['episodes']) // 10,
                markerfacecolor='white' if cfg['fixed_steps'] else cfg['color'],
                markeredgewidth=1.5)

        # 添加标准差阴影
        ax.fill_between(d['episodes'],
                        d['loss_mean'] - d['loss_std'],
                        d['loss_mean'] + d['loss_std'],
                        alpha=0.15, color=cfg['color'])

    ax.set_xlabel('Episode', fontweight='bold')
    ax.set_ylabel('Training Loss', fontweight='bold')
    ax.set_title('Training Loss Convergence: 4-Algorithm Comparison', fontweight='bold')
    ax.legend(loc='upper right', frameon=True)  # 损失曲线通常legend放右上
    ax.grid(True, alpha=0.3, linestyle='--')

    # 建议：如果损失下降幅度很大，可以考虑使用对数坐标
    # ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_loss.svg'), format='svg', bbox_inches='tight')
    print(f"Saved: comparison_loss.svg")
    plt.close()


def plot_final_bars(data, save_dir):
    """图3: 最终测试准确率对比柱状图"""
    fig, ax = plt.subplots(figsize=(9, 6))

    # 按性能排序（从高到低）：SGD-y-MAML ≈ SGD-y-FOMAML > FOMAML > MAML
    order = ['sgd_y_maml', 'sgd_y_maml_fo', 'maml_fo', 'maml']

    labels = []
    means = []
    stds = []
    colors = []

    for algo_key in order:
        if algo_key not in data:
            continue
        d = data[algo_key]
        cfg = d['config']
        labels.append(cfg['label'])
        means.append(d['final_mean'])
        stds.append(d['final_std'])
        colors.append(cfg['color'])

    y_pos = np.arange(len(labels))

    # 窄条形 (height=0.5)
    bars = ax.barh(y_pos, means, height=0.5,
                   xerr=stds, color=colors,
                   edgecolor='black', linewidth=1.5, capsize=4, alpha=0.9)

    # 数值标签
    for i, (m, s) in enumerate(zip(means, stds)):
        x_pos = m + s + 0.003
        if x_pos > 0.968:  # 如果太靠右，放左边
            x_pos = m - 0.008
            ha, color = 'right', 'white'
        else:
            ha, color = 'left', 'black'

        ax.text(x_pos, i, f'{m * 100:.2f}%',
                va='center', ha=ha, fontsize=10, fontweight='bold', color=color)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Test Accuracy', fontweight='bold')
    ax.set_title('Final Test Accuracy Comparison', fontweight='bold')
    ax.set_xlim([0.80, 0.98])
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.set_ylim([-0.5, len(labels) - 0.5])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_final_bars.svg'), format='svg', bbox_inches='tight')
    print(f"Saved: comparison_final_bars.svg")
    plt.close()


def main():
    if not os.path.exists(base_dir):
        print(f"Path not found: {base_dir}")
        return

    data = load_comparison_data()
    if not data:
        return

    save_dir = os.path.join(os.path.dirname(os.path.dirname(base_dir)), 'figures_comparison')
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nSaving to: {save_dir}\n")

    plot_convergence(data, save_dir)
    plot_loss_convergence(data, save_dir)  # 新增：绘制损失曲线
    plot_final_bars(data, save_dir)

    print("\nDone! Three figures saved.")  # 更新计数：2→3


if __name__ == "__main__":
    main()