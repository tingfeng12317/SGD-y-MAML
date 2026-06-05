# plot_ablation_final.py
# 四图完整版：abcd命名（Test Acc, Training Loss, Adaptive Steps, Final Comparison）

import json
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300

base_dir = r'result\ablation_b3_drift_0.015\detailed_logs'

# ==================== 最终实验数据（用于柱状图d）====================
FINAL_RESULTS = {
    'B1_Baseline': {
        'val_acc': 0.9407,
        'val_acc_std': 0.0020,
        'label': '1: Baseline (Fixed 5 steps)'
    },
    'B2_Momentum': {
        'val_acc': 0.9582,
        'val_acc_std': 0.0044,
        'label': '2: +Momentum (Fixed 5 steps)'
    },
    'B3_Dual_Convergence': {
        'val_acc': 0.9640,
        'val_acc_std': 0.0019,
        'label': '3: +Dual Convergence'
    },
    'B4_LR_Decay': {
        'val_acc': 0.9646,
        'val_acc_std': 0.0005,
        'label': '4: +LR Decay'
    },
    'B5_Full': {
        'val_acc': 0.9646,
        'val_acc_std': 0.0005,
        'label': '5: Full System(SGD-y-MAML)'
    }
}

# ==================== 配色方案 ====================
COLORS = ['#CF221E', '#0575BF', '#EDB11F', '#702FA8', '#74AB29']

EXPERIMENTS = {
    'B1_Baseline': {
        'label': '1: Baseline (Fixed)',
        'color': COLORS[0],
        'marker': 'o',
        'linestyle': '-',
        'linewidth': 2,
        'markersize': 6,
        'fixed_steps': 5
    },
    'B2_Momentum': {
        'label': '2: +Momentum (Fixed)',
        'color': COLORS[1],
        'marker': 's',
        'linestyle': '--',
        'linewidth': 2,
        'markersize': 6,
        'fixed_steps': 5
    },
    'B3_Dual_Convergence': {
        'label': '3: +Dual Convergence',
        'color': COLORS[2],
        'marker': '^',
        'linestyle': '-.',
        'linewidth': 2,
        'markersize': 7,
        'fixed_steps': None
    },
    'B4_LR_Decay': {
        'label': '4: +LR Decay',
        'color': COLORS[3],
        'marker': 'D',
        'linestyle': ':',
        'linewidth': 2.5,
        'markersize': 6,
        'fixed_steps': None
    },
    'B5_Full': {
        'label': '5: Full System',
        'color': COLORS[4],
        'marker': 'v',
        'linestyle': '-',
        'linewidth': 2.5,
        'markersize': 7,
        'fixed_steps': None
    }
}


def load_data():
    """读取曲线数据（包含loss）"""
    data = {}
    for exp_key, cfg in EXPERIMENTS.items():
        files = sorted(glob.glob(os.path.join(base_dir, f"{exp_key}_seed*.json")))
        if not files:
            continue

        all_records = []
        for f in files:
            with open(f, 'r') as fp:
                content = json.load(fp)
                recs = content.get('records', content) if isinstance(content, dict) else content
                all_records.append(recs)

        if not all_records:
            continue

        episodes = [r['episode'] for r in all_records[0]]
        val_acc, avg_steps, train_loss = [], [], []

        for seed_recs in all_records:
            va, st, tl = [], [], []
            for r in seed_recs:
                va.append(r.get('val_acc', 0) or 0)
                step = r.get('avg_steps')
                st.append(step if step is not None else (cfg['fixed_steps'] or 5))
                tl.append(r.get('train_loss', 0) or 0)
            val_acc.append(va)
            avg_steps.append(st)
            train_loss.append(tl)

        data[exp_key] = {
            'episodes': episodes,
            'val_acc_mean': np.mean(val_acc, axis=0),
            'val_acc_std': np.std(val_acc, axis=0),
            'avg_steps_mean': np.mean(avg_steps, axis=0),
            'avg_steps_std': np.std(avg_steps, axis=0),
            'train_loss_mean': np.mean(train_loss, axis=0),
            'train_loss_std': np.std(train_loss, axis=0),
            'config': cfg
        }
    return data


def plot_a_test_accuracy(data, save_dir):
    """(a) 测试准确率收敛曲线"""
    fig, ax = plt.subplots(figsize=(9, 6))

    for exp_key, d in data.items():
        cfg = d['config']
        ax.plot(d['episodes'], d['val_acc_mean'],
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
                        d['val_acc_mean'] - d['val_acc_std'],
                        d['val_acc_mean'] + d['val_acc_std'],
                        alpha=0.15, color=cfg['color'])

    ax.set_ylim([0.88, 0.98])
    ax.set_xlabel('Episode', fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontweight='bold')
    ax.set_title('(a) Test Accuracy Convergence', fontweight='bold')
    ax.legend(loc='lower right', frameon=True)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation_a_test_accuracy.svg'), format='svg', bbox_inches='tight')
    print(f"Saved: (a) ablation_a_test_accuracy.svg")
    plt.close()


def plot_b_training_loss(data, save_dir):
    """(b) 训练损失收敛曲线（新增）"""
    fig, ax = plt.subplots(figsize=(9, 6))

    for exp_key, d in data.items():
        cfg = d['config']
        ax.plot(d['episodes'], d['train_loss_mean'],
                label=cfg['label'],
                color=cfg['color'],
                marker=cfg['marker'],
                linestyle=cfg['linestyle'],
                linewidth=cfg['linewidth'],
                markersize=cfg['markersize'],
                markevery=len(d['episodes']) // 10,
                markerfacecolor='white' if cfg['fixed_steps'] else cfg['color'],
                markeredgewidth=1.5)

        # 误差阴影
        ax.fill_between(d['episodes'],
                        np.maximum(d['train_loss_mean'] - d['train_loss_std'], 0),
                        d['train_loss_mean'] + d['train_loss_std'],
                        alpha=0.15, color=cfg['color'])

    # 不优化Y轴，使用自动范围或固定0-2.0（根据数据自动调整）
    ax.set_ylim(bottom=0)  # 只设置下限为0，上限自动

    ax.set_xlabel('Episode', fontweight='bold')
    ax.set_ylabel('Training Loss', fontweight='bold')
    ax.set_title('(b) Training Loss Convergence', fontweight='bold')
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation_b_training_loss.svg'), format='svg', bbox_inches='tight')
    print(f"Saved: (b) ablation_b_training_loss.svg")
    plt.close()


def plot_c_adaptive_steps(data, save_dir):
    """(c) 内循环自适应步数"""
    fig, ax = plt.subplots(figsize=(9, 6))

    for ek in ['B1_Baseline', 'B2_Momentum']:
        if ek in data:
            cfg = data[ek]['config']
            ax.axhline(y=5, color=cfg['color'], linestyle='--',
                       linewidth=2, alpha=0.7, label=f"{cfg['label']}")

    for ek in ['B3_Dual_Convergence', 'B4_LR_Decay', 'B5_Full']:
        if ek not in data:
            continue
        d = data[ek]
        cfg = d['config']
        ax.plot(d['episodes'], d['avg_steps_mean'],
                label=cfg['label'],
                color=cfg['color'],
                marker=cfg['marker'],
                linestyle=cfg['linestyle'],
                linewidth=cfg['linewidth'],
                markersize=cfg['markersize'],
                markevery=len(d['episodes']) // 10,
                markerfacecolor=cfg['color'])

        ax.fill_between(d['episodes'],
                        d['avg_steps_mean'] - d['avg_steps_std'],
                        d['avg_steps_mean'] + d['avg_steps_std'],
                        alpha=0.15, color=cfg['color'])

    ax.set_ylim([0, 22])
    ax.set_xlabel('Episode', fontweight='bold')
    ax.set_ylabel('Average Inner Loop Steps', fontweight='bold')
    ax.set_title('(c) Adaptive Inner Loop Steps', fontweight='bold')
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation_c_adaptive_steps.svg'), format='svg', bbox_inches='tight')
    print(f"Saved: (c) ablation_c_adaptive_steps.svg")
    plt.close()


def plot_d_final_comparison(save_dir):
    """(d) 最终测试准确率对比 - 调整比例和条形宽度"""
    # 修改：figsize 改为 (9,6) 与其他三张图一致
    fig, ax = plt.subplots(figsize=(9, 6))

    order = ['B5_Full', 'B4_LR_Decay', 'B3_Dual_Convergence', 'B2_Momentum', 'B1_Baseline']

    labels = []
    means = []
    stds = []
    colors = []

    for ek in order:
        res = FINAL_RESULTS[ek]
        labels.append(res['label'])
        means.append(res['val_acc'])
        stds.append(res['val_acc_std'])
        colors.append(EXPERIMENTS[ek]['color'])

    y_pos = np.arange(len(labels))

    # 修改：添加 height=0.5 使条形变窄（默认0.8），增加条形间距
    bars = ax.barh(y_pos, means, height=0.5,  # 调窄条形
                   xerr=stds,
                   color=colors,
                   edgecolor='black',
                   linewidth=1.5,
                   capsize=4,
                   alpha=0.9)

    # 数值标签位置调整
    for i, (m, s) in enumerate(zip(means, stds)):
        x_pos = m + s + 0.002
        if x_pos > 0.963:
            x_pos = m - 0.008  # 稍微调整位置
            ha = 'right'
            color = 'white'
        else:
            ha = 'left'
            color = 'black'

        ax.text(x_pos, i, f'{m * 100:.2f}%',
                va='center', ha=ha, fontsize=10, fontweight='bold', color=color)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Test Accuracy', fontweight='bold')
    ax.set_title('(d) Final Test Accuracy Comparison', fontweight='bold')
    ax.set_xlim([0.90, 0.975])
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')

    # 调整y轴范围，使条形分布更美观（因为条形变窄了，需要调整边界）
    ax.set_ylim([-0.5, len(labels) - 0.5])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation_d_final_comparison.svg'),
                format='svg', bbox_inches='tight')
    print(f"Saved: (d) ablation_d_final_comparison.svg (9×6比例, 窄条形)")
    plt.close()

def main():
    if not os.path.exists(base_dir):
        print(f"Path not found: {base_dir}")
        return

    data = load_data()
    save_dir = os.path.join(os.path.dirname(base_dir), 'figures_abcd')
    os.makedirs(save_dir, exist_ok=True)

    print(f"Saving ablation figures (a-d) to: {save_dir}\n")

    plot_a_test_accuracy(data, save_dir)
    plot_b_training_loss(data, save_dir)  # 新增
    plot_c_adaptive_steps(data, save_dir)
    plot_d_final_comparison(save_dir)

    print("\nAll ablation figures (a-d) saved!")


if __name__ == "__main__":
    main()