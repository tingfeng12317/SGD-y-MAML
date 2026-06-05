# ablation_analysis.py
"""
消融实验结果分析 - 模块3
功能：读取实验结果，生成统计报告，绘制对比图表
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from ablation_config import AblationConfig

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class AblationAnalyzer:
    """
    消融实验结果分析器
    """

    def __init__(self, results_dir: str = "./results/ablation"):
        self.results_dir = results_dir
        self.logs_dir = os.path.join(results_dir, "logs")
        self.figures_dir = os.path.join(results_dir, "figures")

        # 确保输出目录存在
        os.makedirs(self.figures_dir, exist_ok=True)

        # 加载所有结果
        self.results = self._load_all_results()

    def _load_all_results(self) -> List[Dict]:
        """从logs目录加载所有实验结果"""
        results = []

        if not os.path.exists(self.logs_dir):
            print(f"警告: 日志目录不存在 {self.logs_dir}")
            return results

        for json_file in Path(self.logs_dir).glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if data.get('status') == 'success':
                        results.append(data)
            except Exception as e:
                print(f"加载失败 {json_file}: {e}")

        print(f"✅ 成功加载 {len(results)} 个实验结果")
        return results

    def _group_by_experiment(self) -> Dict[str, List[Dict]]:
        """按实验类型分组"""
        groups = defaultdict(list)
        for r in self.results:
            exp = r.get('experiment', 'unknown')
            groups[exp].append(r)
        return dict(groups)

    def _group_by_variant(self, exp_results: List[Dict]) -> Dict:
        """按变体分组，计算统计量"""
        variants = defaultdict(list)

        for r in exp_results:
            key = r.get('variant_desc', 'unknown')
            variants[key].append(r)

        # 计算每个变体的统计量
        stats = {}
        for desc, runs in variants.items():
            accs = [r['best_val_acc'] for r in runs]
            times = [r['train_time_minutes'] for r in runs]

            stats[desc] = {
                'n_runs': len(runs),
                'mean_acc': np.mean(accs),
                'std_acc': np.std(accs),
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'all_accs': accs,
                'all_times': times
            }

        return stats

    def generate_summary_table(self) -> str:
        """生成Markdown格式的汇总表格"""
        groups = self._group_by_experiment()

        lines = []
        lines.append("# SGD-Y-MAML 消融实验结果汇总\n")
        lines.append(f"**总实验数**: {len(self.results)}\n")
        lines.append(f"**生成时间**: {self._get_timestamp()}\n")
        lines.append("---\n")

        for exp_id in ['A1', 'A2', 'A3', 'A4']:
            if exp_id not in groups:
                continue

            lines.append(f"\n## 实验 {exp_id}\n")

            # 获取该实验的变体统计
            variant_stats = self._group_by_variant(groups[exp_id])

            # 表头
            lines.append("| 变体 | 运行数 | 平均准确率 | 标准差 | 平均时间(分) | 时间标准差 |")
            lines.append("|------|--------|-----------|--------|-------------|-----------|")

            # 按准确率排序
            sorted_variants = sorted(
                variant_stats.items(),
                key=lambda x: x[1]['mean_acc'],
                reverse=True
            )

            for desc, stat in sorted_variants:
                lines.append(
                    f"| {desc} | {stat['n_runs']} | "
                    f"{stat['mean_acc']:.2%} | {stat['std_acc']:.2%} | "
                    f"{stat['mean_time']:.1f} | {stat['std_time']:.1f} |"
                )

            # 最佳变体
            best = sorted_variants[0]
            lines.append(f"\n**最佳变体**: {best[0]} "
                         f"(准确率: {best[1]['mean_acc']:.2%} ± {best[1]['std_acc']:.2%})")

        return "\n".join(lines)

    def plot_experiment_comparison(self, save: bool = True):
        """
        绘制四个实验的对比图（2x2子图）
        """
        groups = self._group_by_experiment()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('SGD-Y-MAML Ablation Study Results', fontsize=14, fontweight='bold')

        experiments = [
            ('A1', 'Max Inner Steps', axes[0, 0]),
            ('A2', 'Decay Trigger K', axes[0, 1]),
            ('A3', 'Noise Floor Threshold', axes[1, 0]),
            ('A4', 'Regularization Weights', axes[1, 1])
        ]

        for exp_id, title, ax in experiments:
            if exp_id not in groups:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                ax.set_title(title)
                continue

            # 获取变体统计
            variant_stats = self._group_by_variant(groups[exp_id])

            # 准备数据
            labels = []
            means = []
            stds = []

            # 按特定顺序排序
            if exp_id == 'A1':
                # 按steps数值排序
                sorted_items = sorted(variant_stats.items(),
                                      key=lambda x: self._extract_number(x[0]))
            elif exp_id == 'A2':
                sorted_items = sorted(variant_stats.items(),
                                      key=lambda x: self._extract_number(x[0]))
            elif exp_id == 'A3':
                sorted_items = sorted(variant_stats.items(),
                                      key=lambda x: self._extract_number(x[0]))
            else:
                # A4保持原顺序
                sorted_items = list(variant_stats.items())

            for desc, stat in sorted_items:
                # 简化标签
                label = self._simplify_label(desc, exp_id)
                labels.append(label)
                means.append(stat['mean_acc'] * 100)  # 转为百分比
                stds.append(stat['std_acc'] * 100)

            # 绘制柱状图
            x_pos = np.arange(len(labels))
            bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                          color='steelblue', alpha=0.7, edgecolor='black')

            # 添加数值标签
            for i, (m, s) in enumerate(zip(means, stds)):
                ax.text(i, m + s + 1, f'{m:.1f}±{s:.1f}',
                        ha='center', va='bottom', fontsize=8)

            ax.set_ylabel('Validation Accuracy (%)', fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
            ax.set_ylim(0, 105)
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save:
            save_path = os.path.join(self.figures_dir, "ablation_summary.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 图表已保存: {save_path}")

        return fig

    def plot_time_accuracy_tradeoff(self, save: bool = True):
        """
        绘制时间-准确率权衡图（散点图）
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # 按实验类型着色
        colors = {'A1': 'red', 'A2': 'blue', 'A3': 'green', 'A4': 'purple'}
        markers = {'A1': 'o', 'A2': 's', 'A3': '^', 'A4': 'D'}

        for result in self.results:
            exp = result.get('experiment')
            acc = result['best_val_acc'] * 100
            time = result['train_time_minutes']

            ax.scatter(time, acc,
                       c=colors.get(exp, 'gray'),
                       marker=markers.get(exp, 'o'),
                       s=100, alpha=0.6, edgecolors='black', linewidth=0.5,
                       label=f"{exp}: {result.get('variant_desc', '')[:20]}")

        # 去重图例
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),
                  loc='lower right', fontsize=8, ncol=2)

        ax.set_xlabel('Training Time (minutes)', fontsize=11)
        ax.set_ylabel('Validation Accuracy (%)', fontsize=11)
        ax.set_title('Time-Accuracy Tradeoff', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(70, 100)

        plt.tight_layout()

        if save:
            save_path = os.path.join(self.figures_dir, "time_accuracy_tradeoff.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 图表已保存: {save_path}")

        return fig

    def plot_learning_curves(self, experiment: str = None, save: bool = True):
        """
        绘制学习曲线（验证准确率随episode变化）
        """
        # 筛选结果
        filtered = self.results
        if experiment:
            filtered = [r for r in self.results if r.get('experiment') == experiment]

        if not filtered:
            print("无可用数据绘制学习曲线")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        # 按变体分组，绘制平均曲线
        from collections import defaultdict
        variant_curves = defaultdict(list)

        for r in filtered:
            desc = r.get('variant_desc', 'unknown')
            val_acc = r.get('history', {}).get('val_acc', [])
            if val_acc:
                variant_curves[desc].append(val_acc)

        # 绘制每个变体的平均曲线
        for desc, curves in variant_curves.items():
            # 找到最短长度，对齐
            min_len = min(len(c) for c in curves)
            aligned = [c[:min_len] for c in curves]

            # 计算平均和标准差
            mean_curve = np.mean(aligned, axis=0)
            std_curve = np.std(aligned, axis=0)

            # episode数
            eval_interval = AblationConfig.ABLATION_EVAL_INTERVAL
            episodes = [(i + 1) * eval_interval for i in range(len(mean_curve))]

            # 简化标签
            short_label = self._simplify_label(desc, experiment or 'A1')

            ax.plot(episodes, np.array(mean_curve) * 100,
                    label=short_label, linewidth=2)
            ax.fill_between(episodes,
                            (mean_curve - std_curve) * 100,
                            (mean_curve + std_curve) * 100,
                            alpha=0.2)

        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Validation Accuracy (%)', fontsize=11)
        ax.set_title(f'Learning Curves ({experiment or "All"})',
                     fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            exp_suffix = f"_{experiment}" if experiment else "_all"
            save_path = os.path.join(self.figures_dir, f"learning_curves{exp_suffix}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 图表已保存: {save_path}")

        return fig

    def generate_recommendation(self) -> str:
        """基于结果生成配置建议"""
        groups = self._group_by_experiment()

        recommendations = []
        recommendations.append("# 配置优化建议\n")

        # A1: 最佳步数
        if 'A1' in groups:
            stats = self._group_by_variant(groups['A1'])
            best = max(stats.items(), key=lambda x: x[1]['mean_acc'])
            rec = (f"**最大内循环步数**: 建议选择 **{self._extract_number(best[0])}** "
                   f"(准确率: {best[1]['mean_acc']:.2%}, "
                   f"时间: {best[1]['mean_time']:.1f}分钟)")
            recommendations.append(rec)

        # A2: 最佳K
        if 'A2' in groups:
            stats = self._group_by_variant(groups['A2'])
            best = max(stats.items(), key=lambda x: x[1]['mean_acc'])
            rec = (f"\n**触发衰减步数K**: 建议选择 **{self._extract_number(best[0])}** "
                   f"(准确率: {best[1]['mean_acc']:.2%})")
            recommendations.append(rec)

        # A3: 最佳eps
        if 'A3' in groups:
            stats = self._group_by_variant(groups['A3'])
            best = max(stats.items(), key=lambda x: x[1]['mean_acc'])
            rec = (f"\n**噪声地板阈值**: 建议选择 **{self._extract_number(best[0])}** "
                   f"(准确率: {best[1]['mean_acc']:.2%})")
            recommendations.append(rec)

        # A4: 最佳正则权重
        if 'A4' in groups:
            stats = self._group_by_variant(groups['A4'])
            best = max(stats.items(), key=lambda x: x[1]['mean_acc'])
            # 提取两个权重
            rec = (f"\n**正则项权重**: 建议使用 **{best[0]}** "
                   f"(准确率: {best[1]['mean_acc']:.2%})")
            recommendations.append(rec)

        # 综合建议
        recommendations.append("\n---\n")
        recommendations.append("### 综合优化配置\n")
        recommendations.append("```python\n")
        recommendations.append("# 基于消融实验的最佳配置\n")

        if 'A1' in groups:
            best_a1 = max(self._group_by_variant(groups['A1']).items(),
                          key=lambda x: x[1]['mean_acc'])
            recommendations.append(f"SGDY_INNER_STEPS_MAX = {self._extract_number(best_a1[0])}")

        if 'A2' in groups:
            best_a2 = max(self._group_by_variant(groups['A2']).items(),
                          key=lambda x: x[1]['mean_acc'])
            recommendations.append(f"SGDY_K = {self._extract_number(best_a2[0])}")

        if 'A3' in groups:
            best_a3 = max(self._group_by_variant(groups['A3']).items(),
                          key=lambda x: x[1]['mean_acc'])
            recommendations.append(f"SGDY_EPS_MIN = {self._extract_number(best_a3[0])}")

        recommendations.append("```")

        return "\n".join(recommendations)

    def _extract_number(self, s: str) -> float:
        """从字符串中提取数字"""
        import re
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", s)
        if numbers:
            return float(numbers[0])
        return 0

    def _simplify_label(self, desc: str, exp_id: str) -> str:
        """简化标签用于图表"""
        if exp_id == 'A1':
            n = self._extract_number(desc)
            return f"steps={int(n)}"
        elif exp_id == 'A2':
            n = self._extract_number(desc)
            return f"K={int(n)}"
        elif exp_id == 'A3':
            n = self._extract_number(desc)
            return f"ε={n}"
        elif exp_id == 'A4':
            # 提取两个数字
            import re
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", desc)
            if len(nums) >= 2:
                return f"stat={nums[0]},smooth={nums[1]}"
        return desc[:20]

    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def run_full_analysis(self):
        """运行完整分析流程"""
        print("=" * 70)
        print("开始消融实验结果分析")
        print("=" * 70)

        if len(self.results) == 0:
            print("❌ 没有找到实验结果，请先运行 ablation_runner.py")
            return

        # 1. 生成汇总表格
        print("\n1. 生成汇总表格...")
        summary_md = self.generate_summary_table()
        summary_path = os.path.join(self.results_dir, "ablation_report.md")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_md)
        print(f"   报告已保存: {summary_path}")

        # 2. 绘制对比图
        print("\n2. 绘制实验对比图...")
        self.plot_experiment_comparison(save=True)

        # 3. 绘制时间-准确率权衡图
        print("\n3. 绘制时间-准确率权衡图...")
        self.plot_time_accuracy_tradeoff(save=True)

        # 4. 绘制学习曲线（每个实验一张）
        print("\n4. 绘制学习曲线...")
        for exp in ['A1', 'A2', 'A3', 'A4']:
            self.plot_learning_curves(experiment=exp, save=True)

        # 5. 生成配置建议
        print("\n5. 生成配置建议...")
        recommendation = self.generate_recommendation()
        rec_path = os.path.join(self.results_dir, "recommendation.md")
        with open(rec_path, 'w', encoding='utf-8') as f:
            f.write(recommendation)
        print(f"   建议已保存: {rec_path}")

        print("\n" + "=" * 70)
        print("✅ 分析完成！所有结果保存在:", self.results_dir)
        print("=" * 70)
        print("\n生成的文件:")
        print(f"  - {summary_path}")
        print(f"  - {rec_path}")
        print(f"  - {self.figures_dir}/ablation_summary.png")
        print(f"  - {self.figures_dir}/time_accuracy_tradeoff.png")
        print(f"  - {self.figures_dir}/learning_curves_*.png")


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='消融实验结果分析')
    parser.add_argument('--results_dir', type=str, default='./results/ablation',
                        help='结果目录路径')
    parser.add_argument('--experiment', type=str, default=None,
                        choices=['A1', 'A2', 'A3', 'A4'],
                        help='只分析特定实验')

    args = parser.parse_args()

    # 运行分析
    analyzer = AblationAnalyzer(results_dir=args.results_dir)
    analyzer.run_full_analysis()


# 测试代码
if __name__ == "__main__":
    print("=" * 70)
    print("测试 AblationAnalyzer 模块...")
    print("=" * 70)

    # 创建模拟数据测试（不需要真实实验结果）
    test_dir = "./test_ablation"
    os.makedirs(os.path.join(test_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "figures"), exist_ok=True)

    # 创建模拟结果
    mock_results = [
        {
            'run_id': 'A1_steps_20_seed42',
            'experiment': 'A1',
            'variant_desc': 'max_steps=20',
            'seed': 42,
            'best_val_acc': 0.95,
            'train_time_minutes': 15.0,
            'status': 'success',
            'history': {'val_acc': [0.6, 0.8, 0.9, 0.95]}
        },
        {
            'run_id': 'A1_steps_60_seed42',
            'experiment': 'A1',
            'variant_desc': 'max_steps=60',
            'seed': 42,
            'best_val_acc': 0.96,
            'train_time_minutes': 35.0,
            'status': 'success',
            'history': {'val_acc': [0.65, 0.85, 0.92, 0.96]}
        },
        {
            'run_id': 'A2_K_5_seed42',
            'experiment': 'A2',
            'variant_desc': 'K=5',
            'seed': 42,
            'best_val_acc': 0.94,
            'train_time_minutes': 20.0,
            'status': 'success',
            'history': {'val_acc': [0.6, 0.8, 0.9, 0.94]}
        }
    ]

    # 保存模拟数据
    for r in mock_results:
        path = os.path.join(test_dir, "logs", f"{r['run_id']}.json")
        with open(path, 'w') as f:
            json.dump(r, f)

    # 测试分析器
    print(f"\n创建模拟数据: {len(mock_results)} 个实验")

    try:
        analyzer = AblationAnalyzer(results_dir=test_dir)

        # 测试表格生成
        print("\n测试1: 生成汇总表格...")
        table = analyzer.generate_summary_table()
        print(table[:500] + "...")

        # 测试图表（使用模拟数据）
        print("\n测试2: 绘制图表...")
        analyzer.plot_experiment_comparison(save=True)
        analyzer.plot_time_accuracy_tradeoff(save=True)

        # 测试建议生成
        print("\n测试3: 生成配置建议...")
        rec = analyzer.generate_recommendation()
        print(rec[:300] + "...")

        print("\n" + "=" * 70)
        print("✅ 模块3测试通过！可以分析真实实验结果")
        print("=" * 70)
        print("\n使用方式:")
        print("  python ablation_analysis.py")
        print("  python ablation_analysis.py --experiment A1")

        # 清理测试数据
        import shutil

        shutil.rmtree(test_dir)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback

        traceback.print_exc()