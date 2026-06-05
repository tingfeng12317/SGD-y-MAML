#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SGD-Y-MAML: Fixed Steps vs Adaptive Steps Comparison (Academic Style)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ==================== Academic Style Configuration ====================
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
rcParams['axes.unicode_minus'] = False
rcParams['mathtext.fontset'] = 'stix'

# Figure dimensions (Nature/Science standard)
FULL_WIDTH = 17  # cm


def cm2inch(*tupl):
    """Convert cm to inches"""
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


# Colors from original code
COLOR_FIXED = '#0575BF'  # Blue - Fixed 5 steps
COLOR_ADAPTIVE = '#CF221E'  # Red - Adaptive steps
COLOR_YELLOW = '#EDB11F'  # Yellow - Annotation


def load_update_magnitudes(csv_path):
    """Load parameter update magnitude CSV"""
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    summary = df.groupby(['episode', 'depth_type'])['update_magnitude'].agg(['mean', 'std']).reset_index()
    pivot = summary.pivot(index='episode', columns='depth_type', values=['mean', 'std'])
    pivot.columns = [f"{col[1]}_{col[0]}" for col in pivot.columns]
    return pivot.reset_index()


def plot_deep_update_comparison(fixed_dir, adaptive_dir, output_dir='./plots'):
    """
    Figure 1: Deep Layer Parameter Update Magnitude
    """
    os.makedirs(output_dir, exist_ok=True)

    fixed_csv = os.path.join(fixed_dir, "update_magnitudes.csv")
    adaptive_csv = os.path.join(adaptive_dir, "update_magnitudes.csv")

    fixed_data = load_update_magnitudes(fixed_csv)
    adaptive_data = load_update_magnitudes(adaptive_csv)

    if fixed_data is None or adaptive_data is None:
        print("Data loading failed")
        return None

    # Figure size: 17cm width, golden ratio height
    fig, ax = plt.subplots(figsize=cm2inch(FULL_WIDTH, FULL_WIDTH / 1.618))

    # Plot lines (academic style: hollow markers)
    ax.plot(fixed_data['episode'], fixed_data['deep_mean'],
            color=COLOR_FIXED, linewidth=2, linestyle='-',
            marker='o', markersize=5, markevery=50,
            markerfacecolor='white', markeredgewidth=1.5,
            label='Fixed 5 Steps')

    ax.plot(adaptive_data['episode'], adaptive_data['deep_mean'],
            color=COLOR_ADAPTIVE, linewidth=2, linestyle='-',
            marker='s', markersize=5, markevery=50,
            markerfacecolor='white', markeredgewidth=1.5,
            label='Adaptive Steps')

    # Statistics annotation
    fixed_deep_avg = fixed_data['deep_mean'].mean()
    adaptive_deep_avg = adaptive_data['deep_mean'].mean()
    improvement = ((adaptive_deep_avg - fixed_deep_avg) / (fixed_deep_avg + 1e-8)) * 100

    ax.text(0.02, 0.98,
            f'Fixed Mean: {fixed_deep_avg:.4f}\n'
            f'Adaptive Mean: {adaptive_deep_avg:.4f}\n'
            f'Improvement: {improvement:+.1f}%',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8DC',
                      edgecolor=COLOR_YELLOW, alpha=0.9, linewidth=1.5))

    # Labels
    ax.set_xlabel('Episodes', fontsize=11)
    ax.set_ylabel('Update Magnitude (L2 Norm)', fontsize=11)
    ax.set_title('Deep Layer Parameter Update Magnitude',
                 fontsize=12, fontweight='bold', pad=15)

    # Legend (academic style)
    ax.legend(fontsize=10, loc='upper right', frameon=True,
              edgecolor='black', fancybox=False, framealpha=0.9)

    # Academic axis style: remove top/right spines, grid behind data
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Y-axis range (from original code to ensure proper height)
    y_min = min(fixed_data['deep_mean'].min(), adaptive_data['deep_mean'].min())
    y_max = max(fixed_data['deep_mean'].max(), adaptive_data['deep_mean'].max())
    margin = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - margin, y_max + margin)

    plt.tight_layout()
    svg_path = os.path.join(output_dir, 'fig1_deep_update_comparison.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white', dpi=300)
    print(f"Figure 1 saved: {svg_path}")
    plt.close()
    return svg_path


def plot_shallow_update_comparison(fixed_dir, adaptive_dir, output_dir='./plots'):
    """
    Figure 2: Shallow Layer Parameter Update Magnitude
    """
    os.makedirs(output_dir, exist_ok=True)

    fixed_csv = os.path.join(fixed_dir, "update_magnitudes.csv")
    adaptive_csv = os.path.join(adaptive_dir, "update_magnitudes.csv")

    fixed_data = load_update_magnitudes(fixed_csv)
    adaptive_data = load_update_magnitudes(adaptive_csv)

    if fixed_data is None or adaptive_data is None:
        print("Data loading failed")
        return None

    fig, ax = plt.subplots(figsize=cm2inch(FULL_WIDTH, FULL_WIDTH / 1.618))

    ax.plot(fixed_data['episode'], fixed_data['shallow_mean'],
            color=COLOR_FIXED, linewidth=2, linestyle='-',
            marker='o', markersize=5, markevery=50,
            markerfacecolor='white', markeredgewidth=1.5,
            label='Fixed 5 Steps')

    ax.plot(adaptive_data['episode'], adaptive_data['shallow_mean'],
            color=COLOR_ADAPTIVE, linewidth=2, linestyle='-',
            marker='s', markersize=5, markevery=50,
            markerfacecolor='white', markeredgewidth=1.5,
            label='Adaptive Steps')

    fixed_shallow_avg = fixed_data['shallow_mean'].mean()
    adaptive_shallow_avg = adaptive_data['shallow_mean'].mean()

    ax.text(0.02, 0.98,
            f'Fixed Mean: {fixed_shallow_avg:.4f}\n'
            f'Adaptive Mean: {adaptive_shallow_avg:.4f}\n'
            f'Ratio: {adaptive_shallow_avg / (fixed_shallow_avg + 1e-8):.2f}×',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8DC',
                      edgecolor=COLOR_YELLOW, alpha=0.9, linewidth=1.5))

    ax.set_xlabel('Episodes', fontsize=11)
    ax.set_ylabel('Update Magnitude (L2 Norm)', fontsize=11)
    ax.set_title('Shallow Layer Parameter Update Magnitude',
                 fontsize=12, fontweight='bold', pad=15)

    ax.legend(fontsize=10, loc='upper right', frameon=True,
              edgecolor='black', fancybox=False, framealpha=0.9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Y-axis range from original code
    y_min = min(fixed_data['shallow_mean'].min(), adaptive_data['shallow_mean'].min())
    y_max = max(fixed_data['shallow_mean'].max(), adaptive_data['shallow_mean'].max())
    margin = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - margin, y_max + margin)

    plt.tight_layout()
    svg_path = os.path.join(output_dir, 'fig2_shallow_update_comparison.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white', dpi=300)
    print(f"Figure 2 saved: {svg_path}")
    plt.close()
    return svg_path


if __name__ == "__main__":
    fixed_dir = "./update_tracking/fixed5_vs_adaptive/sgd_y_maml_seed42"
    adaptive_dir = "./update_tracking/Smax/sgd_y_maml_seed42"
    output_dir = "./plots"

    print("=" * 70)
    print("SGD-Y-MAML Comparison Plot Generation")
    print("=" * 70)

    plot_deep_update_comparison(fixed_dir, adaptive_dir, output_dir)
    plot_shallow_update_comparison(fixed_dir, adaptive_dir, output_dir)

    print("Completed!")