#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SGD-Y-MAML Core Mechanism Visualization (Streamlined: Fig 1 Layer-wise Convergence + Fig 3 Optimization Trajectory)
Purpose: Demonstrate internal algorithm mechanisms, no control group needed
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# ==================== Academic Style Configuration ====================
# English fonts + LaTeX math symbols (solves ∇ display issues)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
rcParams['axes.unicode_minus'] = False
rcParams['mathtext.fontset'] = 'stix'  # Use STIX for math symbols (∇, etc.)

# Figure dimensions (double-column 17cm, single-column 8cm, Nature/Science standard)
FULL_WIDTH = 17  # cm
COLUMN_WIDTH = 8  # cm


def cm2inch(*tupl):
    """Convert cm to inches"""
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


# Colorblind-friendly palette
COLOR_SHALLOW = '#0173B2'  # Blue - shallow layers
COLOR_DEEP = '#DE8F05'  # Orange - deep layers
COLOR_THRESHOLD = '#029E73'  # Green - threshold
COLOR_START = '#029E73'  # Green - start point
COLOR_END = '#DE8F05'  # Orange - end point


# ==================== Figure 1: Layer-wise Convergence ====================
def plot_layer_wise_convergence(df, output_dir='./plots_e'):
    """
    Academic significance: Reveals convergence heterogeneity across network depths
    - Shallow (general features): Fast convergence (within 3 steps)
    - Deep (task-specific): Gradual convergence (8-10 steps)
    Proves theoretical necessity of adaptive step sizes
    """
    os.makedirs(output_dir, exist_ok=True)

    # Aggregate by depth type and inner step
    grouped = df.groupby(['depth_type', 'inner_step'])['residual_norm'].agg(['mean', 'std']).reset_index()

    # Double-column width, golden ratio height
    fig, ax = plt.subplots(figsize=cm2inch(FULL_WIDTH, FULL_WIDTH / 1.618))

    # Define line styles: shallow solid+circle, deep dashed+square
    styles = {
        'shallow': {
            'color': COLOR_SHALLOW,
            'linestyle': '-',
            'marker': 'o',
            'markersize': 5,
            'label': 'Shallow Layers (layers 1-2)'
        },
        'deep': {
            'color': COLOR_DEEP,
            'linestyle': '--',
            'marker': 's',
            'markersize': 5,
            'label': 'Deep Layers (layers 3-4+fc)'
        }
    }

    # Plot curves
    for depth in ['shallow', 'deep']:
        data = grouped[grouped['depth_type'] == depth]
        style = styles[depth]

        ax.plot(data['inner_step'], data['mean'],
                color=style['color'],
                linewidth=2,
                linestyle=style['linestyle'],
                marker=style['marker'],
                markersize=style['markersize'],
                markerfacecolor='white',  # Hollow markers, black-and-white print friendly
                markeredgewidth=1.5,
                markeredgecolor=style['color'],
                label=style['label'])

    # Convergence threshold line (green dotted)
    ax.axhline(y=0.03, color=COLOR_THRESHOLD, linestyle=':', linewidth=1.5,
               label=r'Convergence Threshold $\epsilon_{\mathrm{min}}=0.03$')

    # Axis labels (LaTeX format for ∇)
    ax.set_xlabel('Inner-loop Steps', fontsize=11)
    ax.set_ylabel(r'Residual Norm $\|\nabla\mathcal{L} - \mathrm{buf}\|$', fontsize=11)

    # 简化后的标题
    ax.set_title('Layer-wise Convergence: Shallow vs Deep Layers',
                 fontsize=12, fontweight='bold', pad=15)

    # Legend
    ax.legend(fontsize=10, loc='upper right', frameon=True,
              edgecolor='black', fancybox=False, framealpha=0.9)

    # Grid and borders
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)  # Grid lines behind data
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_xlim(0, 10.5)

    # Remove top and right borders (academic style)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Key annotations: Explain academic significance of layer differences
    ax.annotate('Early Convergence\n(within 3 steps)',
                xy=(2.5, 0.025),
                fontsize=9, ha='center', va='center',
                color=COLOR_SHALLOW,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor=COLOR_SHALLOW, linewidth=1.5, alpha=0.9))

    ax.annotate('Gradual Optimization\n(8-10 steps)',
                xy=(8, 0.08),
                fontsize=9, ha='center', va='bottom',
                color=COLOR_DEEP,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor=COLOR_DEEP, linewidth=1.5, alpha=0.9),
                arrowprops=dict(arrowstyle='->', color=COLOR_DEEP, lw=1.5,
                                connectionstyle='arc3,rad=0.2'))

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'fig1_layer_wise_convergence.svg')
    plt.savefig(save_path, format='svg', bbox_inches='tight', facecolor='white')
    print(f"✅ Figure 1 saved: {save_path}")
    print(
        "   Content: Convergence speed difference between shallow vs deep layers, proving necessity of adaptive step sizes")
    plt.close()


# ==================== Figure 3: Gradient Compression Trajectory ====================
def plot_trajectory(df, output_dir='./plots_e', episode_id=1, task_id=0):
    """
    Academic significance: Visualize SGD-Y gradient compression process
    - X-axis: Original gradient norm (||∇L||, representing task-specific noise intensity)
    - Y-axis: Residual norm (||∇L - buf||, representing signal after noise filtering)
    - Trajectory from upper-left to lower-right: Denoising process from high noise to low noise
    Proves effectiveness of buffer mechanism in extracting meta-knowledge
    """
    os.makedirs(output_dir, exist_ok=True)

    # Select data for specific episode and task (default: first episode, first task)
    data = df[(df['episode'] == episode_id) &
              (df['task_id'] == task_id) &
              (df['layer_name'] == 'layer1.conv.weight')].copy()

    if data.empty:
        print(f"⚠️ Data not found: episode={episode_id}, task={task_id}, trying other combinations...")
        # If specified combination doesn't exist, use first available combination
        available = df[df['layer_name'] == 'layer1.conv.weight'][['episode', 'task_id']].drop_duplicates()
        if not available.empty:
            episode_id = available.iloc[0]['episode']
            task_id = available.iloc[0]['task_id']
            data = df[(df['episode'] == episode_id) &
                      (df['task_id'] == task_id) &
                      (df['layer_name'] == 'layer1.conv.weight')].copy()
            print(f"   Automatically switched to episode={episode_id}, task={task_id}")
        else:
            print("   Error: No valid data found")
            return

    data = data.sort_values('inner_step')

    # Double-column width
    fig, ax = plt.subplots(figsize=cm2inch(FULL_WIDTH, FULL_WIDTH / 1.618))

    # Plot trajectory points (color mapped to steps)
    scatter = ax.scatter(data['grad_norm'], data['residual_norm'],
                         c=data['inner_step'],
                         cmap='viridis',
                         s=50,
                         edgecolors='black',
                         linewidth=0.5,
                         alpha=0.8,
                         zorder=3)

    # Add arrows indicating optimization direction (every 2 steps to avoid overcrowding)
    step_interval = max(1, len(data) // 5)
    for i in range(0, len(data) - step_interval, step_interval):
        ax.annotate('',
                    xy=(data['grad_norm'].iloc[i + step_interval],
                        data['residual_norm'].iloc[i + step_interval]),
                    xytext=(data['grad_norm'].iloc[i],
                            data['residual_norm'].iloc[i]),
                    arrowprops=dict(arrowstyle='->', color='gray',
                                    alpha=0.6, lw=1.5),
                    zorder=2)

    # Mark start point (green circle)
    start = data.iloc[0]
    ax.scatter(start['grad_norm'], start['residual_norm'],
               s=200, marker='o', facecolors=COLOR_START,
               edgecolors='black', linewidth=2,
               label='Start (High Noise)', zorder=5)

    # Mark end point (orange square)
    end = data.iloc[-1]
    ax.scatter(end['grad_norm'], end['residual_norm'],
               s=200, marker='s', facecolors=COLOR_END,
               edgecolors='black', linewidth=2,
               label='End (Low Noise)', zorder=5)

    # Colorbar (steps)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Inner-loop Steps', fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Axis labels (LaTeX format)
    ax.set_xlabel(r'Gradient Norm $\|\nabla\mathcal{L}\|$ (Noise Intensity)', fontsize=11)
    ax.set_ylabel(r'Residual Norm $\|\nabla\mathcal{L} - \mathrm{buf}\|$ (Filtered Signal)', fontsize=11)

    # 简化后的标题
    ax.set_title('Gradient Denoising Dynamics',
                 fontsize=12, fontweight='bold', pad=15)

    # Legend
    ax.legend(fontsize=10, loc='upper left', frameon=True,
              edgecolor='black', fancybox=False, labelspacing=1)

    # Grid and borders
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Add diagonal reference line (y=x, baseline for no compression)
    max_val = max(data['grad_norm'].max(), data['residual_norm'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1, label='No-Compression Baseline')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'fig3_trajectory.svg')
    plt.savefig(save_path, format='svg', bbox_inches='tight', facecolor='white')
    print(f"✅ Figure 3 saved: {save_path}")
    print("   Content: Gradient-residual phase space trajectory, proving effectiveness of buffer denoising mechanism")
    plt.close()


# ==================== Main Function ====================
def main(csv_path, output_dir='./plots_e'):
    print("=" * 70)
    print("SGD-Y-MAML Core Mechanism Visualization")
    print("Fig 1: Layer-wise Convergence Heterogeneity (proves necessity of adaptive step sizes)")
    print("Fig 3: Gradient Compression Dynamics (proves buffer denoising mechanism)")
    print("=" * 70)

    if not os.path.exists(csv_path):
        print(f"❌ Error: Data file not found {csv_path}")
        print("Please modify csv_path to point to the correct gradient_stats.csv path")
        return

    df = pd.read_csv(csv_path)
    print(f"\nData loading completed: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Episodes: {df['episode'].nunique()}, Tasks: {df['task_id'].nunique()}")

    # Generate Fig 1: Layer-wise convergence
    print("\nGenerating Fig 1: Layer-wise Residual Convergence Curves...")
    plot_layer_wise_convergence(df, output_dir)

    # Generate Fig 3: Single task trajectory (using first available episode/task combination)
    print("\nGenerating Fig 3: Single Task Phase Space Trajectory...")
    plot_trajectory(df, output_dir, episode_id=1, task_id=0)

    print("\n" + "=" * 70)
    print("Visualization completed! Output files:")
    print(f"  1. {output_dir}/fig1_layer_wise_convergence.svg")
    print(f"  2. {output_dir}/fig3_trajectory.svg")
    print("=" * 70)


if __name__ == "__main__":
    # Please modify path according to your actual situation
    csv_path = './update_tracking/Smax/sgd_y_maml_seed42/gradient_stats.csv'
    main(csv_path)