# algorithms/sgd_y_maml.py
"""
SGD-Y-MAML 最终版（支持梯度统计追踪 - 可选）
"""

import sys

sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import time
from tqdm import tqdm

from config import Config


class SGDYInner:
    """
    SGD-Y内循环优化器（支持可选梯度统计 + 固定步数模式）
    """

    def __init__(self,
                 params: Dict[str, torch.Tensor],
                 lr: float = 0.01,
                 momentum: float = 0.9,
                 eps_min: float = 0.03,
                 decay_rate: float = 0.9,
                 min_lr: float = 1e-4,
                 K: int = 3,
                 noise_floor_threshold: float = 0.6,
                 adjacent_drift_threshold: float = 0.015,
                 use_lr_decay: bool = True,
                 residual_tolerance: float = 1.1,
                 verbose: bool = False,
                 enable_gradient_stats: bool = False,  # 新增：梯度统计开关（默认关闭）
                 force_fixed_steps: bool = False):  # 新增：强制固定步数（禁用早停）

        self.lr = lr
        self.initial_lr = lr
        self.momentum = momentum
        self.eps_min = eps_min
        self.decay_rate = decay_rate
        self.min_lr = min_lr
        self.K = K
        self.noise_floor_threshold = noise_floor_threshold
        self.adjacent_drift_threshold = adjacent_drift_threshold
        self.use_lr_decay = use_lr_decay
        self.residual_tolerance = residual_tolerance
        self.verbose = verbose
        self.enable_gradient_stats = enable_gradient_stats  # 保存配置
        self.force_fixed_steps = force_fixed_steps  # 保存配置

        self.velocities = {name: torch.zeros_like(p) for name, p in params.items() if p.requires_grad}
        self.state = {
            name: {
                'buf': torch.zeros_like(p),
                'counter': 0,
                'lr_counter': 0,
                'current_lr': lr
            }
            for name, p in params.items() if p.requires_grad
        }

        self.step_count = 0
        self.total_layers = len([n for n, p in params.items() if p.requires_grad])

        self.best_residual_mean = float('inf')
        self.optimal_step = -1
        self.optimal_epoch_reached = False
        self.prev_weights = None
        self.early_stop = False

        # 仅在启用时初始化梯度历史
        self.gradient_history = [] if enable_gradient_stats else None

    def _compute_residual_mean(self, grads: Dict[str, torch.Tensor]) -> float:
        residuals = []
        for name, state in self.state.items():
            if name in grads and grads[name] is not None:
                residual = torch.norm(grads[name] - state['buf']).item()
                residuals.append(residual)
        return sum(residuals) / max(len(residuals), 1) if residuals else 0.0

    def _compute_adjacent_drift(self, current_params: Dict[str, torch.Tensor]) -> float:
        if self.prev_weights is None:
            return float('inf')

        total_drift = 0.0
        valid_layers = 0
        for name in current_params:
            if name in self.prev_weights and current_params[name].requires_grad:
                drift = torch.norm(current_params[name] - self.prev_weights[name]).item()
                param_norm = torch.norm(self.prev_weights[name]).item()
                drift /= (param_norm + 1e-8)
                total_drift += drift
                valid_layers += 1

        return total_drift / max(valid_layers, 1)

    def step(self, params: Dict[str, torch.Tensor],
             grads: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], bool]:

        # 如果已触发早停且非固定步数模式，直接返回
        if self.early_stop and not self.force_fixed_steps:
            return params, True

        self.step_count += 1
        new_params = {}

        # 仅在启用梯度统计时收集数据
        current_step_stats = [] if self.enable_gradient_stats else None

        for name, param in params.items():
            if not param.requires_grad:
                new_params[name] = param
                continue

            grad = grads.get(name)
            if grad is not None:
                state = self.state[name]
                state['buf'].mul_(self.momentum).add_(grad, alpha=1 - self.momentum)

                residual = torch.norm(grad - state['buf']).item()
                is_satisfied = residual < self.eps_min

                if is_satisfied:
                    state['counter'] += 1
                else:
                    state['counter'] = 0

                if self.use_lr_decay:
                    if is_satisfied:
                        state['lr_counter'] += 1
                    else:
                        state['lr_counter'] = 0
                else:
                    state['lr_counter'] = 0

                # 仅在启用时记录梯度统计
                if self.enable_gradient_stats:
                    grad_norm = torch.norm(grad).item()
                    current_step_stats.append({
                        'layer_name': name,
                        'grad_norm': grad_norm,  # 原始梯度模长 ||∇L||
                        'residual_norm': residual,  # 残差模长 ||∇L - buf||
                        'residual_ratio': residual / (grad_norm + 1e-8) if grad_norm > 1e-8 else 0.0
                    })

                v = self.velocities[name]
                v.mul_(self.momentum).add_(grad)
                new_param = param - state['current_lr'] * v
                new_params[name] = new_param
            else:
                new_params[name] = param

        # 保存梯度统计历史（仅当启用时）
        if self.enable_gradient_stats and current_step_stats:
            self.gradient_history.append({
                'step': self.step_count,
                'layer_stats': current_step_stats,
                'mean_grad_norm': sum(s['grad_norm'] for s in current_step_stats) / len(current_step_stats),
                'mean_residual_norm': sum(s['residual_norm'] for s in current_step_stats) / len(current_step_stats)
            })

        # 固定步数模式：跳过所有收敛判断，强制返回 converged=False
        if self.force_fixed_steps:
            # 学习率衰减仍然生效（如果启用）
            if self.use_lr_decay:
                for state in self.state.values():
                    if state['lr_counter'] >= self.K:
                        state['current_lr'] = max(state['current_lr'] * self.decay_rate, self.min_lr)
                        state['lr_counter'] = 0

            self.prev_weights = {n: p.clone().detach() for n, p in new_params.items() if p.requires_grad}
            return new_params, False

        # 非固定步数模式：正常收敛判断（原有逻辑）
        layers_with_k_consecutive = sum(
            1 for state in self.state.values()
            if state['counter'] >= self.K
        )
        ratio_satisfied = layers_with_k_consecutive / max(self.total_layers, 1)

        if ratio_satisfied >= self.noise_floor_threshold:
            current_residual_mean = self._compute_residual_mean(grads)
            if not self.optimal_epoch_reached or current_residual_mean < self.best_residual_mean:
                self.best_residual_mean = current_residual_mean
                self.optimal_step = self.step_count
                self.optimal_epoch_reached = True

        if self.use_lr_decay:
            for state in self.state.values():
                if state['lr_counter'] >= self.K:
                    state['current_lr'] = max(state['current_lr'] * self.decay_rate, self.min_lr)
                    state['lr_counter'] = 0

        if self.optimal_epoch_reached:
            adjacent_drift = self._compute_adjacent_drift(new_params)
            if adjacent_drift < self.adjacent_drift_threshold:
                self.early_stop = True
                return new_params, True

        self.prev_weights = {n: p.clone().detach() for n, p in new_params.items() if p.requires_grad}
        return new_params, False

    def get_gradient_history(self):
        """获取梯度历史（仅当启用统计时有效）"""
        return self.gradient_history if self.enable_gradient_stats else []

    def get_lr_stats(self) -> Dict[str, float]:
        lrs = [state['current_lr'] for state in self.state.values()]
        return {
            'min': min(lrs),
            'max': max(lrs),
            'mean': sum(lrs) / len(lrs),
            'initial': self.initial_lr
        }

    def reset_for_new_task(self) -> None:
        self.step_count = 0
        self.best_residual_mean = float('inf')
        self.optimal_step = -1
        self.optimal_epoch_reached = False
        self.prev_weights = None
        self.early_stop = False

        # 重置时清空历史（仅当启用时）
        if self.enable_gradient_stats:
            self.gradient_history = []

        for state in self.state.values():
            state['counter'] = 0
            state['lr_counter'] = 0
            state['current_lr'] = self.initial_lr


class SGD_Y_MAML:
    """SGD-Y-MAML：支持消融实验配置 + 可选梯度统计"""

    def __init__(self, model: nn.Module, config=None, ablation_config=None):
        from config import Config

        self.config = config or Config()
        self.model = model.to(self.config.DEVICE)
        self.ablation_config = ablation_config or {}

        print(f"[SGD-Y-MAML] 初始化 | ID: {id(self)}")

        self.inner_lr = getattr(self.config, 'SGDY_INNER_LR', 0.01)
        self.outer_lr = getattr(self.config, 'META_LR', 0.001)
        self.inner_steps_max = getattr(self.config, 'SGDY_INNER_STEPS_MAX', 60)
        self.momentum = getattr(self.config, 'SGDY_MOMENTUM', 0.9)
        self.eps_min = getattr(self.config, 'SGDY_EPS_MIN', 0.03)
        self.decay_rate = getattr(self.config, 'SGDY_DECAY_RATE', 0.9)
        self.min_lr = getattr(self.config, 'SGDY_MIN_LR', 1e-4)
        self.K = getattr(self.config, 'SGDY_K', 3)
        self.noise_floor_threshold = getattr(self.config, 'SGDY_NOISE_FLOOR_THRESHOLD', 0.6)
        self.adjacent_drift_threshold = getattr(self.config, 'SGDY_ADJACENT_DRIFT_THRESHOLD', 0.015)

        # 消融配置
        self.use_lr_decay = self.ablation_config.get('USE_LR_DECAY', True)
        self.use_regularization = self.ablation_config.get('USE_REGULARIZATION', False)
        self.lambda_stat = self.ablation_config.get('SGDY_STATISTICAL_WEIGHT', 0.0)
        self.lambda_smooth = self.ablation_config.get('SGDY_GRADIENT_SMOOTH_WEIGHT', 0.0)

        # 关键配置：固定步数 + 梯度统计
        self.force_fixed_steps = self.ablation_config.get('FORCE_FIXED_STEPS', False)
        self.fixed_steps = self.ablation_config.get('FIXED_STEPS', self.inner_steps_max)
        self.enable_gradient_stats = self.ablation_config.get('ENABLE_GRADIENT_STATS', False)

        # 如果启用固定步数，强制使用指定步数作为上限
        if self.force_fixed_steps:
            self.inner_steps_max = self.fixed_steps
            print(f"[SGD-Y-MAML] 固定步数模式: {self.fixed_steps}步 (早停已禁用)")

        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.outer_lr)
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'avg_steps': []}
        self.task_stats = []

    def _forward_with_params(self, x, params, training=True):
        """functional前向传播"""
        x = F.conv2d(x, params['layer1.conv.weight'], params['layer1.conv.bias'], padding=1)
        x = F.batch_norm(x, None, None, params['layer1.bn.weight'], params['layer1.bn.bias'], training=training)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = F.conv2d(x, params['layer2.conv.weight'], params['layer2.conv.bias'], padding=1)
        x = F.batch_norm(x, None, None, params['layer2.bn.weight'], params['layer2.bn.bias'], training=training)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = F.conv2d(x, params['layer3.conv.weight'], params['layer3.conv.bias'], padding=1)
        x = F.batch_norm(x, None, None, params['layer3.bn.weight'], params['layer3.bn.bias'], training=training)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = F.conv2d(x, params['layer4.0.weight'], params['layer4.0.bias'], padding=1)
        x = F.batch_norm(x, None, None, params['layer4.1.weight'], params['layer4.1.bias'], training=training)
        x = F.relu(x)

        x = x.view(x.size(0), -1)
        x = F.linear(x, params['fc.weight'], params['fc.bias'])
        return x

    def _compute_regularization(self, grads_history: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        if len(grads_history) < 2:
            return torch.tensor(0.0, device=self.config.DEVICE)

        reg_loss = torch.tensor(0.0, device=self.config.DEVICE)

        if self.lambda_stat > 0:
            for name in grads_history[0].keys():
                grads_list = [g[name] for g in grads_history if name in g and g[name] is not None]
                if len(grads_list) > 1:
                    stacked = torch.stack(grads_list)
                    variance = torch.var(stacked, dim=0).mean()
                    reg_loss += self.lambda_stat * variance

        if self.lambda_smooth > 0:
            for i in range(1, len(grads_history)):
                for name in grads_history[i].keys():
                    if name in grads_history[i - 1] and grads_history[i][name] is not None and grads_history[i - 1][
                        name] is not None:
                        diff = torch.norm(grads_history[i][name] - grads_history[i - 1][name])
                        reg_loss += self.lambda_smooth * diff

        return reg_loss

    def adapt(self, support_x, support_y, task_id=0):
        params = {name: p.clone() for name, p in self.model.named_parameters() if p.requires_grad}

        # 根据配置初始化SGDYInner
        sgdy = SGDYInner(
            params=params,
            lr=self.inner_lr,
            momentum=self.momentum,
            eps_min=self.eps_min,
            decay_rate=self.decay_rate,
            min_lr=self.min_lr,
            K=self.K,
            noise_floor_threshold=self.noise_floor_threshold,
            adjacent_drift_threshold=self.adjacent_drift_threshold,
            use_lr_decay=self.use_lr_decay,
            verbose=False,
            enable_gradient_stats=self.enable_gradient_stats,  # 传递梯度统计开关
            force_fixed_steps=self.force_fixed_steps  # 传递固定步数开关
        )

        grads_history = []
        step = 0
        converged = False

        # 训练循环
        while step < self.inner_steps_max and not converged:
            out = self._forward_with_params(support_x, params, training=True)
            loss = F.cross_entropy(out, support_y)

            grads = torch.autograd.grad(
                loss, params.values(),
                create_graph=True, retain_graph=True, allow_unused=True
            )
            grads_dict = {name: g for name, g in zip(params.keys(), grads) if g is not None}

            for name in grads_dict:
                grads_dict[name] = grads_dict[name].clamp(-10, 10)

            grads_history.append({k: v.clone().detach() for k, v in grads_dict.items()})

            params, converged = sgdy.step(params, grads_dict)
            step += 1

            # 固定步数模式下，强制覆盖converged标志确保跑满指定步数
            if self.force_fixed_steps:
                converged = False  # 强制继续直到达到inner_steps_max

        # 构建返回信息
        info = {
            'steps': step,
            'converged': converged and not self.force_fixed_steps,  # 固定步数时始终为False
            'optimal_reached': sgdy.optimal_epoch_reached if not self.force_fixed_steps else False,
            'optimal_step': sgdy.optimal_step if not self.force_fixed_steps else -1,
            'early_stop': sgdy.early_stop if not self.force_fixed_steps else False,
            'final_lr_stats': sgdy.get_lr_stats(),
            'grads_history': grads_history,
            'force_fixed_steps': self.force_fixed_steps,  # 标记当前模式
            'gradient_history': sgdy.get_gradient_history() if self.enable_gradient_stats else []  # 仅当启用时返回
        }

        return params, info

    def train_step(self, task_batch, first_order=False):
        self.model.train()
        self.meta_optimizer.zero_grad()

        total_loss = 0.0
        total_acc = 0.0
        batch_stats = []

        for task_idx, (support_x, support_y, query_x, query_y) in enumerate(task_batch):
            support_x = support_x.to(self.config.DEVICE)
            support_y = support_y.to(self.config.DEVICE)
            query_x = query_x.to(self.config.DEVICE)
            query_y = query_y.to(self.config.DEVICE)

            adapted_params, adapt_info = self.adapt(support_x, support_y, task_id=task_idx)

            query_out = self._forward_with_params(query_x, adapted_params, training=True)
            query_loss = F.cross_entropy(query_out, query_y)

            if self.use_regularization and len(adapt_info['grads_history']) > 1:
                reg_loss = self._compute_regularization(adapt_info['grads_history'])
                total_task_loss = query_loss + reg_loss
            else:
                total_task_loss = query_loss

            total_task_loss.backward()

            total_loss += total_task_loss.item()
            pred = query_out.argmax(dim=1)
            acc = (pred == query_y).float().mean().item()
            total_acc += acc

            batch_stats.append({
                'task_id': task_idx,
                'steps': adapt_info['steps'],
                'early_stop': adapt_info['early_stop'],
                'query_loss': query_loss.item()
            })

        self.meta_optimizer.step()
        self.task_stats.extend(batch_stats)

        avg_steps = sum(s['steps'] for s in batch_stats) / len(batch_stats)

        return total_loss / len(task_batch), total_acc / len(task_batch), avg_steps

    def evaluate(self, dataset, num_tasks=100):
        self.model.eval()
        total_loss, total_acc = 0.0, 0.0

        from data.omniglot_dataset import TaskSampler

        for _ in range(num_tasks):
            sampler = TaskSampler(dataset, self.config.N_WAY, self.config.K_SHOT,
                                  self.config.K_QUERY, num_tasks=1)
            sx, sy, qx, qy = sampler.sample_task()
            sx, sy = sx.to(self.config.DEVICE), sy.to(self.config.DEVICE)
            qx, qy = qx.to(self.config.DEVICE), qy.to(self.config.DEVICE)

            with torch.enable_grad():
                adapted_params, _ = self.adapt(sx, sy)

            with torch.no_grad():
                out = self._forward_with_params(qx, adapted_params, training=True)
                loss = F.cross_entropy(out, qy).item()
                acc = (out.argmax(dim=1) == qy).float().mean().item()

            total_loss += loss
            total_acc += acc

        return total_loss / num_tasks, total_acc / num_tasks

    def train(self, train_dataset, val_dataset=None, first_order=False,
              step_callback=None):

        mode_str = f"固定{self.fixed_steps}步" if self.force_fixed_steps else f"自适应步数(max={self.inner_steps_max})"
        stats_str = " | 梯度统计:开" if self.enable_gradient_stats else " | 梯度统计:关"

        print(f"\n{'=' * 60}")
        print(f"SGD-Y-MAML 训练 | {mode_str} | LR衰减:{'开' if self.use_lr_decay else '关'}{stats_str}")
        print(f"{'=' * 60}")

        best_val_acc = 0.0
        start_time = time.time()

        from data.omniglot_dataset import TaskSampler

        # 使用tqdm显示进度，无额外打印
        pbar = tqdm(range(1, self.config.MAX_EPISODES + 1),
                    desc="Training", unit="ep", ncols=100)

        for episode in pbar:
            sampler = TaskSampler(train_dataset, self.config.N_WAY, self.config.K_SHOT,
                                  self.config.K_QUERY, num_tasks=self.config.META_BATCH_SIZE)
            task_batch = [sampler.sample_task() for _ in range(self.config.META_BATCH_SIZE)]

            loss, acc, avg_steps = self.train_step(task_batch, first_order)

            if episode % 10 == 0 or episode == 1:
                self.history['train_loss'].append(loss)
                self.history['train_acc'].append(acc)
                self.history['avg_steps'].append(avg_steps)

                val_loss, val_acc = None, None
                if val_dataset:
                    val_loss, val_acc = self.evaluate(val_dataset, num_tasks=100)
                    self.history['val_loss'].append(val_loss)
                    self.history['val_acc'].append(val_acc)

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc

                # 只在进度条显示关键信息，无额外打印
                pbar.set_postfix({
                    'loss': f'{loss:.3f}',
                    'acc': f'{acc:.2%}',
                    'val': f'{val_acc:.2%}' if val_acc else '-',
                    'best': f'{best_val_acc:.2%}',
                    'steps': f'{avg_steps:.1f}'
                })

                if step_callback:
                    step_callback(
                        episode=episode,
                        train_loss=loss,
                        train_acc=acc,
                        val_loss=val_loss,
                        val_acc=val_acc,
                        avg_steps=avg_steps,
                        best_val_acc=best_val_acc,
                        model=self.model
                    )

        total_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"完成！最佳准确率: {best_val_acc:.2%} | 总时间: {total_time / 60:.1f}min")
        print(f"{'=' * 60}")

        return self.history