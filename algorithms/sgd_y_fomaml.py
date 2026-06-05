# algorithms/sgd_y_fomaml.py
"""
SGD-Y-FOMAML (First-Order MAML with SGD-Y Inner Loop)
完全独立实现，匹配项目中的 OmniglotNet 结构
- 内循环：SGD-Y 双计数器收敛
- 外循环：First-Order MAML（一阶梯度）
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Callable, Optional
import time
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from models.omniglot_net import OmniglotNet
from data.omniglot_dataset import TaskSampler


class SGDYInnerOptimizer:
    """SGD-Y 内循环优化器（双计数器版）"""

    def __init__(self,
                 params: Dict[str, torch.Tensor],
                 lr: float = 0.01,
                 momentum: float = 0.9,
                 eps_min: float = 0.03,
                 decay_rate: float = 0.9,
                 min_lr: float = 1e-6,
                 K: int = 3,
                 noise_floor_threshold: float = 0.6,
                 adjacent_drift_threshold: float = 0.03,
                 use_lr_decay: bool = True,
                 residual_tolerance: float = 1.1,
                 verbose: bool = False):

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

        # 初始化速度缓冲区和状态
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

        # 收敛状态跟踪
        self.best_weights = None
        self.best_residual_mean = float('inf')
        self.optimal_step = -1
        self.optimal_epoch_reached = False

        self.last_weights = None
        self.last_residual_mean = None
        self.prev_weights = None
        self.early_stop = False

    def _compute_residual_mean(self, grads: Dict[str, torch.Tensor]) -> float:
        """计算平均残差"""
        residuals = []
        for name, state in self.state.items():
            if name in grads and grads[name] is not None:
                residual = torch.norm(grads[name] - state['buf']).item()
                residuals.append(residual)
        return sum(residuals) / max(len(residuals), 1) if residuals else 0.0

    def _compute_adjacent_drift(self, current_params: Dict[str, torch.Tensor]) -> float:
        """计算相邻步骤间的参数漂移（归一化）"""
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

    def _select_best_weight(self) -> Dict[str, torch.Tensor]:
        """权重选择逻辑"""
        if self.best_weights is None or self.last_weights is None:
            return self.last_weights if self.last_weights else self.best_weights

        if self.last_residual_mean > self.best_residual_mean * self.residual_tolerance:
            return self.best_weights
        else:
            return self.last_weights

    def step(self, params: Dict[str, torch.Tensor],
             grads: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], bool]:
        """单步更新，执行双收敛检测"""
        if self.early_stop:
            return self._select_best_weight(), True

        self.step_count += 1
        new_params = {}

        # 逐层更新参数
        for name, param in params.items():
            if not param.requires_grad:
                new_params[name] = param
                continue

            grad = grads.get(name)
            if grad is not None:
                state = self.state[name]

                # 更新动量缓冲区
                state['buf'].mul_(self.momentum).add_(grad, alpha=1 - self.momentum)

                # 计算残差
                residual = torch.norm(grad - state['buf']).item()
                is_satisfied = residual < self.eps_min

                # 第一重收敛计数器
                if is_satisfied:
                    state['counter'] += 1
                else:
                    state['counter'] = 0

                # 学习率衰减计数器
                if self.use_lr_decay:
                    if is_satisfied:
                        state['lr_counter'] += 1
                    else:
                        state['lr_counter'] = 0

                # SGD更新
                v = self.velocities[name]
                v.mul_(self.momentum).add_(grad)
                new_param = param - state['current_lr'] * v
                new_params[name] = new_param
            else:
                new_params[name] = param

        # ========== 第一重收敛检测（噪声地板） ==========
        layers_with_k_consecutive = sum(
            1 for state in self.state.values()
            if state['counter'] >= self.K
        )
        ratio_satisfied = layers_with_k_consecutive / max(self.total_layers, 1)

        if ratio_satisfied >= self.noise_floor_threshold:
            current_residual_mean = self._compute_residual_mean(grads)

            if not self.optimal_epoch_reached or current_residual_mean < self.best_residual_mean:
                self.best_weights = {n: p.clone().detach() for n, p in new_params.items() if p.requires_grad}
                self.best_residual_mean = current_residual_mean
                self.optimal_step = self.step_count
                self.optimal_epoch_reached = True

                # 【已删除】第一重收敛打印
                # if self.verbose:
                #     print(f"🎯 第一重收敛 @ Step {self.step_count}: "
                #           f"{ratio_satisfied:.0%}层达标, residual={current_residual_mean:.4f}")

        # ========== 学习率衰减 ==========
        if self.use_lr_decay:
            for state in self.state.values():
                if state['lr_counter'] >= self.K:
                    state['current_lr'] = max(state['current_lr'] * self.decay_rate, self.min_lr)
                    state['lr_counter'] = 0

        # ========== 第二重收敛检测（漂移早停） ==========
        if self.optimal_epoch_reached:
            adjacent_drift = self._compute_adjacent_drift(new_params)

            if adjacent_drift < self.adjacent_drift_threshold:
                self.early_stop = True
                self.last_weights = {n: p.clone().detach() for n, p in new_params.items() if p.requires_grad}
                self.last_residual_mean = self._compute_residual_mean(grads)

                # 【已删除】第二重收敛打印
                # if self.verbose:
                #     chosen = "BEST" if (
                #                 self.last_residual_mean > self.best_residual_mean * self.residual_tolerance) else "LAST"
                #     print(f"⏹️  第二重收敛 @ Step {self.step_count}: "
                #           f"drift={adjacent_drift:.4f}, 选择{chosen}权重")

                return self._select_best_weight(), True

        # 保存当前权重用于下次漂移计算
        self.prev_weights = {n: p.clone().detach() for n, p in new_params.items() if p.requires_grad}
        return new_params, False

    def get_lr_stats(self) -> Dict[str, float]:
        """获取当前学习率统计"""
        lrs = [state['current_lr'] for state in self.state.values()]
        return {
            'min': min(lrs),
            'max': max(lrs),
            'mean': sum(lrs) / len(lrs),
            'initial': self.initial_lr
        }

    def get_convergence_info(self) -> Dict:
        """获取收敛状态信息"""
        return {
            'step_count': self.step_count,
            'optimal_reached': self.optimal_epoch_reached,
            'optimal_step': self.optimal_step,
            'early_stop': self.early_stop,
            'best_residual': self.best_residual_mean,
            'last_residual': self.last_residual_mean,
            'has_best_weights': self.best_weights is not None,
            'has_last_weights': self.last_weights is not None
        }


class SGD_Y_FOMAML:
    """
    SGD-Y-FOMAML: 与MAML相同接口的元学习器
    """

    def __init__(self, model: nn.Module, config=None, ablation_config=None):
        self.config = config or Config
        self.model = model.to(self.config.DEVICE)
        self.ablation_config = ablation_config or {}

        print(f"[SGD-Y-FOMAML] 一阶模式初始化 | ID: {id(self)}")

        # SGD-Y参数
        self.inner_lr = getattr(self.config, 'SGDY_INNER_LR', 0.01)
        self.inner_steps_max = getattr(self.config, 'SGDY_INNER_STEPS_MAX', 20)
        self.momentum = getattr(self.config, 'SGDY_MOMENTUM', 0.9)
        self.eps_min = getattr(self.config, 'SGDY_EPS_MIN', 0.03)
        self.decay_rate = getattr(self.config, 'SGDY_DECAY_RATE', 0.9)
        self.min_lr = getattr(self.config, 'SGDY_MIN_LR', 1e-4)
        self.K = getattr(self.config, 'SGDY_K', 3)
        self.noise_floor_threshold = getattr(self.config, 'SGDY_NOISE_FLOOR_THRESHOLD', 0.6)
        self.adjacent_drift_threshold = getattr(self.config, 'SGDY_ADJACENT_DRIFT_THRESHOLD', 0.03)
        print(f"[DEBUG] drift_threshold = {self.adjacent_drift_threshold}")

        # 消融配置
        self.use_lr_decay = self.ablation_config.get('USE_LR_DECAY', True)
        self.use_regularization = self.ablation_config.get('USE_REGULARIZATION', False)
        self.lambda_stat = self.ablation_config.get('SGDY_STATISTICAL_WEIGHT', 0.0)
        self.lambda_smooth = self.ablation_config.get('SGDY_GRADIENT_SMOOTH_WEIGHT', 0.0)

        # 外循环优化器
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.META_LR)

        # 历史记录
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'avg_steps': []
        }
        self.task_stats = []

        # 收敛监控统计
        self.convergence_stats = {
            'first_convergence_count': 0,
            'second_convergence_count': 0,
            'weight_choice_best': 0,
            'weight_choice_last': 0,
            'total_tasks': 0
        }

    def _forward_with_params(self, x: torch.Tensor, params: Dict[str, torch.Tensor],
                             training: bool = True) -> torch.Tensor:
        """
        使用给定参数进行前向传播（匹配OmniglotNet结构）
        参数命名：layer1.conv.weight, layer1.bn.running_mean等
        """
        # 使用模型本身的batch norm的running stats（与MAML一致）
        # Layer 1
        x = F.conv2d(x, params['layer1.conv.weight'], params['layer1.conv.bias'], padding=1)
        x = F.batch_norm(x,
                         running_mean=self.model.layer1.bn.running_mean,
                         running_var=self.model.layer1.bn.running_var,
                         weight=params['layer1.bn.weight'],
                         bias=params['layer1.bn.bias'],
                         training=training)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Layer 2
        x = F.conv2d(x, params['layer2.conv.weight'], params['layer2.conv.bias'], padding=1)
        x = F.batch_norm(x,
                         running_mean=self.model.layer2.bn.running_mean,
                         running_var=self.model.layer2.bn.running_var,
                         weight=params['layer2.bn.weight'],
                         bias=params['layer2.bn.bias'],
                         training=training)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Layer 3
        x = F.conv2d(x, params['layer3.conv.weight'], params['layer3.conv.bias'], padding=1)
        x = F.batch_norm(x,
                         running_mean=self.model.layer3.bn.running_mean,
                         running_var=self.model.layer3.bn.running_var,
                         weight=params['layer3.bn.weight'],
                         bias=params['layer3.bn.bias'],
                         training=training)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Layer 4（Sequential结构：0=conv, 1=bn）
        x = F.conv2d(x, params['layer4.0.weight'], params['layer4.0.bias'], padding=1)
        x = F.batch_norm(x,
                         running_mean=self.model.layer4[1].running_mean,
                         running_var=self.model.layer4[1].running_var,
                         weight=params['layer4.1.weight'],
                         bias=params['layer4.1.bias'],
                         training=training)
        x = F.relu(x)

        # Flatten and FC
        x = x.view(x.size(0), -1)
        x = F.linear(x, params['fc.weight'], params['fc.bias'])
        return x

    def _compute_regularization(self, grads_history: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """计算统计一致性和梯度平滑正则化"""
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
                    if name in grads_history[i - 1] and grads_history[i][name] is not None and \
                            grads_history[i - 1][name] is not None:
                        diff = torch.norm(grads_history[i][name] - grads_history[i - 1][name])
                        reg_loss += self.lambda_smooth * diff

        return reg_loss

    def adapt(self, support_x: torch.Tensor, support_y: torch.Tensor,
              task_id: int = 0, verbose: bool = False) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """SGD-Y内循环自适应"""
        # 克隆当前模型参数
        params = {name: p.clone() for name, p in self.model.named_parameters() if p.requires_grad}

        # 初始化SGD-Y优化器
        sgdy = SGDYInnerOptimizer(
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
            verbose=verbose
        )

        grads_history = []
        step = 0
        converged = False

        while step < self.inner_steps_max and not converged:
            out = self._forward_with_params(support_x, params, training=True)
            loss = F.cross_entropy(out, support_y)

            grads = torch.autograd.grad(
                loss, params.values(),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )
            grads_dict = {name: g for name, g in zip(params.keys(), grads) if g is not None}

            # 梯度裁剪
            for name in grads_dict:
                grads_dict[name] = grads_dict[name].clamp(-10, 10)

            grads_history.append({k: v.clone().detach() for k, v in grads_dict.items()})

            params, converged = sgdy.step(params, grads_dict)
            step += 1

        # 获取收敛信息并更新统计
        conv_info = sgdy.get_convergence_info()
        final_params = sgdy._select_best_weight()
        if final_params is None:
            final_params = params

        self.convergence_stats['total_tasks'] += 1
        if conv_info['optimal_reached']:
            self.convergence_stats['first_convergence_count'] += 1
        if conv_info['early_stop']:
            self.convergence_stats['second_convergence_count'] += 1

        if conv_info['has_best_weights'] and conv_info['has_last_weights']:
            if conv_info['last_residual'] > conv_info['best_residual'] * sgdy.residual_tolerance:
                self.convergence_stats['weight_choice_best'] += 1
            else:
                self.convergence_stats['weight_choice_last'] += 1
        elif conv_info['has_best_weights']:
            self.convergence_stats['weight_choice_best'] += 1
        elif conv_info['has_last_weights']:
            self.convergence_stats['weight_choice_last'] += 1

        info = {
            'steps': step,
            'converged': converged,
            'optimal_reached': conv_info['optimal_reached'],
            'optimal_step': conv_info['optimal_step'],
            'early_stop': conv_info['early_stop'],
            'final_lr_stats': sgdy.get_lr_stats(),
            'grads_history': grads_history,
            'convergence_info': conv_info,
            'task_id': task_id
        }

        return final_params, info

    def train_step(self, task_batch: List[Tuple], first_order: bool = True,
                   verbose: bool = False) -> Tuple[float, float, float]:
        """外循环训练步骤（与MAML接口一致）"""
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

            # 内循环适应
            adapted_params, adapt_info = self.adapt(support_x, support_y, task_id=task_idx, verbose=verbose)

            # FOMAML一阶近似
            if first_order:
                adapted_params = {
                    name: param.detach().requires_grad_(True)
                    for name, param in adapted_params.items()
                }

            # 查询集评估
            query_out = self._forward_with_params(query_x, adapted_params, training=True)
            query_loss = F.cross_entropy(query_out, query_y)

            # 正则化
            if self.use_regularization and len(adapt_info['grads_history']) > 1:
                reg_loss = self._compute_regularization(adapt_info['grads_history'])
                total_task_loss = query_loss + reg_loss
            else:
                total_task_loss = query_loss

            # 梯度计算
            if first_order:
                grads = torch.autograd.grad(
                    total_task_loss,
                    adapted_params.values(),
                    create_graph=False,
                    allow_unused=True
                )

                param_dict = dict(self.model.named_parameters())
                for (name, _), grad in zip(adapted_params.items(), grads):
                    if grad is not None and name in param_dict:
                        param = param_dict[name]
                        if param.grad is None:
                            param.grad = grad.clone()
                        else:
                            param.grad.add_(grad)
            else:
                total_task_loss.backward()

            total_loss += total_task_loss.item()
            pred = query_out.argmax(dim=1)
            acc = (pred == query_y).float().mean().item()
            total_acc += acc

            batch_stats.append({
                'task_id': task_idx,
                'steps': adapt_info['steps'],
                'early_stop': adapt_info['early_stop']
            })

        self.meta_optimizer.step()
        self.task_stats.extend(batch_stats)

        avg_steps = sum(s['steps'] for s in batch_stats) / len(batch_stats) if batch_stats else 0

        return total_loss / len(task_batch), total_acc / len(task_batch), avg_steps

    def evaluate(self, dataset, num_tasks: int = 100, first_order: bool = True):
        """在验证集上评估（与MAML接口一致）"""
        self.model.eval()
        total_loss, total_acc = 0.0, 0.0

        for _ in range(num_tasks):
            sampler = TaskSampler(
                dataset,
                self.config.N_WAY,
                self.config.K_SHOT,
                self.config.K_QUERY,
                num_tasks=1
            )
            sx, sy, qx, qy = sampler.sample_task()
            sx, sy = sx.to(self.config.DEVICE), sy.to(self.config.DEVICE)
            qx, qy = qx.to(self.config.DEVICE), qy.to(self.config.DEVICE)

            with torch.enable_grad():
                adapted_params, _ = self.adapt(sx, sy, first_order)

            with torch.no_grad():
                out = self._forward_with_params(qx, adapted_params, training=True)
                loss = F.cross_entropy(out, qy).item()
                acc = (out.argmax(dim=1) == qy).float().mean().item()

            total_loss += loss
            total_acc += acc

        return total_loss / num_tasks, total_acc / num_tasks

    def train(self, train_dataset, val_dataset=None, first_order: bool = True,
              step_callback: Optional[Callable] = None):
        """
        完整训练流程 - 与MAML完全一致的接口
        """
        print(f"\n{'=' * 70}")
        print(f"开始训练 SGD-Y-FOMAML")
        print(f"  模式: {'一阶近似 (FOMAML)' if first_order else '二阶导数 (MAML)'}")
        print(f"  内循环: 最大{self.inner_steps_max}步, 噪声地板{self.noise_floor_threshold}")
        print(f"  双重收敛: drift阈值{self.adjacent_drift_threshold}")
        print(f"{'=' * 70}")

        best_val_acc = 0.0
        start_time = time.time()

        pbar = tqdm(range(1, self.config.MAX_EPISODES + 1),
                    desc="SGD-Y-FOMAML",
                    unit="ep",
                    ncols=100)

        for episode in pbar:
            # 采样meta-batch（与MAML相同）
            sampler = TaskSampler(
                train_dataset,
                self.config.N_WAY,
                self.config.K_SHOT,
                self.config.K_QUERY,
                num_tasks=self.config.META_BATCH_SIZE
            )
            task_batch = [sampler.sample_task() for _ in range(self.config.META_BATCH_SIZE)]

            # 训练步骤
            # 【修改】始终使用 verbose=False，不再在前3个epoch显示详细收敛信息
            verbose = False

            loss, acc, avg_steps = self.train_step(task_batch, first_order=first_order, verbose=verbose)

            # 每10步评估
            if episode % 10 == 0 or episode == 1:
                self.history['train_loss'].append(loss)
                self.history['train_acc'].append(acc)
                self.history['avg_steps'].append(avg_steps)

                val_loss, val_acc = None, None
                if val_dataset:
                    val_loss, val_acc = self.evaluate(val_dataset, num_tasks=100, first_order=first_order)
                    self.history['val_loss'].append(val_loss)
                    self.history['val_acc'].append(val_acc)

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc

                pbar.set_postfix({
                    'loss': f'{loss:.3f}',
                    'acc': f'{acc:.2%}',
                    'val': f'{val_acc:.2%}' if val_acc else '-',
                    'best': f'{best_val_acc:.2%}',
                    'steps': f'{avg_steps:.1f}'
                })

                # 回调（与MAML完全一致）
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

        # 打印收敛统计（保留训练结束后的摘要统计）
        print(f"\n{'=' * 70}")
        print("📊 收敛机制统计报告")
        print(f"{'=' * 70}")
        total = self.convergence_stats['total_tasks']
        if total > 0:
            print(f"  总任务数: {total}")
            print(
                f"  第一重收敛(噪声地板): {self.convergence_stats['first_convergence_count']} ({self.convergence_stats['first_convergence_count'] / total:.1%})")
            print(
                f"  第二重收敛(早停): {self.convergence_stats['second_convergence_count']} ({self.convergence_stats['second_convergence_count'] / total:.1%})")
            print(
                f"  权重选择: BEST={self.convergence_stats['weight_choice_best']}, LAST={self.convergence_stats['weight_choice_last']}")
        print(f"{'=' * 70}")
        print(f"✅ 训练完成！最佳验证准确率: {best_val_acc:.2%}")
        print(f"⏱️  总训练时间: {total_time / 60:.1f} 分钟")
        print(f"{'=' * 70}")

        return self.history