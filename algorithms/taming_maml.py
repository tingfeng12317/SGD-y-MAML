# algorithms/taming_maml.py
import sys

sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
from typing import Callable, Optional

from config import Config
from data.omniglot_dataset import TaskSampler


def inner_loop_taming(model, support_x, support_y, inner_steps, inner_lr,
                      alpha_reg=1e-4, first_order=False):
    """
    TamingMAML内循环：带正则化的自适应梯度下降
    alpha_reg: L2正则化系数，防止参数偏离初始值太远
    """
    # 克隆当前参数作为起点
    adapted_params = {name: param.clone() for name, param in model.named_parameters()}

    for step in range(inner_steps):
        # 前向传播
        from algorithms.maml import model_forward_with_params
        out = model_forward_with_params(model, support_x, adapted_params)

        # 任务损失 + L2正则化（约束参数不要偏离太远）
        task_loss = F.cross_entropy(out, support_y)
        reg_loss = 0.0
        for name, param in adapted_params.items():
            if 'weight' in name:  # 只对权重正则化，不正则化bias
                reg_loss = reg_loss + torch.sum(param ** 2)

        loss = task_loss + alpha_reg * reg_loss

        # 计算梯度
        grads = torch.autograd.grad(
            loss,
            adapted_params.values(),
            create_graph=not first_order,
            retain_graph=True,
            allow_unused=True
        )

        # 自适应学习率（随step衰减）+ 梯度裁剪
        current_lr = inner_lr * (0.9 ** step)
        adapted_params = {
            name: param - current_lr * torch.clamp(grad, -1.0, 1.0)
            if grad is not None else param
            for (name, param), grad in zip(adapted_params.items(), grads)
        }

    return adapted_params, loss.item()


def outer_step_taming(model, meta_optimizer, task_batch, inner_steps,
                      inner_lr, alpha_reg=1e-4, first_order=False):
    """
    TamingMAML外循环：带正则化的元更新
    """
    meta_optimizer.zero_grad()

    total_loss = 0.0
    total_acc = 0.0
    num_tasks = len(task_batch)

    for support_x, support_y, query_x, query_y in task_batch:
        # 数据迁移
        support_x = support_x.to(Config.DEVICE)
        support_y = support_y.to(Config.DEVICE)
        query_x = query_x.to(Config.DEVICE)
        query_y = query_y.to(Config.DEVICE)

        # 保存初始参数用于元正则化
        initial_params = {name: param.clone().detach()
                          for name, param in model.named_parameters()}

        # 内循环适应（带正则化）
        adapted_params, _ = inner_loop_taming(
            model, support_x, support_y,
            inner_steps, inner_lr, alpha_reg, first_order
        )

        # 查询集评估
        from algorithms.maml import model_forward_with_params
        query_out = model_forward_with_params(model, query_x, adapted_params)
        task_loss = F.cross_entropy(query_out, query_y)

        # 元正则化：防止元参数过拟合特定任务分布
        meta_reg = 0.0
        for name in adapted_params:
            meta_reg = meta_reg + torch.sum((adapted_params[name] - initial_params[name]) ** 2)

        combined_loss = task_loss + 0.01 * meta_reg / num_tasks

        # 反向传播
        combined_loss.backward()

        # 统计
        total_loss += task_loss.item()
        pred = query_out.argmax(dim=1)
        total_acc += (pred == query_y).float().mean().item()

    # 梯度裁剪后元更新
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    meta_optimizer.step()

    return total_loss / num_tasks, total_acc / num_tasks


class TamingMAML:
    """
    TamingMAML封装：带自适应正则化的稳定MAML
    """

    def __init__(self, model, config=None):
        self.config = config or Config
        self.model = model.to(self.config.DEVICE)

        # 使用AdamW增强正则化效果
        self.meta_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.META_LR,
            weight_decay=1e-4
        )

        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

    def train_step(self, task_batch, first_order=False):
        """单次训练步骤"""
        self.model.train()
        loss, acc = outer_step_taming(
            self.model, self.meta_optimizer, task_batch,
            inner_steps=self.config.INNER_STEPS,
            inner_lr=self.config.INNER_LR,
            alpha_reg=1e-3,
            first_order=first_order
        )
        return loss, acc

    def evaluate(self, dataset, num_tasks=100, first_order=True):
        """
        验证集评估（关键修复：内循环需要梯度，查询评估不需要）
        """
        self.model.eval()
        total_loss, total_acc = 0.0, 0.0

        for _ in range(num_tasks):  # 移除了 torch.no_grad() 包裹
            # 采样任务
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

            # 内循环适应（需要梯度！）
            with torch.enable_grad():
                adapted_params, _ = inner_loop_taming(
                    self.model, sx, sy,
                    self.config.INNER_STEPS,
                    self.config.INNER_LR,
                    alpha_reg=0.0,  # 测试时无正则化
                    first_order=first_order
                )

            # 查询集评估（无梯度）
            with torch.no_grad():
                from algorithms.maml import model_forward_with_params
                out = model_forward_with_params(self.model, qx, adapted_params)
                loss = F.cross_entropy(out, qy).item()
                acc = (out.argmax(dim=1) == qy).float().mean().item()

            total_loss += loss
            total_acc += acc

        return total_loss / num_tasks, total_acc / num_tasks

    def train(self, train_dataset, val_dataset=None, first_order=False,
              step_callback: Optional[Callable] = None):
        """完整训练流程"""
        print(f"\n{'=' * 60}")
        print(f"开始TamingMAML训练 | 自适应正则化内循环")
        print(f"{'=' * 60}")

        best_val_acc = 0.0
        start_time = time.time()
        avg_steps = self.config.INNER_STEPS

        pbar = tqdm(range(1, self.config.MAX_EPISODES + 1),
                    desc="TamingMAML",
                    unit="ep",
                    ncols=100)

        for episode in pbar:
            # 采样meta-batch
            sampler = TaskSampler(
                train_dataset,
                self.config.N_WAY,
                self.config.K_SHOT,
                self.config.K_QUERY,
                num_tasks=self.config.META_BATCH_SIZE
            )
            task_batch = [sampler.sample_task() for _ in range(self.config.META_BATCH_SIZE)]

            # 训练
            loss, acc = self.train_step(task_batch, first_order)

            # 定期评估
            if episode % 10 == 0 or episode == 1:
                self.history['train_loss'].append(loss)
                self.history['train_acc'].append(acc)

                val_loss, val_acc = None, None
                if val_dataset:
                    val_loss, val_acc = self.evaluate(val_dataset)
                    self.history['val_loss'].append(val_loss)
                    self.history['val_acc'].append(val_acc)

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc

                pbar.set_postfix({
                    'loss': f'{loss:.3f}',
                    'acc': f'{acc:.2%}',
                    'val': f'{val_acc:.2%}' if val_acc else '-',
                    'best': f'{best_val_acc:.2%}'
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

        pbar.close()
        total_time = time.time() - start_time

        print(f"\n{'=' * 60}")
        print(f"TamingMAML完成！最佳准确率: {best_val_acc:.2%} | 总时间: {total_time / 60:.1f}min")
        print(f"{'=' * 60}")

        return self.history