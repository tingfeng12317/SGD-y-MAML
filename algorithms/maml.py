# algorithms/maml.py
import sys

sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
from collections import OrderedDict
from typing import Callable, Optional

from config import Config
from models.omniglot_net import OmniglotNet
from data.omniglot_dataset import TaskSampler


def inner_loop(model, support_x, support_y, inner_steps, inner_lr, first_order=False):
    """
    MAML内循环：在支持集上进行几步梯度下降
    """
    # 克隆当前参数作为起点（不修改原模型！）
    adapted_params = {name: param.clone() for name, param in model.named_parameters()}

    for step in range(inner_steps):
        # 使用当前适应参数前向传播
        out = model_forward_with_params(model, support_x, adapted_params)
        loss = F.cross_entropy(out, support_y)

        # 计算梯度（对adapted_params）
        grads = torch.autograd.grad(
            loss,
            adapted_params.values(),
            create_graph=not first_order,  # 二阶MAML需要保留计算图
            retain_graph=True,
            allow_unused=True
        )

        # 手动SGD更新adapted_params
        adapted_params = {
            name: param - inner_lr * grad if grad is not None else param
            for (name, param), grad in zip(adapted_params.items(), grads)
        }

    return adapted_params, loss.item()


def model_forward_with_params(model, x, params):
    """
    使用自定义参数进行前向传播（functional接口）
    """
    # Layer 1
    x = F.conv2d(x, params['layer1.conv.weight'], params['layer1.conv.bias'], padding=1)
    x = F.batch_norm(x,
                     running_mean=model.layer1.bn.running_mean,
                     running_var=model.layer1.bn.running_var,
                     weight=params['layer1.bn.weight'],
                     bias=params['layer1.bn.bias'],
                     training=True)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)

    # Layer 2
    x = F.conv2d(x, params['layer2.conv.weight'], params['layer2.conv.bias'], padding=1)
    x = F.batch_norm(x,
                     running_mean=model.layer2.bn.running_mean,
                     running_var=model.layer2.bn.running_var,
                     weight=params['layer2.bn.weight'],
                     bias=params['layer2.bn.bias'],
                     training=True)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)

    # Layer 3
    x = F.conv2d(x, params['layer3.conv.weight'], params['layer3.conv.bias'], padding=1)
    x = F.batch_norm(x,
                     running_mean=model.layer3.bn.running_mean,
                     running_var=model.layer3.bn.running_var,
                     weight=params['layer3.bn.weight'],
                     bias=params['layer3.bn.bias'],
                     training=True)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)

    # Layer 4 (无pool)
    x = F.conv2d(x, params['layer4.0.weight'], params['layer4.0.bias'], padding=1)
    x = F.batch_norm(x,
                     running_mean=model.layer4[1].running_mean,
                     running_var=model.layer4[1].running_var,
                     weight=params['layer4.1.weight'],
                     bias=params['layer4.1.bias'],
                     training=True)
    x = F.relu(x)

    # Flatten + FC
    x = x.view(x.size(0), -1)
    x = F.linear(x, params['fc.weight'], params['fc.bias'])

    return x


def sample_task_batch(dataset, batch_size, n_way, k_shot, k_query):
    """采样一个meta-batch的任务"""
    sampler = TaskSampler(dataset, n_way, k_shot, k_query, num_tasks=batch_size)
    return [sampler.sample_task() for _ in range(batch_size)]


def outer_step(model, meta_optimizer, task_batch, inner_steps, inner_lr, first_order=False):
    """
    MAML外循环：采样多个任务，内循环适应，查询集上计算loss，元更新
    """
    meta_optimizer.zero_grad()

    total_loss = 0.0
    total_acc = 0.0
    num_tasks = len(task_batch)

    for support_x, support_y, query_x, query_y in task_batch:
        # 将数据移到设备
        support_x = support_x.to(Config.DEVICE)
        support_y = support_y.to(Config.DEVICE)
        query_x = query_x.to(Config.DEVICE)
        query_y = query_y.to(Config.DEVICE)

        # 1. 内循环：在支持集上适应
        adapted_params, _ = inner_loop(
            model, support_x, support_y,
            inner_steps, inner_lr, first_order
        )

        # 2. 在查询集上评估适应效果（这才是元损失！）
        query_out = model_forward_with_params(model, query_x, adapted_params)
        task_loss = F.cross_entropy(query_out, query_y)

        # 3. 反向传播（会经过内循环的二阶导数，如果first_order=False）
        task_loss.backward()

        # 统计
        total_loss += task_loss.item()
        pred = query_out.argmax(dim=1)
        total_acc += (pred == query_y).float().mean().item()

    # 4. 元更新（所有任务的梯度已累加）
    meta_optimizer.step()

    return total_loss / num_tasks, total_acc / num_tasks


class MAML:
    """
    MAML完整封装：整合内循环、外循环、训练、评估
    """

    def __init__(self, model, config=None):
        self.config = config or Config
        self.model = model.to(self.config.DEVICE)
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.META_LR
        )
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

    def train_step(self, task_batch, first_order=False):
        """单次训练步骤"""
        self.model.train()
        loss, acc = outer_step(
            self.model, self.meta_optimizer, task_batch,
            inner_steps=self.config.INNER_STEPS,
            inner_lr=self.config.INNER_LR,
            first_order=first_order
        )
        return loss, acc

    def evaluate(self, dataset, num_tasks=100, first_order=True):
        """在验证集上评估（无梯度更新）"""
        self.model.eval()
        total_loss, total_acc = 0.0, 0.0

        for _ in range(num_tasks):
            # 采样一个任务
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

            # 内循环适应（需要梯度！但不用反向传播到元参数）
            with torch.enable_grad():  # 确保梯度开启
                adapted_params, _ = inner_loop(
                    self.model, sx, sy,
                    self.config.INNER_STEPS,
                    self.config.INNER_LR,
                    first_order
                )

            # 查询集评估（无梯度）
            with torch.no_grad():
                out = model_forward_with_params(self.model, qx, adapted_params)
                loss = F.cross_entropy(out, qy).item()
                acc = (out.argmax(dim=1) == qy).float().mean().item()

            total_loss += loss
            total_acc += acc

        return total_loss / num_tasks, total_acc / num_tasks

    def train(self, train_dataset, val_dataset=None, first_order=False,
              step_callback: Optional[Callable] = None):
        """
        完整训练流程 - 与SGD_Y_MAML接口保持一致
        step_callback(episode, train_loss, train_acc, val_loss, val_acc, avg_steps, best_val_acc, model)
        注意：MAML的avg_steps为固定值INNER_STEPS
        """
        print(f"\n{'=' * 60}")
        print(f"开始MAML训练 | 固定{self.config.INNER_STEPS}步内循环")
        print(f"{'=' * 60}")

        best_val_acc = 0.0
        start_time = time.time()
        avg_steps = self.config.INNER_STEPS  # MAML固定步数

        # 使用tqdm但不频繁打印
        pbar = tqdm(range(1, self.config.MAX_EPISODES + 1),
                    desc="MAML",
                    unit="ep",
                    ncols=100,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')

        for episode in pbar:
            # 采样meta-batch
            task_batch = sample_task_batch(
                train_dataset,
                self.config.META_BATCH_SIZE,
                self.config.N_WAY,
                self.config.K_SHOT,
                self.config.K_QUERY
            )

            # 训练
            loss, acc = self.train_step(task_batch, first_order)

            # 每10步评估和触发回调（与SGD_Y_MAML一致）
            if episode % 10 == 0 or episode == 1:
                self.history['train_loss'].append(loss)
                self.history['train_acc'].append(acc)

                val_loss, val_acc = None, None
                if val_dataset:
                    val_loss, val_acc = self.evaluate(val_dataset, num_tasks=100)
                    self.history['val_loss'].append(val_loss)
                    self.history['val_acc'].append(val_acc)

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc

                # 更新进度条（简洁显示）
                pbar.set_postfix({
                    'loss': f'{loss:.3f}',
                    'acc': f'{acc:.2%}',
                    'val': f'{val_acc:.2%}' if val_acc else '-',
                    'best': f'{best_val_acc:.2%}'
                })

                # 🔥 触发回调（与SGD_Y_MAML完全一致）
                if step_callback:
                    step_callback(
                        episode=episode,
                        train_loss=loss,
                        train_acc=acc,
                        val_loss=val_loss,
                        val_acc=val_acc,
                        avg_steps=avg_steps,  # MAML固定步数
                        best_val_acc=best_val_acc,
                        model=self.model  # 传入模型引用
                    )

        pbar.close()

        total_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"完成！最佳准确率: {best_val_acc:.2%} | 总时间: {total_time / 60:.1f}min")
        print(f"{'=' * 60}")
        return self.history