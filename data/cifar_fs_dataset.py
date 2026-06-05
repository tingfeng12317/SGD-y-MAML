import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import random


class CifarFSDataset(Dataset):
    """
    CIFAR-FS Dataset for Few-Shot Learning
    支持pickle格式，包含标准数据增强
    """

    # CIFAR-FS的标准归一化参数（基于CIFAR-10统计）
    CIFAR_MEAN = [0.485, 0.456, 0.406]
    CIFAR_STD = [0.229, 0.224, 0.225]

    def __init__(self, data_root, split='train', transform=None, augment=False):
        """
        Args:
            data_root: CIFAR-FS数据目录
            split: 'train', 'val', 或 'test'
            transform: 自定义transform（None则使用默认）
            augment: 是否使用数据增强（仅训练时建议开启）
        """
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.augment = augment

        # 加载pickle文件
        pkl_path = os.path.join(data_root, f'cifar_fs_{split}.pickle')
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"找不到文件: {pkl_path}")

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='bytes')

        # 处理不同可能的键名（兼容b'labels'和'labels'）
        if b'data' in data:
            self.images = data[b'data']  # (N, 32, 32, 3) uint8
            self.labels = data[b'labels']
        else:
            self.images = data['data']
            self.labels = data['labels']

        # 确保numpy数组格式正确
        if isinstance(self.images, list):
            self.images = np.array(self.images)
        if isinstance(self.labels, list):
            self.labels = np.array(self.labels)

        # 如果是展平格式 (N, 3072)，重塑为 (N, 32, 32, 3)
        if self.images.ndim == 2:
            self.images = self.images.reshape(-1, 32, 32, 3)

        self.num_samples = len(self.images)

        # 构建类别到索引的映射
        self.label_to_indices = {}
        unique_labels = np.unique(self.labels)
        for label in unique_labels:
            self.label_to_indices[label] = np.where(self.labels == label)[0].tolist()

        self.classes = list(self.label_to_indices.keys())
        self.num_classes = len(self.classes)

        # 设置transform
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transform(augment)

        print(f"✅ CIFAR-FS [{split}] 加载完成: {self.num_samples}张图片, {self.num_classes}个类别")

    def _get_default_transform(self, augment):
        """获取默认transform，训练时包含数据增强"""
        if augment:
            # 训练时：强数据增强（对Few-shot学习至关重要）
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.CIFAR_MEAN, std=self.CIFAR_STD)
            ])
        else:
            # 验证/测试时：仅归一化
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.CIFAR_MEAN, std=self.CIFAR_STD)
            ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = self.images[idx]  # (32, 32, 3) uint8
        label = int(self.labels[idx])

        # 应用transform
        if self.transform:
            img = self.transform(img)

        return img, label

    def get_task(self, n_way, k_shot, k_query, seed=None):
        """
        采样一个N-way K-shot任务

        Returns:
            support_x: (n_way * k_shot, 3, 32, 32)
            support_y: (n_way * k_shot,)
            query_x: (n_way * k_query, 3, 32, 32)
            query_y: (n_way * k_query,)
        """
        if seed is not None:
            random_state = random.getstate()
            random.seed(seed)
            np_random_state = np.random.get_state()
            np.random.seed(seed)
        else:
            random_state = None
            np_random_state = None

        try:
            # 随机选择n_way个类别
            selected_classes = random.sample(self.classes, n_way)

            support_x = []
            support_y = []
            query_x = []
            query_y = []

            for class_idx, cls in enumerate(selected_classes):
                # 从该类别中采样k_shot + k_query张图片
                indices = random.sample(self.label_to_indices[cls], k_shot + k_query)

                support_indices = indices[:k_shot]
                query_indices = indices[k_shot:]

                # 加载support set
                for idx in support_indices:
                    img = self.images[idx]
                    if self.transform:
                        img = self.transform(img)
                    support_x.append(img)
                    support_y.append(class_idx)  # 使用0-n_way-1作为任务内标签

                # 加载query set
                for idx in query_indices:
                    img = self.images[idx]
                    if self.transform:
                        img = self.transform(img)
                    query_x.append(img)
                    query_y.append(class_idx)

            support_x = torch.stack(support_x)
            support_y = torch.tensor(support_y, dtype=torch.long)
            query_x = torch.stack(query_x)
            query_y = torch.tensor(query_y, dtype=torch.long)

            return support_x, support_y, query_x, query_y

        finally:
            # 恢复随机状态
            if random_state is not None:
                random.setstate(random_state)
            if np_random_state is not None:
                np.random.set_state(np_random_state)

    def get_task_batch(self, batch_size, n_way, k_shot, k_query):
        """
        获取一个batch的tasks（用于meta-training）
        返回列表，每个元素是一个task tuple
        """
        tasks = []
        for _ in range(batch_size):
            task = self.get_task(n_way, k_shot, k_query)
            tasks.append(task)
        return tasks