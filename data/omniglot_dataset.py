# omniglot_dataset.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import random
from config import Config


class OmniglotDataset(Dataset):
    """
    Omniglot数据集加载器
    结构: omniglot_data/images_background/alphabet/character/*.png
    """

    def __init__(self, root, transform=None, background=True):
        self.root = root
        self.transform = transform if transform else self.default_transform()

        # 选择background或evaluation集
        split = "images_background" if background else "images_evaluation"
        data_path = os.path.join(root, split)

        # 收集所有图像路径和标签
        self.data = []
        self.labels = []
        label_idx = 0

        # 遍历所有字母表和字符
        for alphabet in sorted(os.listdir(data_path)):
            alphabet_path = os.path.join(data_path, alphabet)
            if not os.path.isdir(alphabet_path):
                continue

            for character in sorted(os.listdir(alphabet_path)):
                char_path = os.path.join(alphabet_path, character)
                if not os.path.isdir(char_path):
                    continue

                # 该字符的所有图像
                images = glob.glob(os.path.join(char_path, "*.png"))
                for img_path in images:
                    self.data.append(img_path)
                    self.labels.append(label_idx)

                label_idx += 1  # 每个字符一个唯一标签

        self.num_classes = label_idx
        print(f"📊 加载Omniglot {'images_background' if background else 'evaluation'}集")
        print(f"   总类别数: {self.num_classes}")
        print(f"   总样本数: {len(self.data)}")

    def default_transform(self):
        return transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1, 1]
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('L')  # 灰度图
        if self.transform:
            image = self.transform(image)

        return image, label


class TaskSampler:
    """
    N-way K-shot 任务采样器
    每个任务包含: K个支持样本 + K_QUERY个查询样本
    """

    def __init__(self, dataset, n_way, k_shot, k_query, num_tasks):
        self.dataset = dataset
        self.n_way = n_way  # N-way
        self.k_shot = k_shot  # K-shot (支持集每类样本数)
        self.k_query = k_query  # 查询集每类样本数
        self.num_tasks = num_tasks  # 总任务数

        # 按类别组织数据索引
        self.class_to_indices = {}
        for idx, label in enumerate(dataset.labels):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)

        self.classes = list(self.class_to_indices.keys())

    def sample_task(self):
        """
        采样一个N-way K-shot任务
        返回: (support_x, support_y, query_x, query_y)
        """
        # 随机选择N个类别
        selected_classes = random.sample(self.classes, self.n_way)

        support_x, support_y = [], []
        query_x, query_y = [], []

        for class_idx, cls in enumerate(selected_classes):
            # 从该类中采样 (k_shot + k_query) 个样本
            indices = self.class_to_indices[cls]
            selected = random.sample(indices, self.k_shot + self.k_query)

            # 前k_shot个作为支持集
            for i in range(self.k_shot):
                img, _ = self.dataset[selected[i]]
                support_x.append(img)
                support_y.append(class_idx)  # 任务内重新编号0~N-1

            # 后k_query个作为查询集
            for i in range(self.k_shot, self.k_shot + self.k_query):
                img, _ = self.dataset[selected[i]]
                query_x.append(img)
                query_y.append(class_idx)

        # 转换为tensor
        support_x = torch.stack(support_x)  # [N*K, 1, 28, 28]
        support_y = torch.tensor(support_y)  # [N*K]
        query_x = torch.stack(query_x)  # [N*Q, 1, 28, 28]
        query_y = torch.tensor(query_y)  # [N*Q]

        return support_x, support_y, query_x, query_y

    def __iter__(self):
        for _ in range(self.num_tasks):
            yield self.sample_task()

    def __len__(self):
        return self.num_tasks


# 测试代码
if __name__ == "__main__":
    from config import Config

    Config.set_seed()

    # 测试数据集加载
    print("\n" + "=" * 50)
    print("测试1: 数据集加载")
    print("=" * 50)

    # 注意：你需要先下载Omniglot数据集
    # 如果没有数据，这里会报错，我们先创建模拟数据测试逻辑
    try:
        dataset = OmniglotDataset(
            root=Config.DATA_ROOT,
            background=True
        )

        # 测试TaskSampler
        print("\n" + "=" * 50)
        print("测试2: 任务采样")
        print("=" * 50)

        sampler = TaskSampler(
            dataset=dataset,
            n_way=Config.N_WAY,
            k_shot=Config.K_SHOT,
            k_query=Config.K_QUERY,
            num_tasks=3  # 只采样3个任务测试
        )

        for i, (sx, sy, qx, qy) in enumerate(sampler):
            print(f"\n任务 {i + 1}:")
            print(f"  支持集: {sx.shape}, 标签: {sy.shape}")
            print(f"  查询集: {qx.shape}, 标签: {qy.shape}")
            print(f"  支持集标签分布: {sy.bincount().tolist()}")
            print(f"  查询集标签分布: {qy.bincount().tolist()}")

            # 验证标签范围
            assert sy.max() < Config.N_WAY and sy.min() >= 0
            assert qy.max() < Config.N_WAY and qy.min() >= 0

        print("\n✅ 所有测试通过!")

    except FileNotFoundError as e:
        print(f"\n⚠️  数据集未找到: {e}")
        print("请下载Omniglot数据集并解压到 ./omniglot_data/")
        print("下载地址: https://github.com/brendenlake/omniglot")
        print("\n数据集结构应为:")
        print("  omniglot_data/")
        print("    images_background/")
        print("      Alphabet_of_the_Magi/")
        print("        character01/")
        print("          0001_01.png ...")