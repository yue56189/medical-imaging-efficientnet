"""医学影像数据集加载器（修正版）。

相对原 GitHub 仓库示例代码的修正：
1. 目录只扫描一次并缓存样本列表；原示例每个样本都执行
   list(self.annotations.items())[idx]，10 万张图会慢到不可用；
2. 用 PIL 读图并统一转 RGB，规避 cv2.imread 的 BGR 通道顺序问题；
3. 标签索引由目录名自动生成，val/test 可复用训练集的 class_to_idx，
   保证编码一致、不会因缺类错位；
4. 附带 build_class_weights()，用于处理医学数据常见的类别不平衡。

目录结构（ImageFolder 风格）：
    data_root/
    ├── train/<class_name>/*.png
    ├── val/<class_name>/*.png
    └── test/<class_name>/*.png   （可选，evaluate.py 用）
"""
import os

from PIL import Image
from torch.utils.data import Dataset

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class MedicalImageDataset(Dataset):
    def __init__(self, root, transform=None, class_to_idx=None):
        """
        Args:
            root: 数据集目录（如 data/chest_xray/train）
            transform: torchvision transforms
            class_to_idx: 外部传入的标签映射；为 None 时按目录名自动生成（排序后）
        """
        self.root = root
        self.transform = transform

        if class_to_idx is None:
            classes = sorted(
                d for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))
            )
            class_to_idx = {c: i for i, c in enumerate(classes)}
        self.class_to_idx = class_to_idx
        self.classes = list(class_to_idx.keys())

        # 一次性扫描，建样本列表 [(path, label_idx), ...]
        self.samples = []
        for cls_name, idx in class_to_idx.items():
            cls_dir = os.path.join(root, cls_name)
            if not os.path.isdir(cls_dir):
                continue  # val/test 可能缺少某些类别，跳过即可
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(IMG_EXTENSIONS):
                    self.samples.append((os.path.join(cls_dir, fname), idx))

        if not self.samples:
            raise FileNotFoundError(
                f"目录 {root} 下未找到图像。请确认目录结构为 <root>/<类别名>/*.png|jpg"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as im:
            image = im.convert("RGB")  # 统一 3 通道，兼容灰度医学图
        if self.transform:
            image = self.transform(image)
        return image, label


def build_class_weights(samples, num_classes, device="cpu"):
    """按 sklearn 的 'balanced' 策略计算类别权重：w_c = N / (K * count_c)。

    Returns:
        torch.FloatTensor，长度 num_classes
    """
    import torch
    counts = [0] * num_classes
    for _, label in samples:
        counts[label] += 1
    total = sum(counts)
    weights = [total / (num_classes * c) if c > 0 else 0.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32, device=device)
