"""通用工具：随机种子、设备、日志、指标曲线绘制、checkpoint 保存。"""
import json
import os
import random
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # 无显示环境
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AverageMeter:
    """统计一个 epoch 内的平均 loss / 准确率。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def make_run_dir(output=None):
    if output is None:
        output = os.path.join("runs", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output, exist_ok=True)
    return output


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_checkpoint(state, path):
    torch.save(state, path)


def plot_curves(history, path):
    """画训练/验证的 loss 与 accuracy 曲线。

    history: [{"stage", "train_loss", "val_loss", "train_acc", "val_acc", ...}, ...]
    """
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, [h["train_acc"] for h in history], label="train")
    ax2.plot(epochs, [h["val_acc"] for h in history], label="val")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 标注两阶段的分界
    stage_boundaries = [
        i for i in range(1, len(history))
        if history[i]["stage"] != history[i - 1]["stage"]
    ]
    for i in stage_boundaries:
        for ax in (ax1, ax2):
            ax.axvline(epochs[i] - 0.5, color="gray", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
