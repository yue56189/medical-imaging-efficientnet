"""在 test 集上做最终评估：分类报告 + 混淆矩阵 + ROC 曲线。

用法：
    python src/evaluate.py --checkpoint runs/xxx/best.pt --data-root data/xxx
（--data-root 下需有 test/ 子目录）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import MedicalImageDataset
from models import build_model
from utils import get_device, save_json
from train import build_transforms


@torch.no_grad()
def main():
    import argparse
    p = argparse.ArgumentParser(description="EfficientNet 医疗影像 test 集评估")
    p.add_argument("--checkpoint", required=True, help="best.pt 路径")
    p.add_argument("--data-root", required=True, help="含 test/ 子目录的数据根目录")
    p.add_argument("--output", default=None, help="输出目录，默认 <checkpoint所在目录>/eval")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_to_idx = ckpt["class_to_idx"]
    classes = sorted(class_to_idx, key=class_to_idx.get)
    num_classes = len(classes)

    model = build_model(ckpt["model_name"], num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    _, val_tf = build_transforms(ckpt["img_size"])
    test_ds = MedicalImageDataset(
        os.path.join(args.data_root, "test"), transform=val_tf,
        class_to_idx=class_to_idx)
    print(f"[INFO] test 集: {len(test_ds)} 张, 类别: {classes}")

    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=(device.type == "cuda"))
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        probs = torch.softmax(model(images), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(labels.tolist())
    probs = np.concatenate(all_probs)
    preds = probs.argmax(axis=1)

    # ---- 指标 ----
    report = classification_report(all_labels, preds, target_names=classes,
                                   output_dict=True, zero_division=0)
    acc = float((preds == np.array(all_labels)).mean())

    # macro AUC（one-vs-rest）
    per_class_auc = {}
    for i, cls in enumerate(classes):
        y_true = (np.array(all_labels) == i).astype(int)
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            per_class_auc[cls] = None
            continue
        fpr, tpr, _ = roc_curve(y_true, probs[:, i])
        per_class_auc[cls] = float(auc(fpr, tpr))
    valid_aucs = [v for v in per_class_auc.values() if v is not None]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else None

    output_dir = args.output or os.path.join(os.path.dirname(args.checkpoint), "eval")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + classification_report(all_labels, preds, target_names=classes, zero_division=0))
    print(f"accuracy = {acc:.4f}")
    if macro_auc is not None:
        print(f"macro AUC (OvR) = {macro_auc:.4f}")

    save_json({
        "accuracy": round(acc, 4),
        "macro_auc": round(macro_auc, 4) if macro_auc is not None else None,
        "per_class_auc": {k: (round(v, 4) if v is not None else None)
                          for k, v in per_class_auc.items()},
        "classification_report": report,
        "confusion_matrix": confusion_matrix(all_labels, preds).tolist(),
        "classes": classes,
        "num_test_images": len(test_ds),
    }, os.path.join(output_dir, "metrics.json"))

    # ---- 混淆矩阵图 ----
    cm = confusion_matrix(all_labels, preds)
    fig, ax = plt.subplots(figsize=(1.2 + 0.9 * num_classes, 1.2 + 0.9 * num_classes))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Confusion Matrix (acc={acc:.3f})")
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # ---- ROC 曲线图 ----
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, cls in enumerate(classes):
        y_true = (np.array(all_labels) == i).astype(int)
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue
        fpr, tpr, _ = roc_curve(y_true, probs[:, i])
        ax.plot(fpr, tpr, label=f"{cls} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (one-vs-rest)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=150)
    plt.close(fig)

    print(f"[DONE] metrics.json / confusion_matrix.png / roc_curves.png 已保存到 {output_dir}")


if __name__ == "__main__":
    main()
