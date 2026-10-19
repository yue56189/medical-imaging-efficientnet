"""两阶段迁移学习训练脚本。

流程：
  阶段一（freeze）：冻结 backbone，只训练分类头，学习率较大（--lr-head）
  阶段二（finetune）：解冻全网络，小学习率全量微调（--lr），余弦退火 + 早停

用法（在项目根目录执行）：
    python src/train.py --data-root data/chest_xray_pneumonia/chest_xray \
        --model efficientnet_b0 --epochs 20 --freeze-epochs 3 --amp
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from sklearn.metrics import f1_score

from config import parse_args
from dataset import MedicalImageDataset, build_class_weights
from models import build_model, freeze_backbone, unfreeze, count_parameters
from utils import (set_seed, get_device, AverageMeter, make_run_dir,
                   save_json, save_checkpoint, plot_curves)

# ImageNet 预训练统计量（灰度医学图已转 3 通道，复用这组均值方差）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size):
    """训练/验证 transforms。

    医学影像增强的两条纪律：
    1. 只做水平翻转 —— 垂直翻转会颠倒解剖方位（X 光胸片尤其忌讳）；
    2. 用双线性插值 —— 双三次插值在锐利边缘（骨骼、异物）上会产生振铃伪影。
    """
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(
            img_size, scale=(0.85, 1.0), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15, interpolation=InterpolationMode.BILINEAR),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224), interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.autocast(device_type=autocast_device, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return loss_meter.avg, acc_meter.avg, macro_f1


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    run_dir = make_run_dir(args.output)
    save_json(vars(args), os.path.join(run_dir, "config.json"))
    print(f"[INFO] device={device} | run_dir={run_dir}")
    if device.type == "cpu":
        print("[WARN] 未检测到 CUDA，将使用 CPU 训练（极慢，建议只用于冒烟测试）")

    train_tf, val_tf = build_transforms(args.img_size)
    train_ds = MedicalImageDataset(os.path.join(args.data_root, "train"), transform=train_tf)
    # val 复用 train 的 class_to_idx，保证标签编码一致；val 缺类时自动跳过
    val_ds = MedicalImageDataset(
        os.path.join(args.data_root, "val"), transform=val_tf,
        class_to_idx=train_ds.class_to_idx)

    num_classes = len(train_ds.classes)
    print(f"[INFO] 类别（{num_classes}）: {train_ds.classes}")
    print(f"[INFO] train={len(train_ds)} 张, val={len(val_ds)} 张")
    if len(val_ds) < 50:
        print(f"[WARN] val 集只有 {len(val_ds)} 张，指标波动会很大，建议用 scripts/split_dataset.py 重新划分")

    class_weights = build_class_weights(train_ds.samples, num_classes, device)
    print(f"[INFO] 类别权重: {[round(w, 3) for w in class_weights.tolist()]}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.workers,
                         pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = build_model(args.model, num_classes=num_classes,
                        pretrained=not args.no_pretrained)
    total, trainable = count_parameters(model)
    print(f"[INFO] 模型 {args.model}: 总参数 {total/1e6:.1f}M，当前可训练 {trainable/1e6:.1f}M")
    model.to(device)

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 两阶段配置：(阶段名, epoch 数, 学习率)
    stages = []
    if args.freeze_epochs > 0:
        stages.append(("freeze", args.freeze_epochs, args.lr_head))
    stages.append(("finetune", args.epochs, args.lr))

    history = []
    best_f1 = -1.0
    global_epoch = 0
    stop_all = False

    for stage_name, stage_epochs, stage_lr in stages:
        if stop_all:
            break
        if stage_name == "freeze":
            freeze_backbone(model)
        else:
            unfreeze(model)
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(params, lr=stage_lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage_epochs)
        no_improve = 0
        print(f"[INFO] === 阶段 {stage_name}: {stage_epochs} epochs, lr={stage_lr} ===")

        for _ in range(stage_epochs):
            global_epoch += 1
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device, use_amp)
            va_loss, va_acc, va_f1 = validate(model, val_loader, criterion, device)
            scheduler.step()
            history.append({
                "epoch": global_epoch, "stage": stage_name,
                "lr": stage_lr,
                "train_loss": round(tr_loss, 4), "val_loss": round(va_loss, 4),
                "train_acc": round(tr_acc, 4), "val_acc": round(va_acc, 4),
                "val_f1": round(va_f1, 4),
            })
            print(f"epoch {global_epoch:>3} [{stage_name}] "
                  f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
                  f"train_acc={tr_acc:.4f} val_acc={va_acc:.4f} val_f1={va_f1:.4f}")

            if va_f1 > best_f1:
                best_f1 = va_f1
                no_improve = 0
                save_checkpoint({
                    "epoch": global_epoch,
                    "stage": stage_name,
                    "model_name": args.model,
                    "img_size": args.img_size,
                    "class_to_idx": train_ds.class_to_idx,
                    "val_f1": round(va_f1, 4),
                    "val_acc": round(va_acc, 4),
                    "model_state": model.state_dict(),
                }, os.path.join(run_dir, "best.pt"))
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"[INFO] 早停：{args.patience} 轮无提升，结束阶段 {stage_name}")
                    stop_all = (stage_name == "finetune")
                    break

    save_checkpoint({
        "epoch": global_epoch,
        "model_name": args.model,
        "img_size": args.img_size,
        "class_to_idx": train_ds.class_to_idx,
        "model_state": model.state_dict(),
    }, os.path.join(run_dir, "last.pt"))

    save_json({
        "classes": train_ds.classes,
        "class_to_idx": train_ds.class_to_idx,
        "history": history,
        "best_val_f1": round(best_f1, 4),
    }, os.path.join(run_dir, "metrics.json"))
    plot_curves(history, os.path.join(run_dir, "curves.png"))

    print(f"[DONE] 最佳 val macro-F1 = {best_f1:.4f}")
    print(f"[DONE] best.pt / metrics.json / curves.png 已保存到 {run_dir}")


if __name__ == "__main__":
    main()
