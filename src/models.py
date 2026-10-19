"""模型构建与冻结/解冻策略（基于 timm 的 EfficientNet）。

timm（PyTorch Image Models）是高效且维护中的预训练模型库，
替代原仓库示例中已停止维护的 efficientnet_pytorch。
"""
import os
import sys

import timm

# huggingface.co 直连不通时的镜像（境内网络常用）
HF_MIRROR = "https://hf-mirror.com"


def build_model(model_name="efficientnet_b0", num_classes=2, pretrained=True):
    """创建模型并自动把分类头输出维度改成 num_classes。

    Args:
        model_name: timm 模型名，如 efficientnet_b0 / efficientnet_b2 / resnet50
        num_classes: 类别数（由训练集目录决定）
        pretrained: 是否加载 ImageNet 预训练权重
    """
    if not pretrained:
        return timm.create_model(model_name, pretrained=False,
                                 num_classes=num_classes)
    try:
        return timm.create_model(model_name, pretrained=True,
                                 num_classes=num_classes)
    except Exception as e:  # noqa: BLE001
        # huggingface_hub 的 ENDPOINT 在 import 时即固定，事后改 os.environ 无效，
        # 必须重启进程。已设过镜像还失败则直接把原始错误抛出去。
        if os.environ.get("HF_ENDPOINT"):
            raise
        print(f"[WARN] 预训练权重下载失败: {type(e).__name__}: {e}")
        print(f"[WARN] 自动切换到 HF 镜像 {HF_MIRROR} 并重启进程...")
        os.environ["HF_ENDPOINT"] = HF_MIRROR
        os.execv(sys.executable, [sys.executable] + sys.argv)


def freeze_backbone(model):
    """冻结除分类头以外的所有参数（第一阶段）。"""
    for p in model.parameters():
        p.requires_grad_(False)
    head = model.get_classifier()  # timm 统一接口，EfficientNet 即最后的 Linear
    for p in head.parameters():
        p.requires_grad_(True)
    return model


def unfreeze(model):
    """解冻全部参数（第二阶段全网络微调）。"""
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
