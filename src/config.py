"""命令行参数配置。所有脚本共用。"""
import argparse


def parse_args():
    p = argparse.ArgumentParser(
        description="EfficientNet 医疗影像分类（迁移学习微调）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", required=True,
                   help="数据根目录，需包含 train/ 与 val/ 子目录，每个子目录下按类别名分文件夹")
    p.add_argument("--model", default="efficientnet_b0",
                   help="timm 模型名：efficientnet_b0/b1/b2/b3、resnet50 等均可")
    p.add_argument("--img-size", type=int, default=224, help="输入图像边长（正方形）")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30,
                   help="第二阶段（全网络微调）epoch 数")
    p.add_argument("--freeze-epochs", type=int, default=3,
                   help="第一阶段（冻结 backbone、只训分类头）epoch 数；0 表示跳过")
    p.add_argument("--lr-head", type=float, default=1e-3, help="第一阶段学习率")
    p.add_argument("--lr", type=float, default=1e-4, help="第二阶段学习率")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5,
                   help="早停：验证 macro-F1 连续多少轮不提升就停止当前阶段")
    p.add_argument("--workers", type=int, default=4, help="DataLoader 进程数，Windows 报错可设 0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None, help="输出目录，默认 runs/<时间戳>")
    p.add_argument("--amp", action="store_true", help="开启混合精度训练（有 GPU 时推荐）")
    p.add_argument("--no-pretrained", action="store_true",
                   help="不使用 ImageNet 预训练权重（仅调试用）")
    return p.parse_args()
