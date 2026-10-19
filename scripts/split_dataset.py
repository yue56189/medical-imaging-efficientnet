"""把扁平数据集划分为 train/val/test（分层抽样 + 固定随机种子）。

适用场景：Kaggle 下载下来只有 <类别>/*.img 一层目录（如 navoneel 脑部 MRI），
或官方 val 集太小（如 paultimothymooney 肺炎数据集的 val 只有 16 张）。

用法：
    python scripts/split_dataset.py --src data/raw/brain_tumor --dst data/mri_brain \
        --ratios 0.7 0.15 0.15

参数：
    --src     源目录，需含 <类别名>/*.png|jpg 子目录
    --dst     输出目录，会生成 dst/{train,val,test}/<类别名>/
    --ratios  train/val/test 比例，默认 0.7 0.15 0.15
    --seed    随机种子（保证可复现）
    --move    用移动代替复制（默认复制，源数据保留）
"""
import argparse
import os
import random
import shutil

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def split_class(files, ratios):
    random.shuffle(files)
    n = len(files)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    # 余数全部给 test，保证总数不变
    return (files[:n_train],
            files[n_train:n_train + n_val],
            files[n_train + n_val:])


def main():
    p = argparse.ArgumentParser(description="扁平数据集分层划分为 train/val/test")
    p.add_argument("--src", required=True, help="源目录（含 <类别名>/*.img）")
    p.add_argument("--dst", required=True, help="输出目录")
    p.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15],
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--move", action="store_true", help="移动而非复制")
    args = p.parse_args()

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        p.error("ratios 之和必须为 1.0")

    random.seed(args.seed)
    classes = sorted(d for d in os.listdir(args.src)
                     if os.path.isdir(os.path.join(args.src, d)))
    if not classes:
        print(f"[ERROR] {args.src} 下没有类别子目录")
        raise SystemExit(1)

    splits = ("train", "val", "test")
    summary = {s: 0 for s in splits}
    for cls in classes:
        src_cls = os.path.join(args.src, cls)
        files = [os.path.join(src_cls, f) for f in sorted(os.listdir(src_cls))
                 if f.lower().endswith(IMG_EXTENSIONS)]
        if len(files) < 3:
            print(f"[WARN] 类别 {cls} 只有 {len(files)} 张，跳过")
            continue
        parts = split_class(files, args.ratios)
        for split, part in zip(splits, parts):
            dst_dir = os.path.join(args.dst, split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for f in part:
                if args.move:
                    shutil.move(f, os.path.join(dst_dir, os.path.basename(f)))
                else:
                    shutil.copy2(f, os.path.join(dst_dir, os.path.basename(f)))
            summary[split] += len(part)
        print(f"类别 {cls}: {len(files)} 张 -> "
              f"train {len(parts[0])} / val {len(parts[1])} / test {len(parts[2])}")

    print(f"\n[DONE] 总计: " + " / ".join(f"{s} {summary[s]}" for s in splits))
    print(f"训练命令: python src/train.py --data-root {args.dst} --model efficientnet_b0 --epochs 20 --amp")


if __name__ == "__main__":
    main()
