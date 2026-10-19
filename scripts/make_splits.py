"""把自带划分的数据集整理成统一的 train/val/test 目录。

适用场景：数据集自带 Training/ 与 Testing/（或 train/ 与 test/），
需要再从 train 里切一部分做 val（很多 Kaggle 数据集没有独立 val）。

用法：
    python scripts/make_splits.py --src D:/datasets/mri_brain_raw \
        --dst D:/datasets/mri_brain --train-from Training --test-from Testing --val-ratio 0.15

产物：
    dst/train/<类别>/*.img
    dst/val/<类别>/*.img
    dst/test/<类别>/*.img
"""
import argparse
import os
import random
import shutil

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(d):
    return [os.path.join(d, f) for f in sorted(os.listdir(d))
            if f.lower().endswith(IMG_EXTENSIONS)]


def copy_all(files, dst):
    os.makedirs(dst, exist_ok=True)
    for f in files:
        shutil.copy2(f, os.path.join(dst, os.path.basename(f)))
    return len(files)


def main():
    p = argparse.ArgumentParser(description="整理自带划分的数据集为 train/val/test")
    p.add_argument("--src", required=True, help="源目录（含 <split>/<类别>/*.img）")
    p.add_argument("--dst", required=True, help="输出目录")
    p.add_argument("--train-from", default="Training", help="源训练划分目录名")
    p.add_argument("--test-from", default="Testing", help="源测试划分目录名")
    p.add_argument("--val-ratio", type=float, default=0.15, help="从训练集切多少做验证")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not 0 < args.val_ratio < 1:
        p.error("--val-ratio 必须在 (0, 1) 之间")

    random.seed(args.seed)
    src_train = os.path.join(args.src, args.train_from)
    src_test = os.path.join(args.src, args.test_from)

    counts = {"train": 0, "val": 0, "test": 0}
    classes = sorted(d for d in os.listdir(src_train)
                     if os.path.isdir(os.path.join(src_train, d)))
    print(f"[INFO] 类别: {classes}")

    for cls in classes:
        files = list_images(os.path.join(src_train, cls))
        random.shuffle(files)
        n_val = int(len(files) * args.val_ratio)
        val_files, train_files = files[:n_val], files[n_val:]
        counts["train"] += copy_all(train_files, os.path.join(args.dst, "train", cls))
        counts["val"] += copy_all(val_files, os.path.join(args.dst, "val", cls))
        print(f"  {cls}: {len(files)} -> train {len(train_files)} / val {len(val_files)}")

        src_test_cls = os.path.join(src_test, cls)
        if os.path.isdir(src_test_cls):
            counts["test"] += copy_all(list_images(src_test_cls),
                                       os.path.join(args.dst, "test", cls))
        else:
            print(f"  [WARN] {src_test_cls} 不存在，{cls} 无 test 数据")

    print(f"\n[DONE] " + " / ".join(f"{k} {v}" for k, v in counts.items()))
    print(f"训练命令: python src/train.py --data-root {args.dst} --model efficientnet_b0 --epochs 20 --amp")


if __name__ == "__main__":
    main()
