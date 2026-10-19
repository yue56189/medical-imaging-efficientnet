"""推理脚本：对单张图片或整个目录做预测，并输出 Grad-CAM 热力图。

用法：
    python src/predict.py --checkpoint runs/xxx/best.pt --image test.png
    python src/predict.py --checkpoint runs/xxx/best.pt --image data/xxx/test/NORMAL --output outputs/predict
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from PIL import Image

from gradcam import GradCAM, overlay_heatmap, load_inference_model
from train import build_transforms
from utils import get_device

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def main():
    import argparse
    p = argparse.ArgumentParser(description="医疗影像推理 + Grad-CAM")
    p.add_argument("--checkpoint", required=True, help="best.pt 路径")
    p.add_argument("--image", required=True, help="单张图片或图片目录")
    p.add_argument("--output", default="outputs/predict")
    p.add_argument("--no-gradcam", action="store_true", help="只输出预测，不画热力图")
    args = p.parse_args()

    device = get_device()
    model, class_to_idx, img_size, device = load_inference_model(args.checkpoint, device)
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    _, val_tf = build_transforms(img_size)

    # 收集待预测图片（目录时递归遍历，支持 test/<类别>/*.jpg 结构）
    if os.path.isdir(args.image):
        paths = []
        for root, _, files in os.walk(args.image):
            for f in sorted(files):
                if f.lower().endswith(IMG_EXTENSIONS):
                    paths.append(os.path.join(root, f))
    else:
        paths = [args.image]
    if not paths:
        print(f"[ERROR] {args.image} 下没有图片")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    print(f"{'file':<50} {'prediction':<20} confidence")
    print("-" * 85)
    for path in paths:
        image_pil = Image.open(path).convert("RGB")
        x = val_tf(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
        cls_idx = int(probs.argmax().item())
        name = os.path.basename(path)
        print(f"{name:<50} {idx_to_class[cls_idx]:<20} {probs[cls_idx].item():.4f}")

        if not args.no_gradcam:
            cam_tool = GradCAM(model)
            with torch.enable_grad():
                cam, _ = cam_tool.generate(x)
            cam_tool.remove()
            out_path = os.path.join(
                args.output, os.path.splitext(name)[0] + "_gradcam.png")
            overlay_heatmap(image_pil, cam).save(out_path)

    print(f"[DONE] 结果与热力图已保存到 {args.output}")


if __name__ == "__main__":
    main()
