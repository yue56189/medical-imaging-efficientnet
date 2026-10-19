"""Grad-CAM 可视化：显示模型分类时关注的图像区域。

实现方式：在最后一个卷积阶段（timm EfficientNet 为 conv_head）注册
forward/backward hook，对目标类别分数反向传播，得到
CAM = relu(sum_k  grad_k 对特征图 k 的空间平均 * 特征图_k)。

用法（一般不需要单独运行，predict.py 已内置调用）：
    python src/gradcam.py --checkpoint runs/xxx/best.pt --image path/to/x.png
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from matplotlib import colormaps

from models import build_model
from utils import get_device
from train import build_transforms, IMAGENET_MEAN, IMAGENET_STD


class GradCAM:
    def __init__(self, model):
        self.model = model
        self._features = None
        self._grads = None
        self._hooks = []

        # 目标层：优先 conv_head（timm EfficientNet 最后一个卷积阶段），
        # 否则退到主干最后一个 block
        target = getattr(model, "conv_head", None)
        if target is None and hasattr(model, "blocks"):
            target = model.blocks[-1]
        if target is None:
            raise RuntimeError("未找到可用于 Grad-CAM 的目标层（conv_head / blocks）")

        self._hooks.append(target.register_forward_hook(self._forward_hook))
        self._hooks.append(target.register_full_backward_hook(self._backward_hook))

    def _forward_hook(self, module, inputs, output):
        self._features = output

    def _backward_hook(self, module, grad_input, grad_output):
        self._grads = grad_output[0]

    def generate(self, x, class_idx=None):
        """x: (1,C,H,W) 已归一化的张量。返回 (cam: HxW float 0~1, class_idx)。"""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        logits[0, class_idx].backward()

        feats = self._features[0]   # (C,H,W)
        grads = self._grads[0]      # (C,H,W)
        weights = grads.mean(dim=(1, 2))               # GAP 得到通道权重
        cam = torch.relu((weights[:, None, None] * feats).sum(dim=0))
        cam = cam.detach().cpu().numpy().astype(np.float32)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam, class_idx

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


def overlay_heatmap(image_pil, cam, alpha=0.45):
    """把 CAM 上采样到原图尺寸，用 jet 伪彩色半透明叠加。返回 PIL.Image。"""
    cam_u8 = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        image_pil.size, Image.BILINEAR)
    heatmap = np.array(colormaps["jet"](np.array(cam_u8)))[:, :, :3]  # (H,W,3) 0~1
    base = np.array(image_pil.convert("RGB")).astype(np.float32) / 255.0
    blended = (1 - alpha) * base + alpha * heatmap
    return Image.fromarray((blended * 255).astype(np.uint8))


def load_inference_model(checkpoint_path, device=None):
    """从 checkpoint 还原模型，返回 (model, class_to_idx, img_size, device)。"""
    device = device or get_device()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_to_idx = ckpt["class_to_idx"]
    model = build_model(ckpt["model_name"], num_classes=len(class_to_idx),
                        pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, class_to_idx, ckpt["img_size"], device


def main():
    import argparse
    p = argparse.ArgumentParser(description="Grad-CAM 单图可视化")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--output", default="outputs/gradcam")
    args = p.parse_args()

    device = get_device()
    model, class_to_idx, img_size, device = load_inference_model(args.checkpoint, device)
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    _, val_tf = build_transforms(img_size)
    image_pil = Image.open(args.image).convert("RGB")
    x = val_tf(image_pil).unsqueeze(0).to(device)

    cam_tool = GradCAM(model)
    with torch.enable_grad():
        cam, cls_idx = cam_tool.generate(x)
    cam_tool.remove()

    probs = torch.softmax(model(x), dim=1)[0]
    print(f"预测: {idx_to_class[cls_idx]}  (置信度 {probs[cls_idx].item():.4f})")

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, os.path.splitext(os.path.basename(args.image))[0] + "_gradcam.png")
    overlay_heatmap(image_pil, cam).save(out_path)
    print(f"[DONE] 已保存 {out_path}")


if __name__ == "__main__":
    main()
