# EfficientNet 医疗影像分类（迁移学习）

用 **EfficientNet 预训练模型 + 两阶段迁移学习** 做医疗影像疾病分类。

本项目复刻并**修正**了一个 GitHub 上的引流仓库
（`QQ767172261/Deep-Learning_Using-Convolutional-Neural-Networks-CNNs-to-select-EfficientNet-pretrained-models-for-`，
该仓库只有 README 广告和几段不可运行的示例代码，数据集需付费）。
本项目用**公开数据集**重写了完整可跑的工程，并已用真实数据完成训练（见「实验结果」）。

## 功能

- 基于 [timm](https://github.com/huggingface/pytorch-image-models) 的 EfficientNet b0~b3（可换任意 timm 模型）
- 两阶段训练：冻结 backbone 训分类头 → 解冻全网络小学习率微调
- 类别不平衡自动加权、余弦退火、早停、混合精度（GPU）
- 评估：分类报告 + 混淆矩阵 + ROC 曲线（one-vs-rest AUC）
- Grad-CAM 热力图可视化（可解释性）
- 公开数据集下载助手（Kaggle / HuggingFace 镜像 / figshare 直链，带 MD5 校验）
- 数据分层划分脚本（train/val/test 按比例、固定种子、可复现）

## 快速开始

```bash
# 1. 安装依赖（Python 3.10+，建议有 NVIDIA GPU；CPU 也能跑，只是慢）
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 境内网络注意：timm 的 ImageNet 预训练权重默认从 huggingface.co 下载，
# 直连不通时设置镜像（代码也会自动检测并重启进程切换，但手动设置更稳）：
#   Windows:  set HF_ENDPOINT=https://hf-mirror.com
#   Linux:    export HF_ENDPOINT=https://hf-mirror.com

# 2. 下载公开数据集（默认 Kaggle；免登录走 HF 镜像或 figshare 直链）
python scripts/download_data.py list                    # 看有哪些数据集
python scripts/download_data.py download mri_brain_hf --dest D:/datasets/mri_brain_raw

# 3. 组织数据目录（从 Training 里切 15% 做 val，Testing 留作 test）
#    本仓库自带的整理逻辑见 docs/技术文档.md 第 6 节

# 4. 训练（两阶段：3 epochs 冻结 + 20 epochs 微调）
python src/train.py --data-root D:/datasets/mri_brain --model efficientnet_b0 \
    --epochs 20 --freeze-epochs 3 --amp

# 5. 评估（混淆矩阵 / ROC / 分类报告 -> runs/xxx/eval/）
python src/evaluate.py --checkpoint runs/xxx/best.pt --data-root D:/datasets/mri_brain

# 6. 单图推理 + Grad-CAM
python src/predict.py --checkpoint runs/xxx/best.pt --image some_mri.jpg
```

## 支持的数据集（链接均已验证）

| 模态 | 数据集 | 来源 | 规模 | 获取方式 |
|---|---|---|---|---|
| MRI | Brain Tumor MRI（masoudnickparvar） | Kaggle / HF 镜像 | 7,200 张，4 类均衡 | `download mri_brain_hf`（免登录） |
| MRI | Brain Tumor（Cheng 2017） | figshare 直链 | 3,064 张，4 类 | `download mri_brain_figshare`（MD5 校验） |
| X光 | Chest X-Ray Pneumonia | Kaggle | 5,856 张，2 类 | `download xray_pneumonia`（需 kaggle.json） |
| X光 | Pneumonia 重划分平衡版 | Kaggle | 5,856 张 | `download xray_pneumonia_balanced` |
| CT | Chest CT-Scan | Kaggle | 1,000 张，4 类 | `download ct_lung` |
| MRI | Brain Tumor 4 类（sartajbhuvaji） | Kaggle | ~3,264 张 | `download mri_brain4` |
| X光 | NIH ChestX-ray14 | Kaggle / NIH 官网 | 112,120 张，14 标签（多标签） | `download nih_chestxray14` |

> 注：Kaggle 下载需要 API 凭证（`~/.kaggle/kaggle.json`）；HF 镜像和 figshare 直链不需要。
> NIH ChestX-ray14 是多标签任务，直接按多分类训练会丢失信息，需改造损失函数（见技术文档）。

## 项目结构

```
medical-imaging-efficientnet/
├── src/
│   ├── config.py      # 命令行参数
│   ├── dataset.py     # 数据加载（修正版，见技术文档）
│   ├── models.py      # timm 模型构建 / 冻结 / 解冻
│   ├── utils.py       # 种子、曲线、checkpoint
│   ├── train.py       # 两阶段训练主脚本
│   ├── evaluate.py    # test 集评估（报告/混淆矩阵/ROC）
│   ├── gradcam.py     # Grad-CAM
│   └── predict.py     # 推理 + 热力图
├── scripts/
│   ├── download_data.py   # 公开数据集下载助手
│   └── split_dataset.py   # train/val/test 分层划分
├── docs/
│   └── 技术文档.md        # 完整技术文档
└── results/           # 本次真实训练的实验产物（模型、指标、图）
```

## 实验结果

已用 Brain Tumor MRI 数据集（7,200 张，4 类）在本机 **CPU 真实训练**（EfficientNet-b0，
两阶段共 10 epochs，40 分钟）：

| 指标 | 值 |
|---|---|
| Test accuracy | **91.4%** |
| Test macro ROC-AUC | **0.9821** |
| 最佳 val macro-F1 | 0.9533（第 7 epoch） |
| 分类报告 | glioma F1 0.87（recall 0.77，短板）/ meningioma 0.90 / notumor 0.94 / pituitary 0.94 |

产物（模型 best.pt 15.6MB、训练曲线、混淆矩阵、ROC、Grad-CAM 示例）见
[`results/`](results/) 目录，完整分析见
[技术文档 - 实验结果](docs/技术文档.md#8-实验结果)。

主要错分：glioma → meningioma（36 张）、glioma → notumor（33 张）。
val(95.3%) 高于 test(91.4%) 属正常——val 参与了模型选择，test 才是无偏泛化估计。

## 技术文档

[`docs/技术文档.md`](docs/技术文档.md)：原仓库分析、数据集选型、训练方法学、
评估方案、已知坑（Kaggle val 集陷阱、患者级数据泄漏、医学增强禁忌）、复现步骤。

## 数据来源与引用

- Brain Tumor MRI Dataset (masoudnickparvar): Kaggle `masoudnickparvar/brain-tumor-mri-dataset`
- Cheng, Jun (2017). brain tumor dataset. figshare. https://doi.org/10.6084/m9.figshare.1512427.v8 (CC BY 4.0)
- Kermany et al., chest-x-ray-pneumonia, Kaggle `paultimothymooney/chest-xray-pneumonia`
- Wang et al., ChestX-ray14, NIH Clinical Center

## License

MIT（代码部分）。数据集版权归各自作者，使用时遵守对应许可并引用原文。
