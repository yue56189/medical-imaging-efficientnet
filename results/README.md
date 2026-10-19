# 实验结果存档

本次实验：**EfficientNet-b0 迁移学习 · Brain Tumor MRI 四分类 · CPU 真实训练**

训练时间：2026-10-19 ｜ 代码版本：见 git log ｜ 训练命令见 `../docs/技术文档.md` 8.1 节

## 文件清单

| 文件 | 内容 |
|---|---|
| `best.pt` | 最佳模型权重（第 7 epoch，按 val macro-F1 选择，15.6MB，含 class_to_idx 与训练配置） |
| `train_metrics.json` | 训练全过程逐 epoch 指标（loss/acc/F1） |
| `test_metrics.json` | test 集最终指标（accuracy/macro-AUC/每类 AUC/分类报告/混淆矩阵原始数据） |
| `training_curves.png` | 训练/验证的 loss 与 accuracy 曲线（含两阶段分界线） |
| `confusion_matrix.png` | test 集混淆矩阵 |
| `roc_curves.png` | test 集 one-vs-rest ROC 曲线 |
| `gradcam_examples/` | Grad-CAM 示例（一张判对 + 一张高置信度错判） |

## 核心数字

| 指标 | 值 |
|---|---|
| Test accuracy | **0.9144**（1600 张） |
| Test macro ROC-AUC | **0.9821** |
| Test macro F1 | 0.9123 |
| 最佳 val macro-F1 | 0.9533 |

| 类别 | precision | recall | F1 | OvR-AUC |
|---|---|---|---|---|
| glioma | 0.987 | **0.770** | 0.865 | 0.950 |
| meningioma | 0.907 | 0.903 | 0.905 | 0.986 |
| notumor | 0.896 | 0.990 | 0.941 | 0.994 |
| pituitary | 0.888 | 0.995 | 0.939 | 0.999 |

## 一句话结论

两阶段迁移学习 work（解冻后首个 epoch val acc 61.7%→88.8%），CPU 40 分钟拿到
test 91.4% / AUC 0.982；短板是 glioma（与 meningioma 形态相近，92/400 被错分），
这是该数据集的公认难点，非代码缺陷。局限与后续方向见技术文档 8.6 节。

## 复现

```bash
python src/train.py --data-root <数据集目录> --model efficientnet_b0 \
    --batch-size 64 --freeze-epochs 2 --epochs 8 --output runs/brain_mri_b0_v1
python src/evaluate.py --checkpoint runs/brain_mri_b0_v1/best.pt --data-root <数据集目录>
```

> 注：本目录的 best.pt 对应上述超参；换数据集/超参需重新训练。
> 数据集本体约 330MB，未入库，下载方式见 `scripts/download_data.py`。
