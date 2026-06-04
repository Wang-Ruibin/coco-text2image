# 🎨 Latent Diffusion Model — COCO 2017 Text-to-Image

> 河海大学 · 生成式人工智能 课程报告

从零实现一个潜扩散模型 (LDM)，在 MS COCO 2017 数据集上完成文本条件图像生成。  
仅使用单张 Quadro RTX 4000 (8GB)，完整跑通 **预处理 → 训练 → 推理** 全流程。

---

## 📐 整体架构

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   阶段一      │     │   阶段二      │     │   阶段三      │
│   预处理      │ ──► │   训练        │ ──► │   推理        │
└──────────────┘     └──────────────┘     └──────────────┘
  COCO 图片            Latent + Text Emb     文本 Prompt
  ──► VAE Encode       ──► U-Net + DDPM     ──► CFG 去噪
  ──► CLIP Encode      ──► MSE Loss          ──► VAE Decode
  ──► .npy 保存        ──► 权重保存           ──► 256×256 图像
```

## 🧩 核心组件

| 组件 | 模型 | 说明 |
|------|------|------|
| **VAE** | `runwayml/stable-diffusion-v1-5` (vae) | 冻结，256×256 → 32×32×4 Latent |
| **CLIP** | `openai/clip-vit-large-patch14` | 冻结，文本 → 77×768 Embedding |
| **U-Net** | 从零训练，142.2M 参数 | 条件去噪网络，支持 Cross-Attention |
| **调度器** | DDPM (训练) / DDIM (推理) | 1000 步训练，50 步推理 |

## ✨ 项目亮点

- **存储优化**：使用 `.npy` 替代 `.pt` 存储预处理数据，空间节省 **~77%**（276 GB → 63 GB）
- **轻量化设计**：U-Net 仅 142.2M 参数，8GB 显存即可训练
- **梯度累加**：等效 Batch Size 24，突破显存限制
- **CFG 支持**：Classifier-Free Guidance 引导，支持推理时调节文本-图像对齐强度
- **断点续传**：预处理与训练均支持中断后恢复

## 🚀 快速开始

### 环境依赖

```bash
pip install torch torchvision transformers diffusers tqdm matplotlib pandas pillow
```

### 数据准备

下载 [COCO 2017](https://cocodataset.org/) 数据集，按以下结构放置：

```
coco2017/
├── images/
│   ├── train2017/     # 训练集图片
│   └── val2017/       # 验证集图片
└── annotations/
    ├── captions_train2017.json
    └── captions_val2017.json
```

### 运行

直接运行 `test3.ipynb`，按顺序执行三个阶段：

1. **数据预处理** — VAE + CLIP 编码，生成 `.npy` 文件
2. **U-Net 训练** — 50 epochs，每 5 epoch 保存检查点
3. **推理生成** — 输入文本 prompt，生成 256×256 图像

## 📊 训练配置

| 参数 | 值 |
|------|-----|
| 等效 Batch Size | 24 (6 × 4 梯度累加) |
| 学习率 | 1e-4 (AdamW, Cosine + Warmup) |
| 训练轮数 | 50 |
| 总优化步数 | ~246,425 |
| 混合精度 | FP16 (autocast + GradScaler) |
| CFG 丢弃概率 | 10% |

## 🖼️ 推理示例

```python
# 交互式生成
interactive_generate(
    prompt="A dog catching a frisbee in a grassy field",
    guidance_scale=7.5,
    num_steps=50,
    seed=42
)
```

支持通过 `guidance_scale` 调节文本引导强度（推荐 5~15）。

## 📁 项目结构

```
├── test3.ipynb                  # 主 Notebook（预处理 + 训练 + 推理）
├── preproc_ldm/                 # 预处理数据 (.npy)
│   ├── train/{latents,texts}/
│   └── val/{latents,texts}/
├── checkpoints_ldm/             # U-Net 权重检查点
├── training_logs_ldm.csv        # 训练日志
└── *.png                        # 生成结果与对比图
```

## 📜 许可证

本项目采用 [MIT 许可证](LICENSE)。

---

<div align="center">

**如果觉得这个项目对你有帮助，请给一个 ⭐ Star 支持一下叭~ (鞠躬)**

*你的 star 是我继续折腾模型的最大动力！(๑•̀ㅂ•́)و✧*

</div>
