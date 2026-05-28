# 2026ProteinDesign — GFP 荧光亮度预测与进化优化

基于 ESM 蛋白语言模型嵌入 + 卷积神经网络回归预测 GFP 荧光亮度，并结合 ESM3 掩码语言模型进行进化优化，设计更高亮度的 GFP 蛋白变体。合成生物学竞赛项目。

## 整体流程

```
                    阶段一: 监督学习 (训练亮度预测器)
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│ GFP_data.xlsx│ ──> │ ESM-3B 嵌入     │ ──> │ Conv1D Regressor │ ──> model_final.pth
│ (14w+ 突变体) │     │ + 数据集导出     │     │ (MSE + Δlog1p)   │
└──────────────┘     └─────────────────┘     └──────────────────┘

                    阶段二: 进化优化 (生成更亮 GFP)
┌──────────────┐     ┌────────────┐     ┌──────────────┐     ┌──────────────┐
│ 种子序列      │ ──> │ ESM3 生成  │ ──> │ 预测器评分     │ ──> │ Top K 选择   │
│              │<─── │ (90%掩码)  │     │ (model.pth)  │     │ (10%)        │
└──────────────┘     └────────────┘     └──────────────┘     └──────────────┘
      ↑                                                              │
      └──────────────── 迭代 10 轮 ──────────────────────────────────┘
```

## 环境配置

```bash
# 创建并激活 conda 环境
conda create -n esm python=3.12
conda activate esm

# 安装依赖
pip install -r requirements.txt
```

## 依赖项

| 包 | 用途 |
|---|---|
| torch | 深度学习框架 |
| numpy | 数值计算 |
| scikit-learn | 数据集划分与 R² 评估 |
| esm | ESM 蛋白语言模型（OpenFold/ESMC） |
| transformers | HuggingFace 模型加载 |
| pandas / openpyxl | 读取 Excel 数据 |
| tqdm | 进度条 |

## 数据准备

将原始 `GFP_data.xlsx` 转换为训练数据集（含 ESM 嵌入）：

```bash
python utils/export_dataset_json.py
```

输出文件：
- `gfp_dataset.json` — 序列与亮度标注
- `gfp_dataset_embeddings.npy` — 预计算的 ESM 嵌入 `[N, 250, 2560]`

## 训练

```bash
python model/train.py
```

训练配置（`model/train.py` 内）：
- 模型: `BrightnessRegressor` — 3 层 Conv1D + 4 层 MLP
- 输入: 每残基 2560 维 ESM 嵌入 × 250 序列长度
- 损失: MSE（对 log1p 变换后的亮度值）
- 优化器: Adam, lr=1e-5
- 数据划分: 80/10/10 (train/val/test)，seed=114514
- 早停: patience=20，梯度裁剪 max_norm=1.0
- 输出: `model_final.pth`, `evaluation_results.json`, `training_logs.json`

## 评估

对单条序列进行亮度预测：

```bash
python model/eval.py
```

输出野生型 GFP 的预测亮度值。

在代码中使用：

```python
from model.eval import EVAL

evaluator = EVAL("model_final.pth")
brightness = evaluator.predict("MSKGEELFTGVV...")
print(f"预测亮度: {brightness:.4f}")
```

## 进化优化管线

运行完整的生成-评分-选择迭代管线：

```bash
python main.py
```

默认配置（`main.py` 内 `CONFIG` 字典）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| 初始序列 | `______AAB______` | 种子掩码序列 |
| 每父代生成数 | 200 | 每轮每个父代生成的子序列数 |
| Top K 比例 | 0.1 | 每轮保留比例 |
| 最大父代数 | 20 | 每轮保留上限 |
| 掩码率 | 0.9 | 子序列中随机掩码的残基比例 |
| 温度 | 1.0 | ESM3 生成温度（实际加 ±20% 随机扰动） |
| 生成轮数 | 10 | 迭代轮数 |
| 序列长度 | 225-250 | 过滤有效序列的长度范围 |

每轮结果保存到 `pipeline_results.json`，最终输出最佳序列与亮度。

## 一键运行

```bash
bash run.sh
```

依次执行：环境激活 → 依赖安装 → 数据集导出 → 训练（日志保存到 `logs/` 目录）。

## 项目结构

```
.
├── main.py                              # 进化优化主管线
├── run.sh                               # 一键运行脚本
├── requirements.txt                     # Python 依赖
├── GFP_data.xlsx                        # 原始突变体数据
├── model/
│   ├── BrightnessRegressor.py           # Conv1D 回归模型定义
│   ├── train.py                         # 训练脚本
│   └── eval.py                          # 推理封装类
├── esm_utils/
│   ├── esmc_embedding.py                # ESMC-6B 序列嵌入
│   ├── esm3_generate.py                 # ESM3 掩码序列生成
│   ├── esmfold2_generate.py             # ESMFold2 结构预测（独立工具）
│   └── wt_extract.py                    # 野生型嵌入提取
├── utils/
│   ├── export_dataset_json.py           # Excel → JSON + 嵌入
│   └── convert_gfp_data.py              # 突变解析与序列构建
└── checkpoints/                         # 训练检查点
```

## 技术细节

### 模型架构

```
Input: [batch, 250, 2560]
  → permute → [batch, 2560, 250]
  → Conv1d(2560→512, k=5, s=2) → BN → GELU → Dropout(0.3)
  → Conv1d(512→256,  k=5, s=2) → BN → GELU → Dropout(0.2)
  → Conv1d(256→128,  k=3, s=2) → BN → GELU → Dropout(0.2)
  → Flatten → [batch, 4096]
  → Linear(4096→1024→512→256→1)
Output: [batch] (预测亮度, log1p 空间)
```

### 数据说明

- 训练数据来自 5 种野生型 GFP（sfGFP, avGFP, amacGFP, cgreGFP, ppluGFP）及其突变体，共 14w+ 条
- 亮度值经过 `log1p` 变换后作为回归目标，取值约 1.3 ~ 3.8
- 序列统一补齐/截断至长度 250
- 使用 ESMC-6B 模型提取每残基 2560 维嵌入
