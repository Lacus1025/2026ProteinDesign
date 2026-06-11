# 2026ProteinDesign — GFP 荧光亮度预测与进化优化

基于 ESM 蛋白语言模型嵌入 + 残差网络回归预测 GFP 荧光亮度及热稳定性（Tm），结合 ESM3 掩码语言模型进行多目标进化优化，设计高亮度、高热稳定性的 GFP 蛋白变体。合成生物学竞赛项目。

## 整体流程

```
                    阶段一: 监督学习 (训练亮度预测器)
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│ GFP_data.xlsx│ ──> │ ESMC-6B 嵌入    │ ──> │ ResBlock Regressor│ ──> model_final.pth
│ (14w+ 突变体) │     │ + 分片导出       │     │ (MSE + log1p)    │
└──────────────┘     └─────────────────┘     └──────────────────┘

                    阶段二: 进化优化 (多目标: 亮度 + ΔTm)
┌──────────────┐     ┌────────────┐     ┌──────────────────────┐     ┌──────────────┐
│ 种子序列      │ ──> │ ESM3 生成  │ ──> │ 亮度 + Tm 双评估      │ ──> │ 综合评分选择  │
│              │<─── │ (1% 掩码)  │     │ → 轮次内全局 Min-Max 归一化│     │ (亮度×ΔTm)   │
└──────────────┘     └────────────┘     │ → composite_score    │     └──────────────┘
      ↑                                  └──────────────────────┘            │
      └─────────────────── 迭代 10 轮 ───────────────────────────────────────┘
```

## 环境配置

```bash
conda create -n esm python=3.12
conda activate esm
pip install -r requirements.txt
```

## 依赖项

| 包 | 用途 |
|---|---|
| torch | 深度学习框架 |
| numpy | 数值计算 |
| scikit-learn | 数据集划分与 R² 评估 |
| transformers | HuggingFace 模型加载 (ESMC-6B) |
| pandas / openpyxl | 读取 Excel 数据 |
| tqdm | 进度条 |

## 数据准备

将原始 `GFP_data.xlsx` 转换为训练数据集（含 ESM 嵌入）：

```bash
python utils/export_dataset_json.py
```

> **14 万条数据采用分片导出**，避免大数据量卡死。每 1000 条记录为一个分片，ESM 嵌入在存储前做 5 段平均池化（`250×2560 → 5×50×2560 → mean(axis=1) → 12800`），大幅降低存储占用（~341 GB → ~5.5 GB）。

输出文件：
- `gfp_dataset_shard_000.json` … `_shard_NNN.json` — 各分片序列与亮度标注
- `gfp_dataset_shard_000_embeddings.npy` … `_shard_NNN_embeddings.npy` — 各分片池化嵌入 `[1000, 12800]`
- `gfp_dataset_shards.json` — 分片清单（总记录数、分片数、路径映射）

## 训练

```bash
python model/train.py
```

训练配置（`model/train.py` 内）：
- 模型: `BrightnessRegressor` — 输入投影 + 5 层残差块（ResidualBlock）+ 输出投影
- 输入: 5段平均池化后的 12800 维 ESM 嵌入（`[5, 50, 2560] → mean(axis=1) → [12800]`）
- 损失: MSE（对 `log1p` 变换后的亮度值）
- 优化器: Adam, lr=5e-4, ReduceLROnPlateau 调度
- 数据划分: 80/10/10 (train/val/test)，seed=114514
- 早停: patience=40，梯度裁剪 max_norm=1.0
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

### 评分机制

每轮对所有父代生成的序列同时评估亮度和热稳定性（Tm）：

1. **亮度预测**: 使用 `EVAL` 类加载 `model_final.pth`
2. **Tm 预测**: 使用 `TM_PREDICTOR` 类加载 ProCeSa GNN 模型
3. **综合评分**: 轮次内对所有序列的亮度和 ΔTm（Tm - Tm_wt）分别做全局 Min-Max 归一化到 [0, 1]，相乘得 `composite_score`
4. **选择**: 按 `composite_score` 降序保留 Top K

### 配置（`main.py` 内 `CONFIG` 字典）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `batch_size_per_parent` | 200 | 每轮每个父代生成的子序列数 |
| `top_k_ratio` | 0.1 | 每轮保留比例 |
| `max_parents` | 20 | 每轮最大父代数 |
| `drop_rate` | 0.01 | 子序列中随机掩码的残基比例 |
| `temperature` | 1.5 | ESM3 生成温度（实际加 ±20% 随机扰动） |
| `num_rounds` | 10 | 迭代轮数 |
| `model_path` | `model_final.pth` | 亮度预测模型路径 |
| `tm_config` | `configs/S_esmc/model3.py` | ProCeSa 模型配置（相对 `ProCeSa/procesa/`） |
| `tm_checkpoint` | `results/S_esmc/seed-101/model3/epoch_best.pth` | ProCeSa 模型权重 |

### 输出

- `pipeline_results_{substrate}.json` — 完整结果（含所有轮次的序列、亮度、Tm、ΔTm、综合评分）
- `pipeline_results_{substrate}_round{N}.json` — 每轮快照
- 终端打印每轮 Top 序列的综合评分、亮度及 ΔTm

## 一键运行

```bash
bash run.sh
```

依次执行：环境激活 → 依赖安装 → 数据集导出 → 训练（日志保存到 `logs/` 目录）。

## 项目结构

```
.
├── main.py                              # 进化优化主管线（亮度+Tm 多目标）
├── run.sh                               # 一键运行脚本
├── requirements.txt                     # Python 依赖
├── GFP_data.xlsx                        # 原始突变体数据
├── model/
│   ├── BrightnessRegressor.py           # ResBlock 回归模型定义
│   ├── train.py                         # 训练脚本
│   └── eval.py                          # 亮度推理封装类 (EVAL)
├── esm_utils/
│   ├── esmc_embedding.py                # ESMC-6B 序列嵌入
│   ├── esm3_generate.py                 # ESM3 掩码序列生成
│   ├── tm_predictor.py                  # ProCeSa Tm 预测封装类 (TM_PREDICTOR)
│   ├── esmfold2_generate.py             # ESMFold2 结构预测（独立工具）
│   └── wt_extract.py                    # 野生型嵌入提取
├── utils/
│   ├── export_dataset_json.py           # Excel → 分片 JSON + 嵌入
│   ├── convert_gfp_data.py              # 突变解析与序列构建
│   ├── fpbase_crawler.py                # FPbase 数据库爬取
│   └── fpbase_to_fasta.py               # JSON → FASTA 转换
├── ProCeSa/                             # ProCeSa 热稳定性预测模型（外部依赖）
└── checkpoints/                         # 训练检查点
```

## 技术细节

### 模型架构

```
Input: [batch, 12800] (5段平均池化后的 ESM 嵌入)
  → Linear(12800→4096) + BN + GELU + Dropout(0.3)
  → ResidualBlock(4096→4096, dropout=0.2)
  → ResidualBlock(4096→512,  dropout=0.2)
  → ResidualBlock(512→256,   dropout=0.1)
  → ResidualBlock(256→128,   dropout=0.1)
  → ResidualBlock(128→64)
  → Linear(64→1) → squeeze
Output: [batch] (预测亮度, log1p 空间)

每个 ResidualBlock = Linear → BatchNorm1d → GELU → Dropout → + shortcut(投影或恒等)
```

### 数据说明

- 训练数据来自 5 种野生型 GFP（sfGFP, avGFP, amacGFP, cgreGFP, ppluGFP）及其突变体，共 14w+ 条
- 亮度值经过 `log1p` 变换后作为回归目标，取值约 1.3 ~ 3.8
- 使用 ESMC-6B 模型提取每残基 2560 维嵌入 `[250, 2560]`，reshape 为 `[5, 50, 2560]` 后沿时间轴平均池化为 `[12800]` 单向量
- 序列过滤条件：长度 225~250（超出范围的直接丢弃）

### 热稳定性预测

Tm 预测使用 ProCeSa 图神经网络，部署在 `esm_utils/tm_predictor.py`：

- **编码**: ESMC-600M 将序列编码为 per-residue embedding + 自注意力矩阵
- **建图**: 以残基为节点（embedding 为特征）、注意力为边（归一化后为边特征）构建 DGL 图
- **推理**: ProCeSa GNN 前向传播输出 Tm 预测值
- **Tm_wt**: 野生型序列的 Tm 作为 ΔTm 计算的基线参考
