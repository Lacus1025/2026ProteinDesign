# 2026ProteinDesign — GFP 荧光亮度预测与进化优化

基于 ESM 蛋白语言模型嵌入 + 残差网络回归预测 GFP 荧光亮度及热稳定性（Tm），结合 ESM3 掩码语言模型 + S4PRED 二级结构引导进行多目标进化优化，设计高亮度、高热稳定性的 GFP 蛋白变体。合成生物学竞赛项目。

## 整体算法管线

本项目由两个阶段构成：**第一阶段**训练亮度预测回归器，**第二阶段**利用该回归器结合 Tm 预测器进行多目标进化优化。

### 阶段一：监督学习 — 训练亮度预测器

```
┌──────────────────┐
│ 1. 数据准备                                    │
│                                                │
│ GFP_data.xlsx                                  │
│ (14w+ 突变体, 5 种野生型 GFP)                    │
│   │                                            │
│   ▼ convert_gfp_data.py                        │
│ 解析突变字符串 (S65G:Y66F), 重组全长序列           │
│ 序列过滤: 225 ≤ len ≤ 250 aa                   │
│   │                                            │
│   ▼ ESMC-6B (600M 参数 PLM)                     │
│ Per-residue embedding [250, 2560]               │
│   │                                            │
│   ▼ 5段平均池化                                  │
│ reshape(5, 50, 2560) → mean(axis=1) → [12800]   │
│   │                                            │
│   ▼ 分片导出 (每1000条一shard)                    │
│ gfp_dataset_shard_*.json + *_embeddings.npy      │
│ gfp_dataset_shards.json (分片清单)               │
│                                                │
│ 存储优化: ~341 GB → ~5.5 GB                     │
└──────────────────┘
         │
         ▼
┌──────────────────┐
│ 2. 模型训练                                    │
│                                                │
│ 数据划分: 80/10/10 (train/val/test)             │
│                                                │
│ BrightnessRegressor                            │
│   Input: [batch, 12800]                        │
│   → Linear(12800→4096) + BN + GELU + Dropout(0.3) │
│   → ResBlock(4096→4096, d=0.2)                 │
│   → ResBlock(4096→512,  d=0.2)                 │
│   → ResBlock(512→256,   d=0.1)                 │
│   → ResBlock(256→128,   d=0.1)                 │
│   → ResBlock(128→64)                           │
│   → Linear(64→1)                               │
│   Output: [batch] (log1p 空间亮度)              │
│                                                │
│ 损失函数: MSE(log1p(brightness), pred)          │
│ 优化器: Adam, lr=5e-4                            │
│ 调度器: ReduceLROnPlateau (factor=0.5, patience=15) │
│ 梯度裁剪: max_norm=1.0                          │
│ 早停: patience=40 epochs                        │
│ 批次大小: 512                                   │
│   │                                            │
│   ▼                                            │
│ model_final.pth + training_logs.json            │
│ evaluation_results.json                        │
└──────────────────┘
```

### 阶段二：进化优化 — 多目标（亮度 + 热稳定性）

```
┌─────────────────────────────────────────────────────┐
│ sfGFP 种子序列 (238 aa, S147P 修正)                   │
│   │                                                  │
│   ▼ 每轮迭代 (共10轮)                                │
│                                                      │
│ ┌──────────────┐    ┌──────────────────────┐         │
│ │ S4PRED 预测   │───>│ 掩码策略              │         │
│ │ 二级结构 (H/E/C)│   │ · 锁定关键功能残基     │         │
│ └──────────────┘    │   (sfGFP: 21个位置固定)  │         │
│                     │ · 排除非coil区域        │         │
│                     │   (保护H/E结构残基)      │         │
│                     │ · 5%随机掩码为 _         │         │
│                     └──────────┬───────────┘         │
│                                │                      │
│                                ▼                      │
│                     ┌──────────────────────┐         │
│                     │ ESM3-open 掩码生成     │         │
│                     │ · num_steps=8          │         │
│                     │ · temperature=1.5      │         │
│                     │   (±20%随机扰动)        │         │
│                     │ · 200 seqs/parent      │         │
│                     └──────────┬───────────┘         │
│                                │                      │
│                                ▼                      │
│                     ┌──────────────────────┐         │
│                     │ 序列质量控制           │         │
│                     │ · 去重                  │         │
│                     │ · 排除无效氨基酸(BJOUXZ) │         │
│                     │ · 排除列表过滤(13w+条)   │         │
│                     │ · 长度检查(225-250)     │         │
│                     └──────────┬───────────┘         │
│                                │                      │
│                                ▼                      │
│                     ┌──────────────────────┐         │
│                     │ 双维评估               │         │
│                     │                        │         │
│                     │  A. 亮度 (EVAL)         │         │
│                     │  ESMC-6B → 5段池化     │         │
│                     │  → BrightnessRegressor │         │
│                     │  → expm1 恢复原始亮度   │         │
│                     │                        │         │
│                     │  B. Tm (TM_PREDICTOR)  │         │
│                     │  ESMC-600M per-residue │         │
│                     │  + self-attention 建图  │         │
│                     │  → ProCeSa GNN → Tm(°C) │         │
│                     └──────────┬───────────┘         │
│                                │                      │
│                                ▼                      │
│                     ┌──────────────────────┐         │
│                     │ 综合评分 (轮内全局归一化) │         │
│                     │                        │         │
│                     │ b_norm ∈ [0,1]         │         │
│                     │ dtm_norm ∈ [0,1]       │         │
│                     │ composite = b²_norm × dtm_norm │
│                     └──────────┬───────────┘         │
│                                │                      │
│                                ▼                      │
│                     ┌──────────────────────┐         │
│                     │ Top-K 选择             │         │
│                     │ Top 10% → 最多20条     │         │
│                     │ → 下一轮父代            │         │
│                     └──────────────────────┘         │
│                                                      │
│ 输出: pipeline_results_sfGFP.json                     │
│       pipeline_results_sfGFP_round{N}.json            │
└─────────────────────────────────────────────────────┘
```

---

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
| transformers | HuggingFace 模型加载 (ESMC-6B, ESMC-600M) |
| pandas / openpyxl | 读取 Excel 数据 |
| tqdm | 进度条 |
| dgl | 图神经网络 (ProCeSa Tm 预测) |
| esm | ESM3 SDK (序列生成) |
| huggingface_hub | HuggingFace 模型认证与下载 |

> **外部模型依赖**（需从 GitHub 克隆到项目同级目录）：
> - [ProCeSa](https://github.com/Lacus1025/ProCeSa) — 热稳定性预测 GNN，放置在 `../ProCeSa/`（即与本项目同级的 ProCeSa 目录）
> - [s4pred](https://github.com/psipred/s4pred) — 二级结构预测，放置在 `../s4pred/`

## 数据准备

将原始 `GFP_data.xlsx` 转换为训练数据集（含 ESM 嵌入）：

```bash
python utils/export_dataset_json.py
```

**处理流程：**

1. **突变解析** — `convert_gfp_data.py` 解析突变字符串（如 `S65G:Y66F` 将丝氨酸65突变为甘氨酸、酪氨酸66突变为苯丙氨酸），在对应野生型序列上应用所有突变得到完整序列；支持多突变（`:` 分隔）和插入（`*0M`）。
2. **序列过滤** — 仅保留长度在 225~250 aa 之间的序列。
3. **嵌入提取** — 使用 ESMC-6B 模型（Biohub/ESMC-6B，600M 参数）提取每条序列的 per-residue 嵌入，输出 `[250, 2560]`（不足250补零，超出截断）。
4. **5段平均池化** — 将 `[250, 2560]` reshape 为 `[5, 50, 2560]`，沿残基轴（axis=1）取均值，得到 `[12800]` 单一向量。大幅降低存储占用（~341 GB → ~5.5 GB）。
5. **分片导出** — 每 1000 条记录为一个分片，分别写入 JSON（序列与标注）和 .npy（池化嵌入）。

**输出文件：**
- `gfp_dataset_shard_{NNN}.json` — 分片序列、GFP类型、亮度标注
- `gfp_dataset_shard_{NNN}_embeddings.npy` — 分片池化嵌入 `[N, 12800]`
- `gfp_dataset_shards.json` — 分片清单（总记录数、分片数、各分片路径与记录数）

## 训练

```bash
python model/train.py
```

### 训练配置

| 参数 | 值 | 说明 |
|---|---|---|
| `embed_dim` | 12800 | 输入维度（5段 × 2560） |
| `learning_rate` | 5e-4 | Adam 学习率 |
| `num_epochs` | 500 | 最大训练轮数 |
| `checkpoint_freq` | 30 | 每 N 轮保存检查点 |
| `early_stopping_patience` | 40 | 验证损失不再下降则提前终止 |
| `batch_size` | 512 | 训练/验证/测试批次大小 |
| `seed` | 114514 | 全局随机种子（数据划分、参数初始化） |

**训练细节：**
- 损失函数：MSE，目标值为 `log1p(brightness)`（将亮度对数压缩至约 1.3~3.8 范围）
- 优化器：Adam (lr=5e-4)，配合 ReduceLROnPlateau（因子0.5，patience=15）
- 梯度裁剪：`max_norm=1.0`，每批次应用
- 数据加载：GPU 上 4 workers + pin_memory，CPU 上 0 workers
- 每轮评估验证集 R²（将 `expm1` 恢复后的预测亮度与真实亮度比较）

**输出文件：**
- `model_final.pth` — 最终模型权重 + 优化器状态 + 配置
- `checkpoints/best_model.pth.tar` — 验证损失最低的检查点
- `checkpoints/checkpoint_epoch_{N}.pth.tar` — 周期检查点
- `training_logs.json` — 每轮训练/验证损失、学习率历史
- `evaluation_results.json` — 测试集 MSE、R²、100 条随机样本预测详情

## 单序列亮度评估

对任意序列进行亮度预测：

```bash
python model/eval.py
```

在代码中使用：

```python
from model.eval import EVAL

evaluator = EVAL("model_final.pth")
brightness = evaluator.predict("MSKGEELFTGVV...")
print(f"预测亮度: {brightness:.4f}")  # 已还原为原始亮度值
```

`EVAL.predict()` 内部流程与训练时一致：ESMC-6B 提取嵌入 → 5段平均池化 → 模型前向传播 → `expm1` 恢复原始尺度。

## 进化优化管线

运行完整的生成-评分-选择迭代管线：

```bash
python main.py
```

### 配置参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `batch_size_per_parent` | 200 | 每轮每个父代生成的子序列数 |
| `top_k_ratio` | 0.1 | 每轮保留比例（Top 10%） |
| `max_parents` | 20 | 每轮最大父代数 |
| `drop_rate` | 0.05 | 序列中随机掩码的残基比例（5%） |
| `temperature` | 1.5 | ESM3 生成温度（实际加 ±20% 随机扰动，范围 1.2~1.8） |
| `num_rounds` | 10 | 进化迭代轮数 |
| `num_steps` | 8 | ESM3 生成步数 |
| `seed` | 42 | 进化管线随机种子 |
| `min_sequence_length` | 225 | 最小有效序列长度 |
| `max_sequence_length` | 250 | 最大有效序列长度 |
| `model_path` | `model_final.pth` | 亮度预测模型路径 |
| `tm_config` | `configs/S_esmc/model3.py` | ProCeSa 模型配置（相对 ProCeSa/procesa/） |
| `tm_checkpoint` | `results/S_esmc/seed-101/model3/epoch_best.pth` | ProCeSa 模型权重 |

### 序列生成

1. **二级结构预测** — 对每条父序列运行 S4PRED，获得每个残基的二级结构（H=螺旋, E=折叠, C=无规卷曲）。
2. **掩码策略** — 在下述位置以外随机选择 5% 残基替换为 `_`（掩码token）：
   - **锁定位置**：sfGFP 的 21 个关键功能残基（M1, R30, N39, L64, T65, Y66, G67, Q94, R96, S99, T105, F145, P147, H148, T153, A163, V171, T203, S205, V206, E222），保护发色团形成、荧光活性和折叠稳定性相关位点。
   - **非coil区域**：S4PRED 预测为 H 或 E 的残基不参与掩码，仅对无规卷曲（C）区域的残基进行掩码，避免破坏蛋白核心结构。
3. **ESM3 生成** — 使用 `esm3-open` 模型对被掩码的序列进行 8 步迭代生成，温度为 1.5 ± 20%（每批次随机扰动，增加多样性）。
4. **质量控制** — 去重 → 排除含无效氨基酸（BJOUXZ）的序列 → 排除 Exclusion_List.csv 中的 13w+ 条已知序列 → 长度检查（225~250 aa）。

### 双维评估与综合评分

对每轮生成的所有有效序列同时进行两项预测：

| 评估维度 | 工具 | 模型 | 输出 |
|---|---|---|---|
| 荧光亮度 | `EVAL` (model/eval.py) | BrightnessRegressor (ResBlock MLP) | 原始亮度值 |
| 热稳定性 | `TM_PREDICTOR` (esm_utils/tm_predictor.py) | ProCeSa GNN | 熔解温度 Tm (°C) |

**综合评分公式（轮内全局 Min-Max 归一化）：**

```
b_norm  = (brightness - b_min) / (b_max - b_min)     ∈ [0, 1]
dtm_norm = (ΔTm - ΔTm_min) / (ΔTm_max - ΔTm_min)     ∈ [0, 1]   (ΔTm = Tm - Tm_wt)
composite_score = b_norm² × dtm_norm
```

- 亮度权重为平方，优先优化荧光亮度，热稳定性作为辅助目标。
- 当轮内所有值相同时（零方差），归一化值退化为 0.5。

### 选择与迭代

- 按 `composite_score` 降序排列，保留 Top 10%（最多 20 条）作为下一轮父代。
- 迭代 10 轮后输出全局最佳序列及其评分。

### 输出

- `pipeline_results_sfGFP.json` — 完整结果（含配置、所有轮次、全局 Top-10、最佳序列）
- `pipeline_results_sfGFP_round{N}.json` — 每轮快照（父代、生成序列、Top 选择）
- 终端实时打印每轮 Top 序列的排名、综合评分、亮度、ΔTm、序列摘要

---

## 一键运行

```bash
bash run.sh
```

依次执行：conda 环境激活 → pip 依赖安装 → 数据集导出（含 ESM 嵌入）→ 训练（日志保存到 `logs/` 目录）。

---

## 项目结构

```
.
├── main.py                              # 进化优化主管线（亮度+Tm 多目标）
├── run.sh                               # 一键运行脚本
├── requirements.txt                     # Python 依赖
├── GFP_data.xlsx                        # 原始突变体数据（14w+）
├── AAseqs of 5 GFP proteins_20260511.txt # 野生型序列参考文件
├── Exclusion_List.csv                    # 排除序列列表（13w+ 条）
├── submission_template.csv              # 竞赛提交模板
├── model/
│   ├── BrightnessRegressor.py           # ResBlock 回归模型定义
│   ├── train.py                         # 训练脚本（含 GFP_Dataset、训练循环、评估）
│   └── eval.py                          # 亮度推理封装类 (EVAL)
├── esm_utils/
│   ├── esmc_embedding.py                # ESMC-6B 序列嵌入（per-residue [250,2560]）
│   ├── esm3_generate.py                 # ESM3-open 掩码序列生成（8步迭代）
│   ├── tm_predictor.py                  # ProCeSa Tm 预测封装类 (TM_PREDICTOR)
│   ├── s4pred_wrapper.py               # S4PRED 二级结构预测封装
│   ├── esmfold2_generate.py             # ESMFold2 结构预测（独立工具）
│   └── wt_extract.py                    # 野生型嵌入提取
├── utils/
│   ├── export_dataset_json.py           # Excel → 分片 JSON + 池化嵌入
│   ├── convert_gfp_data.py              # 突变字符串解析与全长序列构建
│   ├── fpbase_crawler.py                # FPbase 荧光蛋白数据库爬取
│   ├── fpbase_to_fasta.py               # FPbase JSON → FASTA 转换
│   ├── analyze_ss2.py                   # PSIPRED .ss2 二级结构文件解析
│   └── fetch_fireprotdb.py             # FireProtDB Tm 数据抓取
└── checkpoints/                         # 训练检查点
```

> **外部依赖**（需放置在项目同级目录）：
> - `../ProCeSa/procesa/` — ProCeSa 热稳定性预测模型（含 configs、results、weights）
> - `../s4pred/` — S4PRED 二级结构预测模型（含 weights/）
```

---

## 技术细节

### 模型架构：BrightnessRegressor

```
Input: [batch, 12800]                    (5段平均池化后的 ESM 嵌入)
  → Linear(12800→4096) + BN + GELU + Dropout(0.3)
  → ResidualBlock(4096→4096, dropout=0.2)
  → ResidualBlock(4096→512,  dropout=0.2)
  → ResidualBlock(512→256,   dropout=0.1)
  → ResidualBlock(256→128,   dropout=0.1)
  → ResidualBlock(128→64)
  → Linear(64→1) → squeeze
Output: [batch]                           (预测 log1p 空间亮度)

ResidualBlock = Linear → BatchNorm1d → GELU → Dropout → + shortcut(Identity 或 Linear投影)
```

### 嵌入处理：5段平均池化

```
ESMC-6B 输出:      [L, 2560]           (L = 序列长度, 补零/截断到250)
                      │
                      ▼
                  [250, 2560]           (pad/truncate to 250)
                      │
                      ▼ reshape
                  [5, 50, 2560]         (5段 × 每段50残基 × 2560维)
                      │
                      ▼ mean(axis=1)
                  [5, 2560]             (每段取均值)
                      │
                      ▼ flatten
                  [12800]               (最终特征向量)
```

相比直接使用 `[250, 2560]` 完整嵌入（~640K 维），池化后的 12800 维向量在保留结构信息的同时大幅降低了存储和计算开销。

### 蛋白质语言模型

| 用途 | 模型 | 参数量 | 输出 |
|---|---|---|---|
| 亮度预测嵌入 | ESMC-6B (Biohub/ESMC-6B) | 600M | Per-residue hidden state [L, 2560] |
| Tm 预测编码 | ESMC-600M (esmc_600m) | 600M | Per-residue embedding + self-attention |
| 序列生成 | ESM3-open | — | 基于 masked language modeling 的迭代生成 |

### Tm（热稳定性）预测

`TM_PREDICTOR` 将蛋白质序列转换为 DGL 图后由 ProCeSa GNN 进行推理：

1. **编码** — ESMC-600M 提取 per-residue embedding（节点特征）+ self-attention 矩阵
2. **建图** — 残基为节点（embedding 为节点特征），自注意力矩阵归一化后为非零边的边特征
3. **推理** — ProCeSa GNN 前向传播输出 ΔTm 或 Tm 预测值
4. **Tm_wt** — 野生型 sfGFP 序列的 Tm 作为基线，后续序列计算 `ΔTm = Tm - Tm_wt`

### 数据说明

- 训练数据来自 5 种野生型 GFP（sfGFP, avGFP, amacGFP, cgreGFP, ppluGFP）及其单点/多点突变体，共 14w+ 条
- 亮度值经 `log1p` 变换后作为回归目标：`label = ln(brightness + 1)`，取值约 1.3~3.8
- 序列长度过滤：225~250 aa（超出范围直接丢弃）
- 分片存储策略：每 1000 条一个分片，JSON+Numpy 配对，运行时按需 mmap 加载，降低内存占用

### 进化优化设计理念

| 策略 | 目的 |
|---|---|
| 锁定关键功能残基 | 保护发色团形成（Y66, G67）、荧光活性（T65, H148, E222）和折叠稳定性（L64, R96, F145） |
| S4PRED 二级结构引导掩码 | 仅对无规卷曲（coil）区域引入变异，保护 α-螺旋和 β-折叠等核心结构元件 |
| 温度随机扰动（±20%） | 增加每批序列多样性，平衡探索与利用 |
| 排除列表过滤 | 避免生成已知/已发表序列，保证设计新颖性 |
| 亮度² × ΔTm 评分 | 亮度为主要优化目标（平方加权），Tm 为辅助约束，防止过度牺牲热稳定性 |
| 轮内全局归一化 | 每轮独立评分，奖励该轮中相对优势的序列，避免跨轮比较的尺度不一致 |
