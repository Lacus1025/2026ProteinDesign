import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

SEED = 114514


class GFP_Dataset(Dataset):
    def __init__(self, file):
        self.data = json.load(open(file))
        embeddings_path = file.replace(".json", "_embeddings.npy")
        self.embeddings = np.load(embeddings_path).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.data[idx]["brightness"]
        return embedding, label


# 实例化数据集
dataset = GFP_Dataset("./gfp_dataset.json")

# 测试数据集
print("数据集大小:", len(dataset))
emb, label = dataset[0]
print("第 0 个样本 embedding shape:", emb.shape, "label:", label)

X = dataset.embeddings
y = np.array([d["brightness"] for d in dataset.data])

if len(dataset) > 10:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    print(
        f"Split data into training ({len(X_train)}) and validation ({len(X_val)}) sets."
    )
else:
    print("Dataset too small for validation split, using all data for training.")
    X_train, y_train = X, y
    X_val, y_val = None, None

print("X.shape:")
print(X.shape)
print("y.shape:")
print(y.shape)

# --- 4.2 初始化并训练随机森林模型 ---
print("\nTraining Random Forest Regressor...")
rf_model = RandomForestRegressor(
    n_estimators=100,  # 树的数量，可以调整
    random_state=SEED,
    n_jobs=-1,  # 使用所有可用的 CPU 核心
    max_depth=20,  # 限制树的深度，防止过拟合 (可调整)
    min_samples_leaf=3,  # 叶节点最小样本数 (可调整)
)

rf_model.fit(X_train, y_train)
print("Random Forest training complete.")

# --- 4.3 (可选) 评估模型性能 ---
if X_val is not None:
    y_pred_val = rf_model.predict(X_val)
    r2 = r2_score(y_val, y_pred_val)
    print(f"\nModel Performance on Validation Set:")
    print(f"  R-squared (R²): {r2:.4f}")
    # R² 接近 1 表示模型拟合得较好，接近 0 或负数表示拟合很差
else:
    # 可以在训练集上评估，但这通常会过于乐观
    y_pred_train = rf_model.predict(X_train)
    r2_train = r2_score(y_train, y_pred_train)
    print("\nModel Performance on Training Set (may be optimistic):")
    print(f"  R-squared (R²): {r2_train:.4f}")
