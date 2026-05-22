import datetime
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import torch.nn as nn
import torch.optim as optim

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
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size],
    generator=torch.Generator().manual_seed(SEED)
)

# 实例化 DataLoader
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)


for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
    print(f"批次 {batch_idx + 1}")
    print("数据:", batch_data)
    print("标签:", batch_labels)
    if batch_idx == 2:  # 仅显示前 3 个批次
        break

# 测试数据集
print("数据集大小:", len(dataset))

from model.BrightnessRegressor import BrightnessRegressor

model = BrightnessRegressor(960)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0000001)

num_epochs = 10000
model.train()  # 设置模型为训练模式

for epoch in range(num_epochs):
    total_loss = 0
    for data, labels in train_loader:
        outputs = model(data)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失

        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

torch.save(model, 'model.pth')

model.eval()  # 设置模型为评估模式

with torch.no_grad():  # 关闭梯度计算
    for data, labels in test_loader:
        outputs = model(data)
        for i in range(len(outputs)):
            print(f"predicted:{outputs[i].item():.4f}  labels:{labels[i].item():.4f}    error:{outputs[i].item()-labels[i].item():.4f}")
        print()
