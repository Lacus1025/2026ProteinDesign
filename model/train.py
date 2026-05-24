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
import shutil

SEED = 114514

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# 设置随机种子
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class GFP_Dataset(Dataset):
    def __init__(self, file):
        self.data = json.load(open(file))
        embeddings_path = file.replace(".json", "_embeddings.npy")
        self.embeddings = np.load(embeddings_path).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        embedding = torch.from_numpy(embedding)
        label = self.data[idx]["brightness"]
        label = torch.tensor(label, dtype=torch.float32)
        return embedding, label

def save_checkpoint(state, is_best, checkpoint_dir='checkpoints', filename='checkpoint.pth.tar'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)

    if is_best:
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pth.tar')
        shutil.copyfile(checkpoint_path, best_model_path)
        print(f"保存最佳模型到 {best_model_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"加载checkpoint: epoch {start_epoch}, loss {best_loss:.6f}")
        return start_epoch, best_loss
    return 0, float('inf')

def log_training_info(log_path, epoch, train_loss, val_loss=None, lr=None, is_checkpoint=False):
    logs = []
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []

    log_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "epoch": epoch,
        "train_loss": float(train_loss),
        "is_checkpoint": is_checkpoint
    }

    if val_loss is not None:
        log_entry["val_loss"] = float(val_loss)

    if lr is not None:
        log_entry["learning_rate"] = float(lr)

    logs.append(log_entry)

    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    return log_entry

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    true_values = []

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(data)

            predictions.extend(outputs.cpu().numpy())
            true_values.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader.dataset)
    r2 = r2_score(true_values, predictions)

    return avg_loss, r2, predictions, true_values

# 加载数据集
dataset = GFP_Dataset("./gfp_dataset.json")
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size],
    generator=torch.Generator().manual_seed(SEED)
)

# 使用多个worker加速数据加载
num_workers = 4 if device.type == 'cuda' else 0
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,
                         num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                        num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)

print("数据集大小:", len(dataset))
print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

from model.BrightnessRegressor import BrightnessRegressor

CONFIG = {
    "input_dim": 1152,
    "learning_rate": 0.0001,
    "num_epochs": 2000,
    "checkpoint_freq": 10,
    "early_stopping_patience": 200,
    "checkpoint_dir": "checkpoints",
    "log_file": "training_logs.json",
    "device": device.type
}

# 创建模型并移动到GPU
model = BrightnessRegressor(CONFIG["input_dim"])
model = model.to(device)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

# 可选：添加学习率调度器
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], 'checkpoint.pth.tar')
start_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer)

num_epochs = CONFIG["num_epochs"]
checkpoint_freq = CONFIG["checkpoint_freq"]
early_stopping_patience = CONFIG["early_stopping_patience"]
log_file = CONFIG["log_file"]

print(f"开始训练，从epoch {start_epoch + 1} 到 {num_epochs}")
print(f"Checkpoint保存频率: 每{checkpoint_freq}个epoch")
print(f"早停耐心值: {early_stopping_patience}")

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0

    # 训练阶段
    for data, labels in train_loader:
        # 将数据移动到GPU
        data, labels = data.to(device), labels.to(device)

        outputs = model(data)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # 验证阶段
    val_loss, val_r2, _, _ = evaluate_model(model, test_loader, criterion, device)

    # 学习率调度
    # scheduler.step(val_loss)

    # 打印训练信息
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, R2: {val_r2:.4f}, LR: {current_lr:.2e}")

    # 记录日志
    log_entry = log_training_info(log_file, epoch+1, avg_train_loss, val_loss, current_lr, is_checkpoint=False)

    # 检查是否为最佳模型
    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss = val_loss
        patience_counter = 0
        print(f"  -> 新的最佳模型！验证损失: {best_val_loss:.6f}")
    else:
        patience_counter += 1

    # 保存checkpoint
    if (epoch + 1) % checkpoint_freq == 0 or epoch == num_epochs - 1:
        checkpoint_state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_val_loss,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'config': CONFIG
        }
        save_checkpoint(checkpoint_state, is_best, CONFIG["checkpoint_dir"],
                       f'checkpoint_epoch_{epoch+1}.pth.tar')

        # 记录checkpoint日志
        log_training_info(log_file, epoch+1, avg_train_loss, val_loss, current_lr, is_checkpoint=True)

    # 早停检查
    if patience_counter >= early_stopping_patience:
        print(f"早停触发！在epoch {epoch+1} 停止训练")
        break

# 保存最终模型
final_model_path = 'model_final.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_loss': best_val_loss,
    'config': CONFIG
}, final_model_path)
print(f"最终模型保存到 {final_model_path}")

# 最终评估
print("\n" + "="*50)
print("最终模型评估")
print("="*50)

model.eval()
total_loss = 0
all_predictions = []
all_labels = []
detailed_results = []

with torch.no_grad():
    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * len(data)

        # 收集详细结果用于日志
        outputs_cpu = outputs.cpu()
        labels_cpu = labels.cpu()

        for i in range(len(outputs_cpu)):
            pred = outputs_cpu[i].item()
            true = labels_cpu[i].item()
            all_predictions.append(pred)
            all_labels.append(true)

            if true != 0:
                relative_error = (pred - true) / true
                rel_error_sq = relative_error ** 2
            else:
                rel_error_sq = (pred - true) ** 2

            detailed_results.append({
                "predicted": round(pred, 4),
                "true_value": round(true, 4),
                "absolute_error": round(pred - true, 4),
                "relative_error_squared": round(rel_error_sq, 6)
            })

            # 只打印前100个结果避免输出过多
            if len(detailed_results) <= 100:
                print(f"predicted:{pred:.4f}  labels:{true:.4f}    error:{pred - true:.4f}    loss:{rel_error_sq:.6f}")

avg_loss = total_loss / len(test_loader.dataset)
r2 = r2_score(all_labels, all_predictions)

# 保存评估结果到JSON
final_results = {
    "evaluation_summary": {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(test_loader.dataset),
        "mean_squared_error": round(avg_loss, 6),
        "r2_score": round(r2, 4),
        "best_val_loss_during_training": round(best_val_loss, 6),
        "final_epoch": epoch + 1,
        "device": device.type
    },
    "detailed_predictions": detailed_results[:50],  # 只保存前50个详细结果避免文件过大
    "training_config": CONFIG
}

# 追加或覆盖最终结果
# 保存评估结果到JSON - 简化版本
results_file = "evaluation_results.json"

# 创建新的评估结果条目
new_result = {
    "evaluation_summary": {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(test_loader.dataset),
        "mean_squared_error": round(avg_loss, 6),
        "r2_score": round(r2, 4),
        "best_val_loss_during_training": round(best_val_loss, 6),
        "final_epoch": epoch + 1,
        "device": device.type if torch.cuda.is_available() else 'cpu'
    },
    "detailed_predictions": detailed_results[:50],
    "training_config": CONFIG
}

# 读取现有结果或创建新列表
if os.path.exists(results_file):
    with open(results_file, 'r', encoding='utf-8') as f:
        try:
            results_list = json.load(f)
            # 确保是列表格式
            if not isinstance(results_list, list):
                results_list = [results_list]
        except:
            results_list = []
else:
    results_list = []

# 追加新结果
results_list.append(new_result)

# 保存
with open(results_file, 'w', encoding='utf-8') as f:
    json.dump(results_list, f, ensure_ascii=False, indent=2)

print(f"\n平均损失 (MSE): {avg_loss:.6f}")
print(f"R² 分数: {r2:.4f}")
print(f"评估结果已保存到: {results_file} (总共 {len(results_list)} 次评估)")

# 清理GPU缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"\nGPU显存使用情况:")
    print(f"  已分配: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"  已缓存: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
