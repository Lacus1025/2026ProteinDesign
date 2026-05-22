#!/bin/bash

# 创建logs目录
mkdir -p logs

# 生成带时间戳的日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/training_${TIMESTAMP}.log"

conda activate esm

pip install -r requirements.txt

echo "开始构建数据集"

python utils/export_dataset_json.py

echo "开始训练，日志保存到: ${LOG_FILE}"

# 运行训练并保存日志
python model/train.py 2>&1 | tee "${LOG_FILE}"

echo "训练完成，日志已保存到: ${LOG_FILE}"
