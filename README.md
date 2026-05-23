## USAGE

~~~bash
conda create -n esm python

conda activate esm

pip install -r requirements.txt

python utils/export_dataset_json.py

python model/train.py
~~~

## todo:
<!-- - model/train.py保存模型功能 -->
- model/train.py根据不同type训练不同模型                    (待定)
- utils/export_dataset_json.py根据不同type导出不同数据集    (待定)
- utils/remove_exclusion_seq.py移除不符生成结果
- model/eval.py评估序列
- model/generate.py生成序列
- model/generate.py打分hook(可选)

##
