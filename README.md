# UFISH-test

## 目录 ####
- 0.加载环境
- 1.数据集及模型概况
- 2.数据集处理流程
- 3.UFISH Finetune
- 4.Spotiflow Finetune
- 5.deepBlink Finetune
- 6.Rule-based


---
## 0.加载环境 ####
```
# 输出环境
# conda env export > ufish.yml 
# 安装环境
conda env create -f ufish.yml

conda activate ufish
```
## 1.数据集及模型概况 ####
目前数据集主要分为了3个部分，`UD1`, `UD2（文章提交）`, `UD3（250715最新）`    
其中，model40.pth是使用了UD2

## 2.数据集处理流程 ####
见[2.数据集处理流程](https://github.com/y741269430/UFISH-test/blob/main/%E6%95%B0%E6%8D%AE%E9%9B%86%E5%A4%84%E7%90%86%E6%B5%81%E7%A8%8B.md)

## 3.UFISH Finetune ####
```
cd UD2_finetune
```
#### 3.1 直接运行默认模型 ####
```
nohup ufish load_weights v1.0-alldata-ufish_c32.pth \
- predict_imgs ../FISH_spots/UD2_datasets/text \
predict --img_glob=\"*.tif\" \
--intensity_threshold 0.5 &
```
#### 3.2 Finetune ####  
``` 
CUDA_VISIBLE_DEVICES=0 nohup ufish load-weights ../v1.0-alldata-ufish_c32.pth \
- train -n 100 \
../FISH_spots/UD2_datasets/train 100 \
../FISH_spots/UD2_datasets/val \
--model_save_dir ./model_finetune/ \
--batch_size 16 >> train_output.log 2>&1 &
```
输出日志转换成txt用于绘图：
```
grep -E "Train Loss.*Val Loss" train_output.log  > train_val_loss.txt
```
模型存放于此：
```
ls model_finetune
```
#### 3.2.1 Check your fine-tuned model ####  
```python
import re
import matplotlib.pyplot as plt

epochs = []
train_loss = []
val_loss = []

pattern = r"Epoch (\d+)/\d+, Train Loss: ([\d\.]+), Val Loss: ([\d\.]+)"

with open("./train_val_loss.txt", "r") as file:
    for line in file:
        match = re.search(pattern, line)
        if match:
            epoch = int(match.group(1))  # 提取 Epoch
            train_loss_value = float(match.group(2))  # 提取 Train Loss
            val_loss_value = float(match.group(3))  # 提取 Val Loss
            epochs.append(epoch)
            train_loss.append(train_loss_value)
            val_loss.append(val_loss_value)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, 'b', label='Train Loss')
plt.plot(epochs, val_loss, 'r', label='Val Loss')

plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.grid(True)
plt.legend()

plt.xlim(0, max(epochs))
plt.xticks(range(0, 101, 5))

# 如果有最佳验证损失的 Epoch，可以添加一条竖线标记
# best_epoch = 80  # 最佳 Epoch
# plt.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Val Epoch ({best_epoch})')

# save_path = "./loss_plot.pdf"
# plt.savefig(save_path, dpi=300, bbox_inches='tight') 

# 显示图表
plt.show()
```
#### 3.3 Prediction & Evaluation ####  
训练完后，根据上述图表寻找最优的模型 (e.g., `model_80.pth`) 
```
# run prediction 
nohup ufish load_weights ./model_finetune/model_80.pth \
- predict_imgs ../FISH_spots/UD2_datasets/test \
./predict_test --img_glob="*.tif" \
--intensity_threshold 0.1 \
--axes='yx' &
```
```
# run evaluation
nohup ufish evaluate_imgs ./predict_test ../FISH_spots/UD2_datasets/test ./eval_UD2_text.csv &
```
获取`eval_UD2_text.csv`后，可使用以下代码输出平均F1 score
```python
import pandas as pd
eval_df = pd.read_csv("./eval.csv")
eval_df = eval_df[eval_df['true num'] != 0]  # remove empty image
print("Mean f1(cutoff=3.0):", eval_df['f1(cutoff=3)'].mean())
```
#### 3.4 Train a new model ####
使用以下代码执行重头训练（获得一个新的模型）
```
CUDA_VISIBLE_DEVICES=0 nohup ufish train \
- train -n 100 \
../FISH_spots/UD2_datasets/train 100 \
../FISH_spots/UD2_datasets/val \
--model_save_dir ./new_model/ \
--batch_size 16 >> train_output.log 2>&1 &
```
运行完毕后，执行以下命令获取日志
```
grep -E "Train Loss.*Val Loss" train_output.log  > train_val_loss.txt
```
接下来依次执行 3.2.1, 3.3.

---
## 4.Spotiflow Finetune ####
```
cd spt_finetune
```
#### 4.1 直接运行默认模型 ####
```
spotiflow-predict ../FISH_spots/UD2_datasets/test -pm general -o predict -t 0.5 -d gpu &
```
#### 4.2 Finetune ####  
第一次执行的时候，会下载模型及数据集，存放于`~/.spotiflow/models/general`
```
CUDA_VISIBLE_DEVICES=0 nohup spotiflow-train ../FISH_spots/UD2_datasets/ \
--finetune-from ./general \
-o model_finetune \
--batch-size 4 \
--device cuda \
--num-epochs 100 >> train_output.log 2>&1 &
```
使用以下代码转换输出的log:
```python
import pandas as pd
import re

# 文件路径
log_file = "./train_output.log"
output_csv = "./train_val_loss.txt"

# 解析函数
def parse_log(file_path):
    pattern = re.compile(
        r"Epoch (\d+):.*?" +
        r"val_loss=([\d\.]+).*?" +
        r"val_f1=([\d\.]+).*?" +
        r"val_acc=([\d\.]+).*?" +
        r"heatmap_loss=([\d\.]+).*?" +
        r"flow_loss=([\d\.]+).*?" +
        r"train_loss=([\d\.]+)"
    )
    
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line.strip())  # 使用search而不是match
            if match:
                records.append({
                    'Epoch': int(match.group(1)),
                    'val_loss': float(match.group(2)),
                    'val_f1': float(match.group(3)),
                    'val_acc': float(match.group(4)),
                    'heatmap_loss': float(match.group(5)),
                    'flow_loss': float(match.group(6)),
                    'train_loss': float(match.group(7))
                })
    return pd.DataFrame(records)
try:
    df = parse_log(log_file)
    if not df.empty:
        df_unique = df.drop_duplicates(subset=['Epoch'], keep='last')
        df_unique.to_csv(output_csv, index=False)
        print("处理成功，结果已保存到:", output_csv)
        print(df_unique)
    else:
        print("警告: 没有找到任何匹配的Epoch数据！")
except Exception as e:
    print("发生错误:", str(e))
```
#### 4.2.1 Check your fine-tuned model ####  
```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = "./train_val_loss.txt"
data = pd.read_csv(file_path)

columns_to_plot = ['Epoch', 'val_loss', 'train_loss']

plt.figure(figsize=(9, 5))

for column in columns_to_plot[1:]:
    plt.plot(data['Epoch'], data[column], label=column)

plt.title('Training Metrics Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.grid(True)
plt.legend()

plt.xlim(0, 100)
plt.xticks(range(0, 101, 5))

#save_path = "./loss_plot.pdf"
#plt.savefig(save_path, dpi=300, bbox_inches='tight') 

plt.show()
```
#### 4.3 Prediction & Evaluation ####  
```
# run prediction 
CUDA_VISIBLE_DEVICES=0 nohup spotiflow-predict ../FISH_spots/UD2_datasets/test \
-md ./model_finetune -o ./predict_test -t 0.5 -d cuda &
```
使用脚本对输出csv进行格式转换
```
# run evaluation 1 (Processing... Please wait.)
bash spt2pred.sh ./predict_test
```
```
# run evaluation 2
nohup ufish evaluate_imgs ./predict_test ../FISH_spots/UD2_datasets/test ./eval_UD2_text.csv &
```

---
## 5.deepBlink Finetune ####
```
cd dpl_finetune
```
#### 5.1 直接运行默认模型 ####
使用以下代码下载模型，或网站下载[here](https://figshare.com/articles/software/Download/13220717)
```
deepblink download
```
```
deepblink predict --model dpbl_model/smfish.h5 -i ../FISH_spots/UD2_datasets/test -o ./predict &
```
#### 5.2 Predict ####  
```
nohup deepblink predict --model ./230903_210048_deepBlink_is_sweet.h5 -i ../FISH_spots/UD2_datasets/test/ -o ./predict_test/ &
```
这里需要手动删除一些只有表头的文件，不然执行以下代码会报错
```
# 检查只有表头的文件
wc -l predict/*.csv | awk '$1 < 3 && $1 != "total" {print $2}'
# 然后把这些只有表头的文件删除
wc -l predict/*.csv | awk '$1 < 3 && $1 != "total" {system("rm -f "$2)}'
```
使用脚本对输出csv进行格式转换
```
# 对输出的结果进行转换
python dpl2pred.py
```
```
# run evaluation
nohup ufish evaluate_imgs ./predict_test ../FISH_spots/UD2_datasets/test ./eval_raw.csv &
```
#### 5.3 Train ####
#### 5.3.1 Trans UD2_datasets to npz ####
使用以下脚本对`UD2_datasets`转换为`deepblink.npz`
```python
import os
import numpy as np
from skimage import io
import pandas as pd

# 数据根目录
base_path = './FISH_spots/UD2_datasets/'

# 输出文件名
output_name = 'deepblink'

def load_dataset_from_folder(folder_path):
    """
    加载指定文件夹下的所有 tif/csv 对应数据
    """
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    images = []
    labels = []

    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        csv_file = base_name + '.csv'
        img_path = os.path.join(folder_path, img_file)
        csv_path = os.path.join(folder_path, csv_file)

        # 读取图像
        img = io.imread(img_path)
        images.append(img)

        # 读取 CSV 标注点
        try:
            df = pd.read_csv(csv_path, header=1)
            labels.append(df.values)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            labels.append(np.zeros(shape=(0, 2)))

    return np.array(images), np.array(labels, dtype=object)

# 分别加载三个集合
train_data, train_labels = load_dataset_from_folder(os.path.join(base_path, 'train'))
valid_data, valid_labels = load_dataset_from_folder(os.path.join(base_path, 'val'))
test_data, test_labels = load_dataset_from_folder(os.path.join(base_path, 'test'))

# 保存为 .npz 文件
np.savez(
    f"./{output_name}.npz",
    x_train=train_data,
    y_train=train_labels,
    x_valid=valid_data,
    y_valid=valid_labels,
    x_test=test_data,
    y_test=test_labels
)

print("✅ 数据集已成功加载并保存！")
```
#### 5.3.2 Train ####
执行以下代码，会生成一个`config.yaml`，然后进去填写路径，betchsize等信息
```
deepblink config
```
运行
```
nohup deepblink train --config config.yaml -g 0 &
```
#### 5.4 Prediction & Evaluation ####  
```
nohup deepblink predict --model ./fine_md/250610_224839_deepBlink_is_sweet.h5 -i ../FISH_spots/UD2_datasets/test/ -o ./predict_finetune/ &
```
这里需要手动删除一些只有表头的文件，不然执行以下代码会报错
```
# 检查只有表头的文件
wc -l predict_finetune/*.csv | awk '$1 < 3 && $1 != "total" {print $2}'
# 然后把这些只有表头的文件删除
wc -l predict_finetune/*.csv | awk '$1 < 3 && $1 != "total" {system("rm -f "$2)}'
```
使用脚本对输出csv进行格式转换
```
# 对输出的结果进行转换
python dpl2pred.py
```
```
# run evaluation
nohup ufish evaluate_imgs ./predict_finetune ../FISH_spots/UD2_datasets/test ./eval_UD2_text.csv &
```

---
#### 比较 F1 score 或 Mean distance ####
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 定义文件路径和模型名称
data_sources = [
    ("./UD2_finetune/eval_raw.csv", "U-fish") ,
    ("./UD2_finetune/eval_UD2_text.csv", "U-fish Finetune"),
    ("./spt_finetune/eval_raw.csv", "Spotiflow"),
    ("./spt_finetune/eval_UD2_text.csv", "Spotiflow Finetune"),
    ("./dp_finetune/eval_raw.csv", "deepBlink"),
    ("./dp_finetune/eval_UD2_text.csv", "deepBlink Finetune"),
    ("./new-ud2/eval_UD2_text.csv", "BF_test1.0_f1"),
    ("./new-ud2/eval_UD2_text.csv", "RS_test1.0_f1"),
    ("./new-ud2/eval_UD2_text.csv", "ST_test1.0_f1"),
    ("./new-ud2/eval_UD2_text.csv", "TM_test1.0_f1")
]

# 定义目标列名
source_column_name = 'source'

# 定义需要绘制的指标
target_metrics = ["f1(cutoff=3)"]
#target_metrics = ["mean distance"]

# 初始化一个空的 DataFrame 用于存储合并数据
all_data = pd.DataFrame()

# 读取并合并所有 CSV 数据
for file, model_name in data_sources:
    data = pd.read_csv(file)
    data['model'] = model_name  # 添加模型名称
    all_data = pd.concat([all_data, data], ignore_index=True)

# 按照'source'列的首字母顺序对数据进行排序
all_data[source_column_name] = all_data[source_column_name].astype(str)
all_data = all_data.sort_values(by=source_column_name, key=lambda col: col.str.lower())

# 设置模型顺序

model_order = ["U-fish", "U-fish Finetune",  "Spotiflow", "Spotiflow Finetune", "deepBlink", "deepBlink Finetune", "Big-FISH", "RS-FISH", "Starfish", "TrackMate"]

# 循环绘制每个指标的箱线图
for target_metric in target_metrics:
    plt.figure(figsize=(11,6))
    sns.boxplot(
        x=source_column_name, 
        y=target_metric, 
        hue="model",  # 使用模型名称区分不同数据来源
        data=all_data,
        showmeans=False,  # 显示均值点
        meanprops={'marker':'D', 'markerfacecolor':'red'},  # 红色菱形表示均值
        palette="tab10",  # 设置颜色调色板
        hue_order=model_order  # 设置模型顺序
    )

    # 添加一条水平线 (y=0.85)
    #plt.axhline(y=0.85, color='r', linestyle='--', linewidth=1.5, label='Threshold (0.85)')
    
    # 设置 y 轴刻度
    if target_metric == "f1(cutoff=3)":
        plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95,1]) 
    elif target_metric == "mean distance":
        plt.yticks([0, 1]) 

    # 设置标题
    if target_metric == "f1(cutoff=3)":
        plt.title("Performance comparison on UD2")
    elif target_metric == "mean distance":
        plt.title("Performance comparison on UD2")

    # 其他设置
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.4, 1), loc = 'upper right')  # 显示图例，包含水平线标签
    plt.tight_layout()

    #save_path = "/home/jjyang/jupyter_file/boxplot_ud2_all.pdf"
    #plt.savefig(save_path, dpi=300, bbox_inches='tight') 

    # 显示图像
    plt.show()
```

## 6.Rule-based ##    
使用cpu计算以下软件的运行时间
```bash
# Big-FISH
CUDA_VISIBLE_DEVICES=-1 taskset -c 67-86 nohup bash -c "time -p python BF_test.py ./test/ /home/jjyang/jupyter_file/test_speed/predict_BF/ > ./test_bf.log" > program1.log 2> time1.log &

# Starfish
CUDA_VISIBLE_DEVICES=-1 taskset -c 67-86 nohup bash -c "time -p python ST_test.py ./test/ /home/jjyang/jupyter_file/test_speed/predict_ST/ > ./test_st.log" > program1.log 2> time1.log &

# RS-FISH
CUDA_VISIBLE_DEVICES=-1 taskset -c 67-86 nohup bash -c "time -p ../env/Fiji.app/ImageJ-linux64 --headless --run RS_test.ijm > ./test_rs.log " > program1.log 2> time1.log &

# TrackMate
CUDA_VISIBLE_DEVICES=-1 taskset -c 67-86 nohup bash -c "time -p ../env/Fiji.app/ImageJ-linux64 --ij2 --mem=24G --headless --run TM_test.py > ./test_tm.log" > program1.log 2> time1.log &
```






