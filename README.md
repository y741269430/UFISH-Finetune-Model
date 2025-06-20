# UFISH-test

## 目录 ####
- 0.加载环境
- 1.数据集及模型概况
- 2.数据集处理流程
- 3.UFISH Finetune
- 4.Spotiflow Finetune


---
## 0.加载环境 ####

## 1.数据集及模型概况 ####

## 2.数据集处理流程 ####

## 3.UFISH Finetune ####
```
cd UD2_finetune
```
#### 3.1 Finetune ####  
``` 
CUDA_VISIBLE_DEVICES=0 nohup ufish load-weights ../v1.0-alldata-ufish_c32.pth \
- train -n 100 \
../FISH_spots/UD2_datasets/train 100 \
../FISH_spots/UD2_datasets/val \
--model_save_dir ./model_finetune/ \
--batch_size 16 >> train_output.log 2>&1 &
```
The finally train log will be saved:
```
grep -E "Train Loss.*Val Loss" train_output.log  > train_val_loss.txt
```
The finally fine-tuned model will be saved:
```
ls model_finetune
```
#### 3.1.1 Check your fine-tuned model ####  
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
#### 3.2 Prediction & Evaluation ####  
After completing the training, we can select the best-performing model (e.g., `model_80.pth`) based on the validation dataset and evaluate its performance on the test dataset.
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
After obtaining the `eval_UD2_text.csv` file, we can use the following code to output the mean F1 score in the current window, in order to evaluate your results.
```python
import pandas as pd
eval_df = pd.read_csv("./eval.csv")
eval_df = eval_df[eval_df['true num'] != 0]  # remove empty image
print("Mean f1(cutoff=3.0):", eval_df['f1(cutoff=3)'].mean())
```
#### 3.3 Train a new model ####
If you are not satisfied with your fine-tuned model and its evaluation results, you can choose to train a new model using these training and validation datasets. 
```
CUDA_VISIBLE_DEVICES=0 nohup ufish train \
- train -n 100 \
../FISH_spots/UD2_datasets/train 100 \
../FISH_spots/UD2_datasets/val \
--model_save_dir ./new_model/ \
--batch_size 16 >> train_output.log 2>&1 &
```
After the training is completed, you can continue running the code as described above:
```
grep -E "Train Loss.*Val Loss" train_output.log  > train_val_loss.txt
```
Then, proceed with Steps 3.1.1, 3.2, and 3.3.

---
## 4.Spotiflow Finetune ####
```
cd spt_finetune
```
#### 4.1 Finetune ####  
Upon the first execution, the general model will be automatically downloaded and saved to `~/.spotiflow/models/general`
```
CUDA_VISIBLE_DEVICES=0 nohup spotiflow-train ../FISH_spots/UD2_datasets/ \
--finetune-from ./general \
-o model_finetune \
--batch-size 4 \
--device cuda \
--num-epochs 100 >> train_output.log 2>&1 &
```
The finally train log will be saved:
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

# 解析并去重
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
#### 4.1.1 Check your fine-tuned model ####  
```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = "./train_val_loss.txt"
data = pd.read_csv(file_path)

# 提取需要绘制的列
columns_to_plot = ['Epoch', 'val_loss', 'train_loss']

# 创建一个新的图形
plt.figure(figsize=(9, 5))

# 绘制每列的数据
for column in columns_to_plot[1:]:
    plt.plot(data['Epoch'], data[column], label=column)

# 添加标题和标签
plt.title('Training Metrics Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.grid(True)
plt.legend()

plt.xlim(0, 100)
plt.xticks(range(0, 101, 5))

save_path = "./loss_plot.pdf"
plt.savefig(save_path, dpi=300, bbox_inches='tight') 

# 显示图表
plt.show()
```
#### 4.2 Prediction & Evaluation ####  
```
# run prediction 
CUDA_VISIBLE_DEVICES=0 nohup spotiflow-predict ../FISH_spots/UD2_datasets/test \
-md ./model_finetune -o ./predict_test -t 0.5 -d cuda &
```
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
#### 5.1 Predict ####  
```
nohup deepblink predict --model ./230903_210048_deepBlink_is_sweet.h5 -i ../FISH_spots/UD2_datasets/test/ -o ./predict/ &

nohup deepblink predict --model ./fine_md/250610_224839_deepBlink_is_sweet.h5 -i ../FISH_spots/UD2_datasets/test/ -o ./predict_finetune/ &

```












