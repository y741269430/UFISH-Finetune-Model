# UFISH-test

## 目录 ####
- 0.加载环境
- 1.数据集及模型概况
- 2.数据集处理流程
- 3.UFISH Finetune
- 4.Spotiflow Finetune
- 5.deepBlink Finetune


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

## 2.数据集处理流程 ####
### 2.1 依据图片上的命名，拆分通道    
举例说明我是如何把一个TIF，根据文件名，拆分通道的     
```python
#### 神经节 ####

from PIL import Image
import os

# 定义路径
input_dir = '/home/jjyang/jupyter_file/my_finetune/call点图像整理/神经节/2/bigwarp'  # 输入文件夹（包含多通道 TIF 文件）
output_dir = '/home/jjyang/jupyter_file/my_finetune/call点图像整理/神经节/2/RS-FISH'  # 输出文件夹

# 创建输出文件夹（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 遍历输入文件夹中的所有 TIF 文件
for file_name in os.listdir(input_dir):
    if not file_name.endswith('.tif'):
        continue  # 跳过非 TIF 文件

    tif_path = os.path.join(input_dir, file_name)  # 完整路径
    print(f"正在处理文件：{tif_path}")

    try:
        # 打开多通道 TIF 文件
        img = Image.open(tif_path)

        # 解析 TIF 文件名，提取基因名称
        base_name, _ = os.path.splitext(file_name)  # 去掉扩展名
        parts = base_name.split('-')[-1].split(' ')  # 提取最后一个部分并分割基因名称
        gene_names = [part for part in parts if part]  # 过滤空字符串

        # 确保基因名称数量不超过通道数 - 1（因为第一个通道是 DAPI）
        if len(gene_names) > img.n_frames - 1:
            raise ValueError(f"基因名称数量超过通道数！TIF 文件 {file_name} 包含 {img.n_frames} 个通道，但找到 {len(gene_names)} 个基因名称。")

        # 循环遍历所有帧/通道
        for i in range(img.n_frames):
            img.seek(i)  # 移动到指定帧
            frame = img.copy()

            # 确定新文件名
            if i == 0:
                new_file_name = "DAPI.tif"  # 第一个通道是 DAPI
            elif i <= len(gene_names):
                gene_name = gene_names[i - 1]  # 对应的基因名称
                new_file_name = f"{gene_name}.tif"
            else:
                new_file_name = f"Unknown_Ch{i}.tif"  # 如果基因名称不足，则使用默认名称

            # 保存图片
            output_path = os.path.join(output_dir, new_file_name)
            frame.save(output_path)
            print(f"已保存：{output_path}")

    except Exception as e:
        print(f"处理文件 {tif_path} 时出错：{e}")
```
结果如下：    
<img src="https://github.com/y741269430/UFISH-test/blob/main/Imgs/p1.jpg" width="800" />      

### 2.2 把TIF以及CSV复制到新目录下    
举例说明我是如何把上面2.1所拆分的TIF文件，复制到新的目录下（规则是，根据原TIF的路径，组合成一个新文件名）     
```python
import os
import shutil

# 定义源目录和目标目录路径
source_root_dir = '/home/jjyang/jupyter_file/my_finetune/call点图像整理/神经节'
target_dir = '/home/jjyang/jupyter_file/my_finetune/call点图片及csv/'

# 创建目标目录（如果不存在）
os.makedirs(target_dir, exist_ok=True)

def copy_and_rename_files(source_path, target_path):
    try:
        print(f"Copying and renaming files from {source_path} to {target_path}:\n")
        
        # 提取源目录的关键信息（例如父文件夹名称）
        parent_folder_name = os.path.basename(os.path.dirname(source_path))  # 获取上一级文件夹名称
        grandparent_folder_name = os.path.basename(os.path.dirname(os.path.dirname(source_path)))  # 获取上两级文件夹名称
        
        # 遍历源目录中的所有文件
        for file_name in os.listdir(source_path):
            source_file_path = os.path.join(source_path, file_name)
            
            # 筛选 .csv 和 .tif 文件
            if file_name.lower().endswith(('.csv', '.tif')):
                # 提取文件名和扩展名
                base_name, ext = os.path.splitext(file_name)
                
                # 构造新文件名
                new_file_name = f"{grandparent_folder_name}-{parent_folder_name}-{base_name}{ext}"
                target_file_path = os.path.join(target_path, new_file_name)
                
                # 复制并重命名文件
                shutil.copy(source_file_path, target_file_path)
                print(f"Copied and renamed: {file_name} -> {new_file_name}")
    
    except FileNotFoundError:
        print(f"The directory {source_path} does not exist.")
    except PermissionError:
        print(f"Permission denied accessing {source_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")

def process_all_subfolders(root_dir, target_dir):
    """
    遍历 root_dir 下的所有子文件夹，并对每个符合条件的子文件夹调用 copy_and_rename_files。
    """
    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)
        
        # 检查是否为目录
        if os.path.isdir(subfolder_path):
            rs_fish_path = os.path.join(subfolder_path, 'RS-FISH')
            
            # 如果该子文件夹包含 RS-FISH 子目录，则处理它
            if os.path.exists(rs_fish_path) and os.path.isdir(rs_fish_path):
                print(f"\nProcessing subfolder: {subfolder}")
                copy_and_rename_files(rs_fish_path, target_dir)

# 调用函数处理所有子文件夹
process_all_subfolders(source_root_dir, target_dir)

print("\nAll files have been processed.")
```
结果如下：    
<img src="https://github.com/y741269430/UFISH-test/blob/main/Imgs/p2.jpg" width="800" />    

### 2.3 把RS-FISH生成的CSV结果，转换成两列    
举例说明如何把CSV文件取出其中两列，并重命名，更换列顺序    
```python
import os
import pandas as pd

def rename_and_reorder_csv_columns(directory_path):
    """遍历指定目录下的所有 .csv 文件，并将列名修改为 axis-0 和 axis-1，同时调整列顺序"""
    try:
        print(f"Processing CSV files in directory: {directory_path}\n")
        
        # 遍历目录中的所有文件
        for file_name in os.listdir(directory_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(directory_path, file_name)
                
                try:
                    # 读取CSV文件
                    df = pd.read_csv(file_path)
                    
                    # 检查是否有 'x' 和 'y' 列
                    if 'x' not in df.columns or 'y' not in df.columns:
                        print(f"Skipping file {file_name} because it does not contain 'x' and 'y' columns.")
                        continue
                    
                    # 修改列名为 axis-1 和 axis-0
                    df.rename(columns={'y': 'axis-0', 'x': 'axis-1'}, inplace=True)
                    
                    # 调整列顺序，使 axis-0 在第一列，axis-1 在第二列
                    df = df[['axis-0', 'axis-1']]
                    
                    # 保存修改后的CSV文件
                    df.to_csv(file_path, index=False)
                    print(f"Successfully processed file: {file_name}")
                
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}\n")
    
    except FileNotFoundError:
        print(f"The directory {directory_path} does not exist.")
    except PermissionError:
        print(f"Permission denied accessing {directory_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")

# 示例调用
# 请替换以下参数为你实际的目录路径
directory_path = '/home/jjyang/jupyter_file/my_finetune/temp'

rename_and_reorder_csv_columns(directory_path)
```
结果如下：    
<img src="https://github.com/y741269430/UFISH-test/blob/main/Imgs/p3.jpg" width="550" />    
<img src="https://github.com/y741269430/UFISH-test/blob/main/Imgs/p4.jpg" width="550" />   

### 2.4 裁剪TIF以及CSV    
举例说明如何把文件夹里面的TIF以及CSV，裁剪成指定大小（512*512）    
```python
import os
import pandas as pd
from skimage.io import imread, imsave
import logging

# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义输入和输出目录
image_input_dir = '/home/jjyang/jupyter_file/my_finetune/temp/'
csv_input_dir = image_input_dir
image_output_dir = '/home/jjyang/jupyter_file/my_finetune/temp_cut/'
csv_output_dir = image_output_dir

os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(csv_output_dir, exist_ok=True)

# 定义裁剪尺寸
crop_size = 512

def process_image_and_csv(image_path, csv_path):
    try:
        # 读取原始图像
        im0 = imread(image_path)
    except Exception as e:
        logging.error(f"Error reading image file {image_path}: {e}")
        return

    try:
        # 读取原始CSV文件
        df0 = pd.read_csv(csv_path, usecols=[0, 1])  # 假设CSV只有两列：Y和X
    except Exception as e:
        logging.error(f"Error reading CSV file {csv_path}: {e}")
        return

    # 检查并转换数据类型
    def convert_to_numeric(df, column):
        df.loc[:, column] = pd.to_numeric(df[column], errors='coerce')
        return df.dropna(subset=[column])
    
    df0 = convert_to_numeric(convert_to_numeric(df0, df0.columns[0]), df0.columns[1])

    total_points_before = len(df0)  # 记录原始点的数量
    logging.info(f"Total points before cropping: {total_points_before}")

    if total_points_before == 0:
        logging.warning(f"No points found in the original CSV file {csv_path}. Skipping processing.")
        return

    # 获取图像文件名的基名（不包括扩展名）
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 遍历图像进行裁剪
    for start_y in range(0, im0.shape[0], crop_size):
        for start_x in range(0, im0.shape[1], crop_size):
            end_y = min(start_y + crop_size, im0.shape[0])
            end_x = min(start_x + crop_size, im0.shape[1])

            # 确保当前裁剪区域大小为512x512
            if (end_y - start_y == crop_size) and (end_x - start_x == crop_size):
                # 裁剪图像
                cropped_im0 = im0[start_y:end_y, start_x:end_x]

                # 构建输出文件名
                image_filename = f"{base_name}_cropped_{start_y}_{start_x}.tif"
                image_filepath = os.path.join(image_output_dir, image_filename)

                # 保存裁剪后的图像
                try:
                    imsave(image_filepath, cropped_im0)
                    logging.info(f"Cropped image saved to {image_filepath}")
                except Exception as e:
                    logging.error(f"Error saving cropped image to {image_filepath}: {e}")
                    continue

                # 过滤出位于当前裁剪区域内的点
                cropped_df = df0[(df0[df0.columns[0]] >= start_y) & (df0[df0.columns[0]] < end_y) &
                                 (df0[df0.columns[1]] >= start_x) & (df0[df0.columns[1]] < end_x)].copy()

                total_points_after = len(cropped_df)
                logging.info(f"Points before cropping: {total_points_before}, after cropping: {total_points_after}.")

                # 如果裁剪后的DataFrame为空，则跳过保存
                if cropped_df.empty:
                    logging.warning("No points found in this cropped region. Skipping CSV save.")
                    continue

                # 调整坐标系
                cropped_df.loc[:, df0.columns[0]] -= start_y
                cropped_df.loc[:, df0.columns[1]] -= start_x

                # 构建输出文件名
                csv_filename = f"{base_name}_cropped_{start_y}_{start_x}.csv"
                csv_filepath = os.path.join(csv_output_dir, csv_filename)

                # 保存调整后的CSV文件
                try:
                    cropped_df.to_csv(csv_filepath, index=False)
                    logging.info(f"Adjusted CSV saved to {csv_filepath}")
                except Exception as e:
                    logging.error(f"Error saving adjusted CSV to {csv_filepath}: {e}")

# 获取所有图像文件路径
image_files = [f for f in os.listdir(image_input_dir) if f.endswith('.tif')]

for image_file in image_files:
    # 构建图像文件的完整路径
    image_path = os.path.join(image_input_dir, image_file)

    # 根据图像文件名构建对应的CSV文件名
    base_name = os.path.splitext(image_file)[0]
    csv_file = f"{base_name}.csv"
    csv_path = os.path.join(csv_input_dir, csv_file)

    # 检查CSV文件是否存在
    if not os.path.exists(csv_path):
        logging.warning(f"Warning: CSV file {csv_file} does not exist.")
        continue

    # 处理当前图像和CSV文件
    process_image_and_csv(image_path, csv_path)

logging.info("Batch cropping and CSV processing completed.")
```
检查一下一共有多少张TIF及CSV
```python
import os

def count_files(directory):
    return len([item for item in os.listdir(directory) 
               if os.path.isfile(os.path.join(directory, item))])

# 示例
file_count = count_files('/home/jjyang/jupyter_file/my_finetune/temp_cut/')
print(f"文件数量：{file_count}")
```

### 2.5 删除不配对的TIF和CSV      
上述2.4裁剪了TIF以及CSV，然而TIF里面一些空白的地方，是没有CSV的结果的，所以在裁剪的时候会丢失这部分信息，即TIF与CSV不对齐了。于是我们把这些没有对齐的文件进行删除    
```python
base_directory = '/home/jjyang/jupyter_file/my_finetune/'
folder_name = 'temp_cut'

from pathlib import Path
import os

def get_base_filename(filename):
    """从文件名中提取基础部分（去掉扩展名）"""
    return filename.stem

def filter_files_by_pairs(base_directory, folder_name):
    """过滤指定文件夹中的文件，只保留同时具有 .tif 和 .csv 的文件对"""
    # 构建完整路径
    folder_path = Path(base_directory) / folder_name

    if not folder_path.exists():
        print(f"The directory {folder_path} does not exist.")
        return

    tif_files = {}
    csv_files = {}

    # 收集所有 .tif 和 .csv 文件
    for item in folder_path.iterdir():
        if item.is_file():
            base_name = get_base_filename(item)
            if base_name:
                if item.suffix == '.tif':
                    tif_files[base_name] = item
                elif item.suffix == '.csv':
                    csv_files[base_name] = item

    # 找出同时存在 .tif 和 .csv 的文件对
    paired_bases = set(tif_files.keys()) & set(csv_files.keys())

    # 删除那些没有配对的文件
    for base_name, file_path in list(tif_files.items()) + list(csv_files.items()):
        if base_name not in paired_bases:
            try:
                os.remove(file_path)
                print(f"Deleted unpaired file: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

    # 打印保留的文件对
    for base_name in paired_bases:
        print(f"Kept paired files:")
        print(f"  TIF: {tif_files[base_name]}")
        print(f"  CSV: {csv_files[base_name]}")

filter_files_by_pairs(base_directory, folder_name)
```
结果如下：    
<img src="https://github.com/y741269430/UFISH-test/blob/main/Imgs/p5.jpg" width="600" />  


### 2.6    
```python
import os
import random
import shutil
from collections import defaultdict
import sys

# 自定义的日志文件类，用于重定向 stdout
class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

# 设置日志文件路径
log_path = '/home/jjyang/jupyter_file/finetune_model/dataset_split/output.log'
os.makedirs(os.path.dirname(log_path), exist_ok=True)
sys.stdout = Logger(log_path)

# 下面是原有逻辑的函数和主流程
def get_unique_sample_names(folder_path):
    sample_files = defaultdict(list)
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp', '.csv')):
            name_part = filename.rsplit('.', 1)[0]
            if "_cropped_" in name_part:
                sample_name = name_part.split("_cropped_")[0]
                sample_files[sample_name].append(filename)
    
    return sample_files

def split_samples(sample_dict, test_ratio=0.2, train_ratio=0.6, val_ratio=0.2):
    assert test_ratio + train_ratio + val_ratio == 1.0, "比例总和必须为1"

    samples = list(sample_dict.keys())
    random.shuffle(samples)

    total = len(samples)
    test_end = int(total * test_ratio)
    train_end = test_end + int(total * train_ratio)

    test_samples = samples[:test_end]
    train_samples = samples[test_end:train_end]
    val_samples = samples[train_end:]

    return {
        'test': test_samples,
        'train': train_samples,
        'val': val_samples
    }

def copy_samples_to_folders(sample_dict, folder_path, split_result, target_base_dir):
    # 创建目标文件夹
    for folder in ['mytest', 'mytrain', 'myval']:
        os.makedirs(os.path.join(target_base_dir, folder), exist_ok=True)

    # 复制文件
    for folder, sample_list in split_result.items():
        for sample_name in sample_list:
            for filename in sample_dict[sample_name]:
                src = os.path.join(folder_path, filename)
                dst = os.path.join(target_base_dir, folder, filename)
                shutil.copy(src, dst)  # 如果是移动而不是复制，用 shutil.move
                print(f"已复制 {filename} 到 {folder}")

# 使用示例
folder_path = '/home/jjyang/jupyter_file/my_finetune/temp_cut/'
target_base_dir = '/home/jjyang/jupyter_file/finetune_model/'

print("🔍 正在收集样本...")
sample_dict = get_unique_sample_names(folder_path)

print("🔄 正在划分样本...")
split_result = split_samples(sample_dict)

print("🚚 正在复制文件...")
copy_samples_to_folders(sample_dict, folder_path, split_result, target_base_dir)

print("✅ 完成！")
```



## 3.UFISH Finetune ####
```
cd UD2_finetune
```
#### 3.1 直接运行默认模型 ####
```
nohup ufish load_weights v1.0-alldata-ufish_c32.pth - predict_imgs ../FISH_spots/UD2_datasets/text predict --img_glob=\"*.tif\" --intensity_threshold 0.5 &
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
    ("./new-ud2/eval_UD2_text.csv", "Big-FISH"),
    ("./new-ud2/eval_UD2_text.csv", "RS-FISH"),
    ("./new-ud2/eval_UD2_text.csv", "Starfish"),
    ("./new-ud2/eval_UD2_text.csv", "TrackMate")
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





