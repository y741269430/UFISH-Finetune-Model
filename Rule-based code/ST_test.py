import os
from skimage import io
from starfish.core.spots.FindSpots import BlobDetector
import time
import argparse

# 命令行参数解析
parser = argparse.ArgumentParser(description='批量处理图像并进行斑点检测')
parser.add_argument('input_path', help='输入图像所在的文件夹路径')
parser.add_argument('output_path', help='CSV结果输出文件夹路径')

args = parser.parse_args()

# 设置路径
ims_path = args.input_path
out_path = args.output_path

#ims_path = '/home/jjyang/jupyter_file/test_speed/size512/test50/'
#out_path = '/home/jjyang/jupyter_file/test_speed/predict_ST/'

def process_im(im_path, out_path, sig, thr):
    im = io.imread(im_path)
    
    detector = BlobDetector(
        min_sigma=0.0,
        max_sigma=sig,
        num_sigma=sig,  
        threshold=thr,
        measurement_type='mean',   
        is_volume=False,
        exclude_border=False  
    )
            
    spots = detector.image_to_spots(im).spot_attrs

    df_path = os.path.join(out_path, f'ST_{os.path.basename(im_path)}_sig{sig}_thr{thr}.csv')

    df = spots.data
    df.rename(columns={'y': 'axis-0', 'x': 'axis-1'}, inplace=True)
    df1 = df[['axis-0', 'axis-1']] 

    df1.to_csv(df_path, index=False)
    print(f'Finished {df_path}', flush=True)

general_params = {
    "datasets": {
        "sig": 1,
        "thr": 0.001095  
    }
}

image_files = [os.path.join(ims_path, f) for f in os.listdir(ims_path) if f.endswith('.tif')]

start_time = time.time()
for im_path in image_files:
    print(f'Processing image: {im_path}', flush=True)
    params = general_params.get('datasets', general_params)
    process_im(im_path, out_path=out_path, **params)

total_time = int(round((time.time() - start_time) * 1000))
print(f'Total processing time: {total_time}ms')