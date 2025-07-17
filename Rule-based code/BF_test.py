#!/usr/bin/env python
# coding: utf-8

import os
from glob import glob
import time
import pandas as pd
import numpy as np
import bigfish.stack as stack
import bigfish.detection as detection
import bigfish
import argparse

print("Big-FISH version: {0}".format(bigfish.__version__), flush=True)



# 命令行参数解析
parser = argparse.ArgumentParser(description='批量处理图像并进行斑点检测')
parser.add_argument('input_path', help='输入图像所在的文件夹路径')
parser.add_argument('output_path', help='CSV结果输出文件夹路径')

args = parser.parse_args()

# 设置路径
path = args.input_path
csv_path = args.output_path

ims_path = path  # 如果 ims_path 和 path 是一致的，也可以直接用 path

#path = '/path/to/input/'
#path = '/home/jjyang/jupyter_file/test_speed/size512/test50/'
#ims_path = path

# path = '/path/to/output/
#csv_path = '/home/jjyang/jupyter_file/test_speed/predict_BF/'

## I've copied their code and changed it a bit to get all possible thresholds
def _get_breaking_point(x, y):
    """Select the x-axis value where a L-curve has a kink.
    Assuming a L-curve from A to B, the 'breaking_point' is the more distant
    point to the segment [A, B].
    Parameters
    ----------
    x : np.array, np.float64
        X-axis values.
    y : np.array, np.float64
        Y-axis values.
    Returns
    -------
    breaking_point : float
        X-axis value at the kink location.
    x : np.array, np.float64
        X-axis values.
    y : np.array, np.float64
        Y-axis values.
    """
    # select threshold where curve break
    slope = (y[-1] - y[0]) / len(y)
    y_grad = np.gradient(y)
    m = list(y_grad >= slope)
    j = m.index(False)
    m = m[j:]
    x = x[j:]
    y = y[j:]
    if True in m:
        i = m.index(True)
    else:
        i = -1
    breaking_point = float(x[i])

    return breaking_point, x, y

def _get_spot_counts(thresholds, value_spots):
    """Compute and format the spots count function for different thresholds.
    Parameters
    ----------
    thresholds : np.ndarray, np.float64
        Candidate threshold values.
    value_spots : np.ndarray
        Pixel intensity values of all spots.
    Returns
    -------
    count_spots : np.ndarray, np.float64
        Spots count function.
    """
    # count spots for each threshold
    count_spots = np.log([np.count_nonzero(value_spots > t)
                          for t in thresholds])
    count_spots = stack.centered_moving_average(count_spots, n=5)

    # the tail of the curve unnecessarily flatten the slop
    count_spots = count_spots[count_spots > 2]
    thresholds = thresholds[:count_spots.size]

    return thresholds, count_spots


def spots_thresholding(image, mask_local_max, threshold,
                       remove_duplicate=True):
    """Filter detected spots and get coordinates of the remaining spots.
    In order to make the thresholding robust, it should be applied to a
    filtered image (bigfish.stack.log_filter for example). If the local
    maximum is not unique (it can happen with connected pixels with the same
    value), connected component algorithm is applied to keep only one
    coordinate per spot.
    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    mask_local_max : np.ndarray, bool
        Mask with shape (z, y, x) or (y, x) indicating the local peaks.
    threshold : float or int
        A threshold to discriminate relevant spots from noisy blobs.
    remove_duplicate : bool
        Remove potential duplicate coordinates for the same spots. Slow the
        running.
    Returns
    -------
    spots : np.ndarray, np.int64
        Coordinate of the local peaks with shape (nb_peaks, 3) or
        (nb_peaks, 2) for 3-d or 2-d images respectively.
    mask : np.ndarray, bool
        Mask with shape (z, y, x) or (y, x) indicating the spots.
    """
    # check parameters
    stack.check_array(image,
                      ndim=[2, 3],
                      dtype=[np.uint8, np.uint16, np.float32, np.float64])
    stack.check_array(mask_local_max,
                      ndim=[2, 3],
                      dtype=[bool])
    stack.check_parameter(threshold=(float, int),
                          remove_duplicate=bool)

    # remove peak with a low intensity
    mask = (mask_local_max & (image > threshold))
    if mask.sum() == 0:
        spots = np.array([], dtype=np.int64).reshape((0, image.ndim))
        return spots, mask

    # make sure we detect only one coordinate per spot
    if remove_duplicate:
        # when several pixels are assigned to the same spot, keep the centroid
        cc = label(mask)
        local_max_regions = regionprops(cc)
        spots = []
        for local_max_region in local_max_regions:
            spot = np.array(local_max_region.centroid)
            spots.append(spot)
        spots = np.stack(spots).astype(np.int64)

        # built mask again
        mask = np.zeros_like(mask)
        mask[spots[:, 0], spots[:, 1]] = True

    else:
        # get peak coordinates
        spots = np.nonzero(mask)
        spots = np.column_stack(spots)

    return spots, mask

def _get_candidate_thresholds(pixel_values):
    """Choose the candidate thresholds to test for the spot detection.
    Parameters
    ----------
    pixel_values : np.ndarray
        Pixel intensity values of the image.
    Returns
    -------
    thresholds : np.ndarray, np.float64
        Candidate threshold values.
    """
    # choose appropriate thresholds candidate
    start_range = 0
    end_range = int(np.percentile(pixel_values, 99.9999))
    if end_range < 100:
        thresholds = np.linspace(start_range, end_range, num=100)
    else:
        thresholds = [i for i in range(start_range, end_range + 1)]
    thresholds = np.array(thresholds)

    return thresholds
def automated_threshold_setting(image, mask_local_max):
    """Automatically set the optimal threshold to detect spots.
    In order to make the thresholding robust, it should be applied to a
    filtered image (bigfish.stack.log_filter for example). The optimal
    threshold is selected based on the spots distribution. The latter should
    have a kink discriminating a fast decreasing stage from a more stable one
    (a plateau).
    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    mask_local_max : np.ndarray, bool
        Mask with shape (z, y, x) or (y, x) indicating the local peaks.
    Returns
    -------
    optimal_threshold : int
        Optimal threshold to discriminate spots from noisy blobs.
    """
    # check parameters
    stack.check_array(image,
                      ndim=[2, 3],
                      dtype=[np.uint8, np.uint16, np.float32, np.float64])
    stack.check_array(mask_local_max,
                      ndim=[2, 3],
                      dtype=[bool])

    # get threshold values we want to test
    thresholds = _get_candidate_thresholds(image.ravel())

    # get spots count and its logarithm
    first_threshold = float(thresholds[0])
    spots, mask_spots = spots_thresholding(
        image, mask_local_max, first_threshold, remove_duplicate=False)
    value_spots = image[mask_spots]
    thresholds, count_spots = _get_spot_counts(thresholds, value_spots)

    # select threshold where the kink of the distribution is located
    optimal_threshold, _, _ = _get_breaking_point(thresholds, count_spots)

    return thresholds.astype(float), optimal_threshold


def process_im(im, sigma, voxel_size_yx, psf_yx, im_path, gamma, alpha, beta):

    ## Filter image LoG:
    rna_log = stack.log_filter(im, sigma)

    ## Detect local maxima:
    mask = detection.local_maximum_detection(rna_log, min_distance=sigma)

    thr_range = [0, -6, -3, -2, -1, 1, 2, 3, 6]

    ## Find defualt threshold + all thresholds:
    all_thrs, default_thr = automated_threshold_setting(rna_log, mask)

    # Find thresholds to test around the default
    idx_default_thr = np.where(all_thrs == default_thr)[0][0]
    thrs_to_use_idxs = [it + idx_default_thr for it in thr_range if 0 < (it + idx_default_thr) < all_thrs.size]
    thrs_to_use = all_thrs[thrs_to_use_idxs]

    ## Iterate thresholds
    n_spots = []
    for threshold in thrs_to_use:

        time_tmp2 = time.time()

        ## Detect spots
        spots, _ = detection.spots_thresholding(rna_log, mask, threshold)

        thr_n_spots = spots.shape[0]

        conditions_str = (f'BF_{os.path.basename(im_path)}'
                            f'_sigyx{sigma[0]}thr{threshold:.3f}_alpha{alpha}_beta{beta}_gamma{gamma}')

        ### Dense region decomposition:
        spots_post_decomposition, dense_regions, reference_spot = detection.decompose_dense(
            im, spots, voxel_size_yx, psf_yx,
            alpha=alpha,
            beta=beta,
            gamma=gamma)
        if spots_post_decomposition.shape[0] < thr_n_spots:
            df = pd.DataFrame(data=spots, columns=["axis-0", "axis-1"])
            df_path = os.path.join(os.path.dirname(csv_path),
                                    (f'{conditions_str}_direct_spots.csv'))
            df.to_csv(df_path, index=False)
        else:
            df = pd.DataFrame(data=spots_post_decomposition, columns=["axis-0", "axis-1"])
            df_path = os.path.join(os.path.dirname(csv_path),
                                    (f'{conditions_str}_spots_decomposition.csv'))
            df.to_csv(df_path, index=False)
        time2 = int(round((time.time() - time_tmp2) * 1000))

        print(f'saving to {df_path}')
        print(f'running single image time: {time2}ms')


## Set parameters:
# General params for all test set. 
#general_params = { 
#        "voxel_size_yx":1,
#        #"sig_yx":1.0,
#        "psf_yx":1.0,  # psf_yx = sig_yx * voxel_size_yx         
#        "sigma": (1.0, 1.0),    # sigma = (sig_yx, sig_yx)
#        "alpha": 0.7999999999999999,
#        "beta": 1.2,
#        "gamma": 4
#}

general_params = { 
        "voxel_size_yx":1,
        #"sig_yx":1.5,
        "psf_yx":1.5,  # psf_yx = sig_yx * voxel_size_yx
        "sigma": (1.5, 1.5),    # sigma = (sig_yx, sig_yx)
        "alpha": 0.7999999999999999,
        "beta": 0.8,
        "gamma": 6.1
}

# Gets a list of image files to process:
image_files = glob(os.path.join(path, ims_path, "*.tif"))

time_tmp1 = time.time()
## Process images:
for im_path in image_files:

    im = stack.read_image(im_path)
    print(f'processing image: {im_path}', flush=True)

    dataset_name = os.path.basename(im_path).split("_")[0]

    #params = dataset_params.get(dataset_name,special_params)
    #params = special_params.get(dataset_name , special_params)

    process_im(im, im_path=im_path, **general_params)

time1 = int(round((time.time() - time_tmp1)*1000))
print(f'running all images time: {time1}ms')