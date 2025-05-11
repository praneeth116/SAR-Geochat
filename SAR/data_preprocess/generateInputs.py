import os
import glob
import cv2
import numpy as np
import pandas as pd
import math
import multiprocessing
from functools import partial
from tqdm  import tqdm

def mkdir_if_missing(d):
    if not os.path.exists(d):
        os.makedirs(d)

def generateInputs(scene_path, detections, slc_folder):

    scene_id = scene_path.split("/")[-1].split("_")[0]
    
    scene_detects = detections[detections["scene_id"] == scene_id]
    
    band_VH = cv2.imread(scene_path, 0)
    # scene_path_VV = scene_path.replace("VH", "VV")
    # band_VV = cv2.imread(scene_path_VV, 0)

    slc_path = os.path.join(slc_folder, "{}.png".format(scene_id))
    mask = cv2.imread(slc_path, 0) 
    if(mask.shape != band_VH.shape):
        mask = cv2.resize(mask, (band_VH.shape[1],band_VH.shape[0]))
    
    img_height = band_VH.shape[0] 
    img_width = band_VH.shape[1]  

    patch_size = 504
    stride = 504 - 50

    input_crops_folder = "../single_channel_rgb"
    mkdir_if_missing(input_crops_folder)

    for start_h in range(0, img_height-patch_size+stride, stride):
        for start_w in range(0, img_width-patch_size+stride, stride):

            cur_end_h = min(start_h+patch_size, img_height)
            cur_end_w = min(start_w+patch_size, img_width)
            cur_start_h = cur_end_h-patch_size
            cur_start_w = cur_end_w-patch_size

            band_VH_data = band_VH[cur_start_h:cur_end_h, cur_start_w:cur_end_w]
            # band_VV_data = band_VV[cur_start_h:cur_end_h, cur_start_w:cur_end_w]
            X = mask[cur_start_h:cur_end_h, cur_start_w:cur_end_w]
            threshold_value = 127 
            max_value = 255 
            _, threshold = cv2.threshold(band_VH_data, threshold_value, max_value, cv2.THRESH_BINARY)
            threshold_data = cv2.multiply(threshold, X)

            if np.sum(threshold_data) == 0:
                continue
            
            window_str = "{}_{}".format(cur_start_h, cur_start_w)
            img_name = window_str

            # channel_3  = cv2.divide(band_VV_data, band_VH_data)
            input_rgb_image = cv2.merge((band_VH_data,band_VH_data,band_VH_data))
            output_path = os.path.join(input_crops_folder,"{}_{}.png".format(scene_id, img_name))
            cv2.imwrite(output_path, input_rgb_image)
            
    return

def main(split="valid"):
    detect_file = "../data/raw_data/validation.csv"
    detections = pd.read_csv(detect_file, low_memory=False)
    scene_list = glob.glob("../data/{}_8bit/*VH.png".format(split))
    slc_folder = "../data/landMasks/valid"

    pool = multiprocessing.Pool(processes=2)
    with tqdm(total=len(scene_list), ncols=120) as t:
        for _ in pool.imap_unordered(partial(generateInputs, detections=detections, slc_folder=slc_folder), scene_list):
            t.update(1)

    # for scene_path in tqdm(scene_list):
    #     generateInputs(scene_path=scene_path, detections=detections, slc_folder=slc_folder)
    

if __name__ == "__main__":
    main()

# strings /usr/lib64/libstdc++.so.6 | grep GLIBCXX
# ldd /home/cvpr_ug_4/GeoChat/scripts/finetune.sh | grep libstdc++
