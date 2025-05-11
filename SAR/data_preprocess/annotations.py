import cv2
import json
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import glob

def generateAnnotations(scene_path, detections, slc_folder):

    scene_id = scene_path.split("/")[-1].split("_")[0]
    scene_detects = detections[detections["scene_id"] == scene_id]
    
    band_VH = cv2.imread(scene_path, 0)

    slc_path = os.path.join(slc_folder, "{}.png".format(scene_id))
    mask = cv2.imread(slc_path, 0) 
    if(mask.shape != band_VH.shape):
        mask = cv2.resize(mask, (band_VH.shape[1], band_VH.shape[0]))

    img_height = band_VH.shape[0] 
    img_width = band_VH.shape[1]  
    
    patch_size = 504
    stride = 504 - 50

    mask_fish = np.zeros((img_height, img_width))

    for i in range(len(scene_detects)):
        row = scene_detects.iloc[i]
        detect_row = row["detect_scene_row"]
        detect_col = row["detect_scene_column"]
        confidence  = row["confidence"]
        is_vessel = row["is_vessel"]
        is_fishing = row["is_fishing"]
        if confidence != "LOW":
            if is_vessel:
                if is_fishing:
                    mask_fish[detect_row, detect_col] = 1
                else:
                    mask_fish[detect_row, detect_col] = 2
            else:
                mask_fish[detect_row, detect_col] = 3
    

    annotations = []
    for start_h in range(0, img_height-patch_size+stride, stride):
        for start_w in range(0, img_width-patch_size+stride, stride):

            cur_end_h = min(start_h+patch_size, img_height)
            cur_end_w = min(start_w+patch_size, img_width)
            cur_start_h = cur_end_h-patch_size
            cur_start_w = cur_end_w-patch_size

            band_VH_data = band_VH[cur_start_h:cur_end_h, cur_start_w:cur_end_w]

            X = mask[cur_start_h:cur_end_h, cur_start_w:cur_end_w]
            threshold_value = 127 
            max_value = 255 
            _, threshold = cv2.threshold(band_VH_data, threshold_value, max_value, cv2.THRESH_BINARY)
            threshold_data = cv2.multiply(threshold, X)

            if np.sum(threshold_data) == 0:
                continue
            
            window_str = "{}_{}".format(cur_start_h, cur_start_w)
            img_name = window_str

            mask_fish_data = mask_fish[cur_start_h:cur_end_h, cur_start_w:cur_end_w]
            vessels_fish_list = np.argwhere(mask_fish_data == 1)
            vessels_no_fish_list = np.argwhere(mask_fish_data == 2)
            no_vessels_list = np.argwhere(mask_fish_data == 3)
           
            vessels_fish_list = ''.join(['{<%d><%d>}' % (x[0], x[1]) for x in vessels_fish_list])
            vessels_no_fish_list = ''.join(['{<%d><%d>}' % (x[0], x[1]) for x in vessels_no_fish_list])
            no_vessels_list = ''.join(['{<%d><%d>}' % (x[0], x[1]) for x in no_vessels_list])
            
            annotation = {
                "id": f"{scene_id}_{img_name}",
                "image": f"{scene_id}_{img_name}.png",
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\n Detect and list the centers of all fishing vessels, non-fishing vessels, and non-vessels ."
                    },
                    {
                        "from": "gpt",
                        "value": {
                            "fishing_vessels": vessels_fish_list,
                            "non_fishing_vessels": vessels_no_fish_list,
                            "non_vessels": no_vessels_list
                        }
                    }
                ]
            }
            

            annotations.append(annotation)

    return annotations


def process_all_scenes(detections, slc_folder, img_folder, output_file="annotations.json"):
    annotations = []
    
    for scene_path in tqdm(glob.glob(os.path.join(img_folder, '*_VH.png'))):
        scene_annotations = generateAnnotations(scene_path, detections, slc_folder)
        annotations.extend(scene_annotations)

    with open(output_file, "w") as f:
        json.dump(annotations, f, indent=4)
        
    print(f"Annotations saved to {output_file}")


if __name__ == "__main__":
    csv_path = "../data/raw_data/validation.csv"
    img_folder = "../data/valid_8bit/"
    detections = pd.read_csv(csv_path)
    slc_folder = "../data/landMasks/valid"
    process_all_scenes(detections=detections, slc_folder=slc_folder, img_folder=img_folder)
