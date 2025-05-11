import os
import cv2
import numpy as np
import glob

def createMasks(npy_path , img_folder, masks_folder):
    scene_id  = npy_path.split("/")[-1].split("_")[0] 
    scene_path = os.path.join(img_folder, "{}_VH.png".format(scene_id))
    
    image = cv2.imread(scene_path, 0)
    height, width = image.shape

    data = np.load(npy_path, allow_pickle=True)

    image = np.ones(( width,height), dtype=np.uint8) * 255
    
    for i in data:
        polygon_points = np.array(i, dtype=np.int32)
        polygon_points = polygon_points.reshape((-1, 1, 2))
            
        cv2.fillPoly(image, [polygon_points], (0))

    image = cv2.flip(image, 0)
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    output_path = os.path.join(masks_folder, "{}.png".format(scene_id))
    cv2.imwrite(output_path, image)
    print(f"Processed and saved: {scene_id}")

def main():
    img_folder = "../data/valid_8bit"
    masks_folder = "../data/landMasks/valid"

    if not os.path.exists(masks_folder):
        os.makedirs(masks_folder)

    npy_folder = "../data/shoreline/validation"
    npy_files = glob.glob(f"{npy_folder}/*.npy")

    for npy_path in npy_files:
        createMasks(npy_path , img_folder, masks_folder)

if __name__ == "__main__":
    main()