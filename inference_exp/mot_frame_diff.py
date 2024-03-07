# Primitive dependencies
import os
import json
import numpy as np
from colorama import Fore, Style

#  My modules
import utils.coco_utils as coco_utils
from utils.image_loader import Image_Loader
from utils.frame_diff_calculator import Frame_Diff_Calculator


mot_json_dir = "json/mot/dpm"
mot_jsons = os.listdir(mot_json_dir)
print(mot_jsons)

frames = 50

for mot_json in mot_jsons:
    print(f'Processing {mot_json}')
    mot_dataset_json = os.path.join(mot_json_dir, mot_json)
    mot_dataset = json.load(open(mot_dataset_json, "r"))
    img_dir = mot_dataset["img_dir"]

    image_loader = Image_Loader()
    image_loader.load_images(img_dir, 1, frames)
    # image_loader.print_loaded_images()

    frame_diff_calculator = Frame_Diff_Calculator()
    frame_diff_calculator.most_recent_frame = image_loader.get_image(1)

    for i in range(2, frames+1):
        frame = image_loader.get_image(i)
        frame_diff_calculator.calcualte_pixel_difference(frame)
        frame_diff_calculator.calculate_histogram_difference(frame)
        frame_diff_calculator.calculate_ssim(frame)

        frame_diff_calculator.most_recent_frame = frame

    avg_pixel_diff = round(np.mean(frame_diff_calculator.pixel_diffs), 3)
    avg_hist_diff = round(np.mean(frame_diff_calculator.hist_diffs), 3)
    avg_ssim = round(np.mean(frame_diff_calculator.ssims), 3)

    print(f'avg_pixel_diff: {avg_pixel_diff}, avg_hist_diff: {avg_hist_diff}, avg_ssim: {avg_ssim}')

