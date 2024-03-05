import json
import utils.coco_utils as coco_utils
import os

mot_json_dir = "json/mot"

mot_jsons = os.listdir(mot_json_dir)
print(mot_jsons)

for mot_json in mot_jsons:
    mot_dataset_json = os.path.join(mot_json_dir, mot_json)
    print("Converting", mot_dataset_json)

    mot_dataset = json.load(open(mot_dataset_json, "r"))
    gt_file = mot_dataset["gt_file"]
    gt_json = mot_dataset["gt_json"]
    img_dir = mot_dataset["img_dir"]

    coco_utils.convert_gt_mot_to_coco(gt_file, img_dir, gt_json, True)

