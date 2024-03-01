import json
import utils.coco_utils as coco_utils

mot_dataset_json = "mot17_02_dpm.json"
mot_dataset_json = "mot17_02_frcnn.json"

mot_dataset = json.load(open(mot_dataset_json, "r"))
gt_file = mot_dataset["gt_file"]
gt_json = mot_dataset["gt_json"]
img_dir = mot_dataset["img_dir"]

coco_utils.convert_gt_mot_to_coco(gt_file, img_dir, gt_json)

# print("Ground truth file:", gt_file)
# print("Image directory:", img_dir)
