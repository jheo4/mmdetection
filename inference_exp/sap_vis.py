# Primitive dependencies
import copy
import os
import tempfile
import torch
import numpy as np
import pycocotools.mask as mask_util
import cv2
import json
from colorama import Fore, Style

# MMdetection
import mmdet
from mmdet.apis import init_detector, inference_detector # for inference
from mmdet.evaluation import CocoMetric # for coco metric evaluation
from mmengine.fileio import dump
from mmdet.registry import VISUALIZERS

#  My modules
import utils.coco_utils as coco_utils
from utils.model_manager import Model_Manager
from utils.coco_gt_loader import COCO_GT_Loader # COCO BBox format: [x1, y1, w, h] left-top corner, width, height
from utils.image_loader import Image_Loader
from utils.coco_evaluator import COCO_Evaluator

from utils.my_visualizer import My_Visualizer

model_manager = Model_Manager()
image_loader = Image_Loader()
coco_gt_loader = COCO_GT_Loader()

model_json_file = 'json/model_configs.json'
data_json_file = 'json/mot/dpm/mot17_13.json'

############### 1. Load models ###############
model_manager.add_model_cases_from_json(model_json_file)

############### 2. Set data paths  ###############
tmp_dir = tempfile.TemporaryDirectory()
data_config = json.load(open(data_json_file))
gt_json = data_config['gt_json']
img_dir = data_config['img_dir']

############### 3. Load json gt annotations & images  ###############
coco_gt_loader.load_gts(gt_json)
first_img_id = coco_gt_loader.get_first_image_id()
image_loader.load_images(img_dir, first_img_id, 500)

# coco_gt_loader.print_image_info()
# coco_gt_loader.print_annotation_info()
# coco_gt_loader.print_category_info()

############### 4. Inference and evaluation  ###############
print(Fore.GREEN + 'INFERENCE AND EVALUATION' + Style.RESET_ALL)
num_of_images = 1

image_id = 15
gt_offset = 10

pred_img = image_loader.get_image(image_id)
gt_img1 = image_loader.get_image(image_id + gt_offset)
gt_img2 = copy.deepcopy(gt_img1)

annotation_for_eval = coco_gt_loader.get_annotation_for_evaluation(image_id + gt_offset)
annotation_for_eval2 = coco_gt_loader.get_annotation_for_evaluation(image_id)

annot_bboxes = []
for annot in annotation_for_eval:
    annot_bboxes.append(annot['bbox'])

annot_bboxes2 = []
for annot in annotation_for_eval2:
    annot_bboxes2.append(annot['bbox'])

my_visualizer = My_Visualizer()
bboxes_img1 = my_visualizer.draw_bboxes(gt_img1, bboxes1=annot_bboxes, color1=(0, 255, 0), thickness1=4)
my_visualizer = My_Visualizer()
bboxes_img2 = my_visualizer.draw_bboxes(gt_img2, bboxes1=annot_bboxes2, color1=(0, 0, 255), thickness1=4)
cv2.imwrite(f'bboxes_img.jpg', bboxes_img1)
cv2.imwrite(f'bboxes_img1.jpg', bboxes_img2)
exit(0)

print(Fore.RED + f'\tannotation_for_eval: {annot_bboxes}' + Style.RESET_ALL)

for model_case in model_manager.models:
    model_context = model_manager.get_model(model_case)
    result = inference_detector(model_context.model, pred_img)
    result = coco_utils.filter_result_by_score(result, 0.3) # filter prediction results by confidence score
    tep_res = coco_utils.filter_result_by_categories(result, [0, 1, 2, 3, 4, 5, 6, 7, 8])
    result_to_eval = coco_utils.bbox_detection_to_eval_dict(result)

    result_bboxes_tensor = result_to_eval['bboxes']
    result_bboxes = result_bboxes_tensor.tolist()
    print(Fore.BLUE + f'\tresult_bboxes: {result_bboxes}' + Style.RESET_ALL)

    bboxes_img = my_visualizer.draw_bboxes(gt_img, bboxes1=annot_bboxes, color1=(0, 255, 0), thickness1=2, bboxes2=result_bboxes, color2=(0, 0, 255), thickness2=2)
    cv2.imwrite(f'bboxes_img_{model_case}.jpg', bboxes_img)

