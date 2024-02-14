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


model_manager = Model_Manager()
image_loader = Image_Loader()
coco_gt_loader = COCO_GT_Loader()

model_json_file = 'model_configs.json'
print(Fore.YELLOW + f'model_json_file: {model_json_file}' + Style.RESET_ALL)

data_json_file = 'dataset.json'
print(Fore.YELLOW + f'data_json_file: {data_json_file}' + Style.RESET_ALL)

############### 1. Load models ###############
print(Fore.GREEN + 'LOADING MODELS' + Style.RESET_ALL)
model_configs = json.load(open(model_json_file))
model_dir = model_configs['model_dir']
model_specs = model_configs['models']
print(f'\tmodel_dir: {model_dir}')

for model_spec in model_specs:
    model_case = model_spec['case']
    model_cfg = model_dir + model_spec['cfg']
    model_weights = model_dir + model_spec['weight']
    print(f'\tmodel_case: {model_case}')
    print(f'\t\tmodel_cfg: {model_cfg}')
    print(f'\t\tmodel_weights: {model_weights}')

    model_manager.add_model(model_case, model_cfg, model_weights, device='cuda:0', evaluator=COCO_Evaluator())

print(Fore.GREEN + 'LOADED MODELS' + Style.RESET_ALL)
model_manager.print_all_models()


############### 2. Set data paths  ###############
print(Fore.GREEN + 'LOAD IMG DATA and GT ANNOTATIONS' + Style.RESET_ALL)
# temp dir for intermediate results
tmp_dir = tempfile.TemporaryDirectory()
data_config = json.load(open(data_json_file))
gt_dir = data_config['gt_dir']
img_dir = data_config['img_dir']
print(f'\tgt_dir: {gt_dir}')
print(f'\timg_dir: {img_dir}')

image_loader.load_images(img_dir)
image_loader.print_loaded_images()


############### 3. Load json gt annotations  ###############
gt_json_file = gt_dir + 'annotation.json'
coco_gt_loader.load_gts(gt_json_file)
coco_gt_loader.print_image_info()
coco_gt_loader.print_annotation_info()

############### 4. Inference and evaluation  ###############
print(Fore.GREEN + 'INFERENCE AND EVALUATION' + Style.RESET_ALL)
images = 2

for image_index in range(images):
    img = image_loader.get_image(image_index)
    annotation_for_eval = coco_gt_loader.get_annotation_for_evaluation(image_index)

    for model_case in model_manager.models:
        print(f'\tmodel_case: {model_case}')
        model_context = model_manager.get_model(model_case)

        # inference
        result = inference_detector(model_context.model, img)
        result_to_eval = coco_utils.bbox_detection_to_eval_dict(result)

        # take remove all other torch tensor elements except the first one
        result_to_eval['bboxes'] = result_to_eval['bboxes'][:2]
        result_to_eval['labels'] = result_to_eval['labels'][:2]
        result_to_eval['scores'] = result_to_eval['scores'][:2]

        # print shape
        # print('\tbboxes: ', result_to_eval['bboxes'])
        # print('\tlabels: ', result_to_eval['labels'])
        # print('\tscores: ', result_to_eval['scores'])

        # Single image evaluation
        coco_metric = CocoMetric(ann_file=None,
                                 metric=['bbox'],
                                 classwise=False,
                                 outfile_prefix=f'{tmp_dir.name}/test', )

        coco_metric.dataset_meta = dict(classes=coco_gt_loader.get_category_names())
        coco_metric.process({},
            [dict(pred_instances=result_to_eval, img_id=0, ori_shape=(427, 640), instances=annotation_for_eval)])
        eval_results = coco_metric.evaluate(size=1)
        model_context.evaluator.update_mAP(eval_results)

for model_case in model_manager.models:
    model_context = model_manager.get_model(model_case)
    model_context.evaluator.print_eval_result()

