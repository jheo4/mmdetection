# Primitive dependencies
import copy
import os
import tempfile
import torch
import numpy as np
import pycocotools.mask as mask_util
import cv2
import json

# MMdetection
import mmdet
from mmdet.apis import init_detector, inference_detector # for inference
from mmdet.evaluation import CocoMetric # for coco metric evaluation
from mmengine.fileio import dump
from mmdet.registry import VISUALIZERS

from colorama import Fore, Style


#  My modules
import coco_utils
from model_manager import Model_Manager

# COCO BBox format: [x1, y1, w, h] left-top corner, width, height

model_manager = Model_Manager()

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

    model_manager.add_model(model_case, model_cfg, model_weights, device='cuda:0')

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

img_files = os.listdir(img_dir)
img_files.sort()
print(f'\timg_files: {img_files}') # img_files: ['000001.jpg', ... ]

############### 3. Load json gt annotations  ###############
gt_json_file = gt_dir + 'annotation.json'
gt_json = json.load(open(gt_dir + 'annotation.json'))
gt_json_categories = gt_json['categories']
gt_json_annotations = gt_json['annotations']
category_by_name = []

for category in gt_json_categories:
    category_by_name.append(category['name'])


annotation_instances = []
annots = []
image_id = 0
for annotation in gt_json_annotations:
    print(f'\tannotation: {annotation}')
    if annotation['image_id'] != image_id:
        image_id = annotation['image_id']
        annotation_instances.append(annots)
        annots = []
    annot = dict(image_id = annotation['image_id'], bbox_label = annotation['category_id'],bbox = annotation['bbox'])
    annots.append(annot)

if len(annots) > 0:
    annotation_instances.append(annots)

print(f'\tannotation_instances: {annotation_instances}')


############### 4. Inference and evaluation  ###############
print(Fore.GREEN + 'INFERENCE AND EVALUATION' + Style.RESET_ALL)

res = []

for img_file, img_idx in zip(img_files, range(len(img_files))):
    img_path = img_dir + img_file
    img = cv2.imread(img_path)
    print(f'\timg: {img_path}')

    # iterate all models
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

        res.append(result_to_eval)

        # print shape
        print('\tbboxes: ', result_to_eval['bboxes'])
        print('\tlabels: ', result_to_eval['labels'])
        print('\tscores: ', result_to_eval['scores'])

        # Single image evaluation
        coco_metric = CocoMetric(ann_file=None,
                                 metric=['bbox'],
                                 classwise=False,
                                 outfile_prefix=f'{tmp_dir.name}/test')

        coco_metric.dataset_meta = dict(classes=category_by_name)
        coco_metric.process({},
            [dict(pred_instances=result_to_eval, img_id=0, ori_shape=(427, 640), instances=annotation_instances[img_idx])])
        eval_results = coco_metric.evaluate(size=2)
        print(f'eval_results: {eval_results}')


# evaluation test
# res, annotation_instances
coco_metric_full = CocoMetric(ann_file=None, metric=['bbox'], classwise=False, outfile_prefix=f'{tmp_dir.name}/test')
coco_metric_full.dataset_meta = dict(classes=category_by_name)
coco_metric_full.process({}, [dict(pred_instances=res, img_id=0, ori_shape=(427, 640), instances=annotation_instances[0])])

