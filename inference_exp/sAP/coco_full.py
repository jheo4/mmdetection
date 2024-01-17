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
from mmdet.apis import init_detector, inference_detector # for inference
from mmdet.evaluation import CocoMetric # for coco metric evaluation
from mmengine.fileio import dump
from mmdet.registry import VISUALIZERS


#  My modules
import coco_utils
from model_manager import Model_Manager
model_manager = Model_Manager()

############### 1. Load models ###############
model_configs = json.load(open("model_configs.json"))
model_dir = model_configs['model_dir']
model_specs = model_configs['models']
print(f'model_dir: {model_dir}')
# print(f'model_specs: {model_specs}')
print('='*120)

for model_spec in model_specs:
    model_case = model_spec['case']
    model_cfg = model_dir + model_spec['cfg']
    model_weights = model_dir + model_spec['weight']
    print(f'model_case: {model_case}')
    print(f'\tmodel_cfg: {model_cfg}')
    print(f'\tmodel_weights: {model_weights}')

    model_manager.add_model(model_case, model_cfg, model_weights, device='cuda:0')
print('='*120)

model_manager.print_all_models()


############### 2. Set data paths  ###############
# temp dir for intermediate results
tmp_dir = tempfile.TemporaryDirectory()
data_config = json.load(open("dataset.json"))
gt_dir = data_config['gt_dir']
img_dir = data_config['img_dir']

img_files = os.listdir(img_dir)
img_files.sort()
print(f'img_files: {img_files}') # img_files: ['000001.jpg', ... ]

############### 3. Load json gt annotations  ###############
gt_json_file = gt_dir + 'annotation.json'
gt_json = json.load(open(gt_dir + 'annotation.json'))
gt_json_categories = gt_json['categories']
gt_json_annotations = gt_json['annotations']
category_by_name = []

for category in gt_json_categories:
    category_by_name.append(category['name'])

annotation_instances = []
for annotation in gt_json_annotations:
    print(f'\tannotation: {annotation}')
    annotation_instance = dict(bbox_label = annotation['category_id'],bbox = annotation['bbox'])
    annotation_instances.append(annotation_instance)

print(f'\tannotation_instances: {annotation_instances}')

############### 4. Inference and evaluation  ###############
for img_file in img_files:
    img_path = img_dir + img_file
    img = cv2.imread(img_path)
    print(f'\timg_path: {img_path}')

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

        # print shape


        print(result_to_eval['bboxes'])
        print(result_to_eval['labels'])
        print(result_to_eval['scores'])

        print(gt_json['annotations'])
        #print("="*120)
        #print(f'\tresult_to_eval: {result_to_eval}')
        #print("="*120)

        coco_metric = CocoMetric(ann_file=None,
                                 metric=['bbox'],
                                 classwise=False,
                                 outfile_prefix=f'{tmp_dir.name}/test')

        coco_metric.dataset_meta = dict(classes=category_by_name)
        coco_metric.process({},
            [dict(pred_instances=result_to_eval, img_id=0, ori_shape=(427, 640), instances=annotation_instances)])
        eval_results = coco_metric.evaluate(size=1)
        print(f'eval_results: {eval_results}')

