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


############### 2. Load data ###############
# temp dir for intermediate results
tmp_dir = tempfile.TemporaryDirectory()
data_config = json.load(open("dataset.json"))
gt_dir = data_config['gt_dir']
img_dir = data_config['data_dir']



