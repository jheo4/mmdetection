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
data_json_file = 'mot_jsons/mot17_02_dpm.json'

print(Fore.YELLOW + f'model_json_file: {model_json_file}' + Style.RESET_ALL)
print(Fore.YELLOW + f'data_json_file: {data_json_file}' + Style.RESET_ALL)

############### 1. Load models ###############
print(Fore.GREEN + 'LOADING MODELS' + Style.RESET_ALL)
model_manager.add_model_cases_from_json(model_json_file)

############### 2. Set data paths  ###############
print(Fore.GREEN + 'LOAD IMG DATA and GT ANNOTATIONS' + Style.RESET_ALL)
# temp dir for intermediate results
tmp_dir = tempfile.TemporaryDirectory()
data_config = json.load(open(data_json_file))
gt_json = data_config['gt_json']
img_dir = data_config['img_dir']

print(f'\tgt_json: {gt_json}')
print(f'\timg_dir: {img_dir}')


############### 3. Load json gt annotations & images  ###############
coco_gt_loader.load_gts(gt_json)
first_img_id = coco_gt_loader.get_first_image_id()
print(f'\tfirst_img_id: {first_img_id}')
image_loader.load_images(img_dir, first_img_id)
# image_loader.print_loaded_images()

# coco_gt_loader.print_image_info()
# coco_gt_loader.print_annotation_info()
# coco_gt_loader.print_category_info()

############### 4. Inference and evaluation  ###############
print(Fore.GREEN + 'INFERENCE AND EVALUATION' + Style.RESET_ALL)
num_of_images = 1

for image_index in range(num_of_images):
    image_id = first_img_id + image_index
    img = image_loader.get_image(image_id)
    print(f'\timage_id: {image_id}, image_file: {image_loader.get_image_filepath(image_id)}')

    annotation_for_eval = coco_gt_loader.get_annotation_for_evaluation(image_id)
    print(f'\tannotation_for_eval: {annotation_for_eval}')

    for model_case in model_manager.models:
        print(f'\tmodel_case: {model_case}')
        model_context = model_manager.get_model(model_case)

        # inference
        result = inference_detector(model_context.model, img)
        result_to_eval = coco_utils.bbox_detection_to_eval_dict(result)
        result_to_eval['bboxes'] = result_to_eval['bboxes'][:12]
        result_to_eval['labels'] = result_to_eval['labels'][:12]
        result_to_eval['scores'] = result_to_eval['scores'][:12]

        # print(f'\tresult_to_eval["bboxes"][0]: {result_to_eval["bboxes"][0]}')

        # list to torch tensor
        result_to_eval['bboxes'][0] = torch.tensor([1359.1, 413.27, 120.26, 362.77])
        result_to_eval['bboxes'][1] = torch.tensor([571.03, 402.13, 104.56, 315.68])
        result_to_eval['bboxes'][2] = torch.tensor([650.8, 455.86, 63.98, 193.94])
        result_to_eval['bboxes'][3] = torch.tensor([721.23, 446.86, 41.871, 127.61])
        result_to_eval['bboxes'][4] = torch.tensor([454.06, 434.36, 97.492, 294.47])
        result_to_eval['bboxes'][5] = torch.tensor([1254.6, 446.72, 33.822, 103.47])
        result_to_eval['bboxes'][6] = torch.tensor([1301.1, 237.38, 195.98, 589.95])
        result_to_eval['bboxes'][7] = torch.tensor([1480.3, 413.27, 120.26, 362.77])
        result_to_eval['bboxes'][8] = torch.tensor([552.72, 473.9, 29.314, 89.943])
        result_to_eval['bboxes'][9] = torch.tensor([1097.0, 433.0, 39.0, 119.0])
        result_to_eval['bboxes'][10] = torch.tensor([543.19, 442.1, 44.948, 136.84])
        result_to_eval['bboxes'][11] = torch.tensor([1017.0, 425.0, 39.0, 119.0])


        print(f'\tlen(annotation_for_eval): {len(annotation_for_eval)}')
        print(f'\tlen(result_to_eval["bboxes"]): {len(result_to_eval["bboxes"])}')
        print(f'\tresult_to_eval["bboxes"]: {result_to_eval["bboxes"]}')

        for i in range(len(result_to_eval['labels'])):
            result_to_eval['labels'][i] = 0

        # Single image evaluation
        coco_metric = CocoMetric(ann_file=None,
                                 metric=['bbox'],
                                 classwise=False,
                                 outfile_prefix=f'{tmp_dir.name}/test', )

        coco_metric.dataset_meta = dict(classes=coco_gt_loader.get_category_names())
        coco_metric.process({},
            [dict(pred_instances=result_to_eval, img_id=1, ori_shape=(1080, 1920), instances=annotation_for_eval)])
        eval_results = coco_metric.evaluate(size=1)
        print(f'\teval_results: {eval_results}')
        model_context.evaluator.update_mAP(eval_results)

for model_case in model_manager.models:
    model_context = model_manager.get_model(model_case)
    model_context.evaluator.print_eval_result()

