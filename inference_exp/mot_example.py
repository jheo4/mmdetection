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

model_json_file = 'json/model_configs.json'
data_json_file = 'json/mot/dpm/mot17_02.json'

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
    print(Fore.YELLOW + f'\tannotation_for_eval: {annotation_for_eval}' + Style.RESET_ALL)

    for model_case in model_manager.models:
        print(f'\tmodel_case: {model_case}')
        model_context = model_manager.get_model(model_case)

        # inference
        result = inference_detector(model_context.model, img)
        result = coco_utils.filter_result_by_score(result, 0.3) # filter prediction results by confidence score

        tep_res = coco_utils.filter_result_by_categories(result, [0, 1, 2, 3, 4, 5, 6, 7, 8])
        print(Fore.RED + f'\ttep_res: {tep_res}' + Style.RESET_ALL)

        print("Result to Vis################################################")
        print(result)

        result_to_eval = coco_utils.bbox_detection_to_eval_dict(result)
        print(Fore.RED + f'\tresult_to_eval: {result_to_eval}' + Style.RESET_ALL)

        # tensor array with zeros
        result_to_eval['labels'] = torch.zeros((len(result_to_eval['bboxes']),), dtype=torch.int64)
        print(Fore.RED + f'\tresult_to_eval: {result_to_eval}' + Style.RESET_ALL)


        visualizer_cfg = dict(name='visualizer',
                              type='DetLocalVisualizer',
                              vis_backends=[dict(type='LocalVisBackend'),])
        visualizer = VISUALIZERS.build(visualizer_cfg)
        visualizer.dataset_meta = model_context.model.dataset_meta


        visualizer.add_datasample(name='result',
                                  image=img[:, :, ::-1],
                                  data_sample=result,
                                  draw_gt=False,
                                  pred_score_thr=0.3,
                                  show=True)
        # image_with_result = visualizer.get_image()
        # rgb to bgr
        # image_with_result = image_with_result[:, :, ::-1]
        # cv2.imshow('result', image_with_result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()



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

