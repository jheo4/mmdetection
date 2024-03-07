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
from utils.sap_engine import sAP_Engine

sap_engine = sAP_Engine()

# Model Manager Setting
tmp_dir = tempfile.TemporaryDirectory()
model_manager = Model_Manager()
model_json_file = 'json/model_configs.json'
print(Fore.GREEN + '============================= LOADING MODELS ===========================' + Style.RESET_ALL)
model_manager.add_model_cases_from_json(model_json_file)


# Data Setting
mot_json_dir = "json/mot"
# detectors = ["dpm", "frcnn", "sdp"]
detectors = ["dpm"]
vid_ids = ["13", "04"]
mot_json_files = []
for detector in detectors:
    for vid_id in vid_ids:
        mt_json_file = f"{mot_json_dir}/{detector}/mot17_{vid_id}.json"
        mot_json_files.append(mt_json_file)


data_interval = 33 # 33 ms = 30 fps
emulated_inference_intervals = [33 * i for i in range(10)] # 33ms, 66ms, 99ms, 132ms, 165ms, 198ms, 231ms, 264ms, 297ms
evaluators_for_diff_intervals = [COCO_Evaluator() for _ in range(len(emulated_inference_intervals))]
num_of_images = 100
conf_thr = 0.3


dataset_results = []

for dataset_json in mot_json_files:
    dataset_result = {}

    print(Fore.GREEN + f'============================= {dataset_json} ===========================' + Style.RESET_ALL)
    image_loader = Image_Loader()
    coco_gt_loader = COCO_GT_Loader()

    print(Fore.YELLOW + f'\t# Load Imgs & GT Annotations' + Style.RESET_ALL)

    dataset_config = json.load(open(dataset_json))
    gt_json = dataset_config['gt_json']
    img_dir = dataset_config['img_dir']

    coco_gt_loader.load_gts(gt_json, end_id=num_of_images)
    coco_gt_loader.print_loaded_gt_brief()

    first_img_id = coco_gt_loader.get_first_image_id()
    print(f'\tfirst_img_id: {first_img_id}')
    image_loader.load_images(img_dir, first_img_id, num_of_images)
    image_loader.print_loaded_images_brief()

    # for all models cases
    for model_case in model_manager.models:
        dataset_result[model_case] = {}

        print(f'\tmodel_case: {model_case}')
        model_context = model_manager.get_model(model_case)

        # for all images in each dataset
        for image_index in range(num_of_images):
            image_id = first_img_id + image_index
            img = image_loader.get_image(image_id)
            print(f'\timage_id: {image_id}, image_file: {image_loader.get_image_filepath(image_id)}')

            # annots corresponding the current image
            sap_engine.current_index = image_id
            annots = []
            for inf_interval in emulated_inference_intervals:
                annot_index, jump = sap_engine.get_index_and_jump(inf_interval, data_interval)
                annot_index += jump
                if annot_index < num_of_images+1:
                    # print(f'\timage_id: {image_id}, annot_index: {annot_index}')
                    annotation_for_eval = coco_gt_loader.get_annotation_for_evaluation(annot_index)
                    annots.append(annotation_for_eval)

            # inference
            result = inference_detector(model_context.model, img)
            result = coco_utils.filter_result_by_score(result, conf_thr) # filter prediction results by confidence score
            result = coco_utils.filter_result_by_categories(result, [0, 1, 2, 3, 4, 5, 6, 7, 8]) # filter by categories
            result_to_eval = coco_utils.bbox_detection_to_eval_dict(result)
            result_to_eval['labels'] = torch.zeros((len(result_to_eval['bboxes']),), dtype=torch.int64) # MOT does not identify the class of the object
            # print(Fore.RED + f'\tresult_to_eval: {result_to_eval}' + Style.RESET_ALL)

            '''
            visualizer_cfg = dict(name='visualizer',
                                  type='DetLocalVisualizer',
                                  vis_backends=[dict(type='LocalVisBackend'),])
            visualizer = VISUALIZERS.build(visualizer_cfg)
            visualizer.dataset_meta = model_context.model.dataset_meta
            visualizer.add_datasample(name='result', image=img[:, :, ::-1], data_sample=result,
                                      draw_gt=False, pred_score_thr=conf_thr, show=True)
            '''

            for annot, annot_index in zip(annots, range(len(annots))):
                coco_metric = CocoMetric(ann_file=None, metric=['bbox'], classwise=False, outfile_prefix=f'{tmp_dir.name}/test')
                coco_metric.dataset_meta = dict(classes=coco_gt_loader.get_category_names())
                coco_metric.process({}, [dict(pred_instances=result_to_eval, img_id=1, ori_shape=(1080, 1920), instances=annot)])
                eval_results = coco_metric.evaluate(size=1)

                print(f'annot_index: {annot_index}, eval_results: {eval_results}')

                evaluators_for_diff_intervals[annot_index].update_mAP(eval_results)
                # print(f'\teval_results: {eval_results}')

            for evaluator, eval_index in zip(evaluators_for_diff_intervals, range(len(evaluators_for_diff_intervals))):
                res = {'mAP': round(evaluator.mAP, 3), 'mAP_50': round(evaluator.mAP_50, 3), 'mAP_75': round(evaluator.mAP_75, 3)}
                dataset_result[model_case][eval_index] = res
                # print(Fore.BLUE + f'eval_results: {round(evaluator.mAP_50, 3)}' + Style.RESET_ALL)

        dataset_results.append(dataset_result)


print(Fore.GREEN + f'Dataset Results: {dataset_results}' + Style.RESET_ALL)
for dataset_result in dataset_results:
    for key, value in dataset_result.items():
        print(f'model_case: {key}')
        first = -1
        for diff_frame, metric_result in value.items():
            print(f'\t{diff_frame} frames diff')
            if first == -1:
                map_first = metric_result['mAP']
                map_50_first = metric_result['mAP_50']
                map_75_first = metric_result['mAP_75']
                first = 0
            mAP = metric_result['mAP']
            mAP_50 = metric_result['mAP_50']
            mAP_75 = metric_result['mAP_75']

            if map_first == 0:
                norm_mAP = 0
            else:
                norm_mAP = round(mAP / map_first, 3)

            if map_50_first == 0:
                norm_mAP_50 = 0
            else:
                norm_mAP_50 = round(mAP_50 / map_50_first, 3)

            if map_75_first == 0:
                norm_mAP_75 = 0
            else:
                norm_mAP_75 = round(mAP_75 / map_75_first, 3)


            print(f'\t\tmAP: {mAP}, {norm_mAP}')
            print(f'\t\tmAP_50: {mAP_50}, {norm_mAP_50}')
            print(f'\t\tmAP_75: {mAP_75}, {norm_mAP_75}')

