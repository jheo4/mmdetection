import torch
import os
import cv2
import json
from colorama import Fore, Style

def bbox_detection_to_dict(detection_result) :
    pred_instances = detection_result.get('pred_instances', None)
    labels = pred_instances.get('labels').cpu().numpy()
    scores = pred_instances.get('scores').cpu().numpy()
    bboxes = pred_instances.get('bboxes').cpu().numpy()
    return labels, scores, bboxes


def bbox_detection_to_eval_dict(detection_result):
    labels, scores, bboxes = bbox_detection_to_dict(detection_result)
    eval_dict = dict(bboxes = torch.from_numpy(bboxes),
                     labels = torch.from_numpy(labels),
                     scores = torch.from_numpy(scores))
    return eval_dict


def convert_gt_mot_to_coco(mot_text_path, img_dir, output_json_path, overwrite=False):
    # check if the output_json_path exists
    if os.path.exists(output_json_path) and not overwrite:
        print(Fore.RED + f"Output json file already exists: {output_json_path}" + Style.RESET_ALL)
        return
    elif os.path.exists(output_json_path) and overwrite:
        os.remove(output_json_path)

    out_coco_json = {'images': [], 'annotations': [], 'categories': []}

    # https://motchallenge.net/instructions/

    img_files = os.listdir(img_dir)
    img_files.sort()
    img_indices = [(img_file.split('.')[0]) for img_file in img_files]

    # coco's images
    image0 = cv2.imread(os.path.join(img_dir, img_files[0]))
    height, width, _ = image0.shape
    for img_id, img_file in zip(img_indices, img_files):
        out_coco_json['images'].append({'id': int(img_id), 'file_name': img_file, 'width': width, 'height': height})


    # coco's annotations -- as MOT detection does not identify categories, we will use category_id = 1
    with open(mot_text_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        '''
        https://arxiv.org/pdf/2010.07548.pdf page 29
        0: frame number
        1: identity number
        2: left
        3: top
        4: width
        5: height
        6: confidence score
        7: class
            1: pedestrian
            2: person on vehicle
            3: car
            4: bicycle
            5: motorbike
            6: non motorized vehicle
            7: static person
            8: distractor
            9: occluder
            10: occluder on the ground
            11: occluder full
            12: reflection
        8: visibility ratio
        '''
        parts = line.split(',')
        if float(parts[8]) < 0.5 or int(parts[7]) > 10:
            continue

        # Extract attributes
        frame = int(parts[0])
        id = int(parts[1])
        # coco shape is not (x, y, w, h) but (left top x, left top y, right bottom x, right bottom y)
        bb_left = float(parts[2])
        bb_top = float(parts[3])
        bb_width =  float(parts[4])
        bb_height = float(parts[5])
        bb_right = bb_left + bb_width
        bb_bottom = bb_top + bb_height

        conf = float(parts[6])
        annot = {'id': 0, 'image_id': frame, 'category_id': 0, 'bbox': [bb_left, bb_top, bb_right, bb_bottom]}

        out_coco_json['annotations'].append(annot)
        # print(f"frame: {frame}, id: {id}, bb_left: {bb_left}, bb_top: {bb_top}, bb_width: {bb_width}, bb_height: {bb_height}, conf: {conf}")

    # coco's categories
    out_coco_json['categories'].append({'id': 0, 'name': 'object'})

    # Save to json
    with open(output_json_path, 'w') as json_file:
        json.dump(out_coco_json, json_file, indent=4)


def filter_refined_result_by_score(refined_result, score_threshold):
    # result format: {'bboxes': tensor([[x1, y1, x2, y2], ...]), 'labels': tensor([label1, label2, ...]), 'scores': tensor([score1, score2, ...])}
    bboxes = refined_result['bboxes']
    labels = refined_result['labels']
    scores = refined_result['scores']

    mask = scores > score_threshold # mask: tensor([True, False, True, ...])

    refined_result['bboxes'] = bboxes[mask]
    refined_result['labels'] = labels[mask]
    refined_result['scores'] = scores[mask]

    return refined_result


from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
def filter_result_by_score(result, score_threshold):
    pred_instances = result.get('pred_instances')

    bboxes = pred_instances.get('bboxes')
    labels = pred_instances.get('labels')
    scores = pred_instances.get('scores')
    mask = scores > score_threshold # mask: tensor([True, False, True, ...])

    bboxes = bboxes[mask]
    labels = labels[mask]
    scores = scores[mask]

    filtered_instances = InstanceData()
    filtered_instances.bboxes = bboxes
    filtered_instances.labels = labels
    filtered_instances.scores = scores

    result.pred_instances = filtered_instances

    return result


def filter_result_by_categories(result, category_ids):
    pred_instances = result.get('pred_instances')

    bboxes = pred_instances.get('bboxes')
    labels = pred_instances.get('labels')
    scores = pred_instances.get('scores')

    mask = torch.zeros_like(labels, dtype=torch.bool)
    for category_id in category_ids:
        mask = mask | (labels == category_id) # mask: tensor([True, False, True, ...])

    bboxes = bboxes[mask]
    labels = labels[mask]
    scores = scores[mask]

    filtered_instances = InstanceData()
    filtered_instances.bboxes = bboxes
    filtered_instances.labels = labels
    filtered_instances.scores = scores

    result.pred_instances = filtered_instances

    return result


if __name__ == "__main__":
    detection_result = {
        'pred_instances': {
            'labels': torch.tensor([1, 2, 3]),
            'scores': torch.tensor([0.9, 0.8, 0.7]),
            'bboxes': torch.tensor([[0, 0, 10, 10], [10, 10, 20, 20], [20, 20, 30, 30]])
        }
    }
    print(bbox_detection_to_dict(detection_result))
    print(bbox_detection_to_eval_dict(detection_result))

