import copy
import os
import tempfile

import torch
import numpy as np
import pycocotools.mask as mask_util

from mmdet.evaluation import CocoMetric
from mmengine.fileio import dump

# Temp result dir
tmp_dir = tempfile.TemporaryDirectory()
print(f'tmp_dir.name = {tmp_dir.name}')

# Prepare mock data
## create_dummy_coco_json
mock_json_file = os.path.join(tmp_dir.name, "mock.json")
dummy_mask = np.zeros((10, 10), order='F', dtype=np.uint8)
dummy_mask[:5, :5] = 1
rle_mask = mask_util.encode(dummy_mask)
rle_mask['counts'] = rle_mask['counts'].decode('utf-8')

# https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/md-coco-overview.html
image = {
        'id': 0,
        'width': 640,
        'height': 640,
        'file_name': 'fake_name.jpg',
        }
annotation_1 = {
        'image_id': 0,
        'category_id': 0,
        'bbox': [50, 60, 20, 20],
        }
annotation_2 = {
        'id': 2,
        'image_id': 0,
        'category_id': 0,
        'area': 900,
        'bbox': [100, 120, 30, 30],
        'iscrowd': 0,
        'segmentation': None,
        }
annotation_3 = {
        'image_id': 0,
        'category_id': 1,
        'bbox': [150, 160, 40, 40],
        }
annotation_4 = {
        'image_id': 0,
        'category_id': 0,
        'bbox': [250, 260, 100, 100],
        }
categories = [
        {
            'id': 0,
            'name': 'car',
            'supercategory': 'car',
            },
        {
            'id': 1,
            'name': 'bicycle',
            'supercategory': 'bicycle',
            },
        ]
mock_json = {
        'images': [image],
        'annotations': [annotation_1, annotation_2, annotation_3, annotation_4],
        'categories': categories,
        }
dump(mock_json, mock_json_file)

## create_dummy_results
bboxes = np.array([[50, 60, 70, 70], [100, 120, 130, 150],[150, 160, 190, 200], [250, 260, 350, 360]])
scores = np.array([1.0, 0.98, 0.96, 0.95])
labels = np.array([0, 0, 1, 0])
dummy_mask = np.zeros((4, 10, 10), dtype=np.uint8)
dummy_mask[:, :5, :5] = 1

mock_predictions = dict(bboxes = torch.from_numpy(bboxes),
                        scores = torch.from_numpy(scores),
                        labels = torch.from_numpy(labels),
                        masks = torch.from_numpy(dummy_mask))



# Test with json
## single coco dataset evaluation


## box and segm dataset evaluation

## invalid custom metric_item

## custom metric_item


# Test wihtout json
# instances in a image idx(0)
instances = [{'bbox_label': 0, 'bbox': [50, 60, 70, 80]},
             {'bbox_label': 0, 'bbox': [100, 120, 130, 150]},
             {'bbox_label': 1, 'bbox': [150, 160, 190, 200]},
             {'bbox_label': 0, 'bbox': [250, 260, 350, 360]}]

# categories, 0 (car) and 1 (bicycle)
categories = ['car', 'bicycle']


# metric settings
coco_metric = CocoMetric(ann_file=None, metric=['bbox'], classwise=False, outfile_prefix=f'{tmp_dir.name}/test')
coco_metric.dataset_meta = dict(classes = categories)

coco_metric.process({}, [dict(pred_instances=mock_predictions,
                              img_id=0,
                              ori_shape=(640, 640),
                              instances=instances)])
eval_results = coco_metric.evaluate(size=1)
print(eval_results)

