import copy
import os
import tempfile

import torch

from mmengine.structures import BaseDataElement, InstanceData
from mmdet.evaluation import MOTChallengeMetric
from mmdet.structures import DetDataSample, TrackDataSample

# OS path separator
sep = os.path.sep
print(f'os.path.sep = {sep}')

# Temp result dir
tmp_dir = tempfile.TemporaryDirectory()
print(f'tmp_dir.name = {tmp_dir.name}')


eval_results = {}
# some calculations to fillout eval_results
metric = MOTChallengeMetric(metric=['HOTA', 'CLEAR', 'Identity'],
                            format_only= False,
                            outfile_prefix=tmp_dir.name)
metric.dataset_meta = {'classes' : ('pedestrian', )}
data_batch = dict(input=None, data_samples=None)

# Mock GT instances (1, 2)
instance = [{'bbox_label': 0,
             'bbox': [0, 0, 100, 100],
             'ignore_flag': 0,
             'instance_id': 1,
             'mot_conf': 1.0,
             'category_id': 1,
             'visibility': 1.0},
            {'bbox_label': 0,
             'bbox': [0, 0, 100, 100],
             'ignore_flag': 0,
             'instance_id': 2,
             'mot_conf': 1.0,
             'category_id': 1,
             'visibility': 1.0}]
instance2 = copy.deepcopy(instance)

# Mock prediction (1, 2)
prediction = dict(bboxes=torch.tensor([[0, 0, 100, 100],
                                       [0, 0, 100, 40]]),
                  instances_id=torch.tensor([1, 2]),
                  scores=torch.tensor([1.0, 1.0]))
prediction2 = copy.deepcopy(prediction)

prediction = InstanceData(**prediction)
prediction2 = InstanceData(**prediction2)

# Mock frame with metadata (1, 2)
img = DetDataSample()
img.pred_track_instances = prediction # consecutive predictions
img.instances = instance
img.set_metainfo(dict(frame_id=0,
                      ori_video_length=2,
                      video_length=2,
                      img_id=1,
                      img_path=f'tmp{sep}img1{sep}000001.jpg'))

img2 = DetDataSample()
img2.pred_track_instances = prediction2 # consecutive predictions
img2.instances = instance2
img2.set_metainfo(dict(frame_id=1,
                       ori_video_length=2,
                       video_length=2,
                       img_id=2,
                       img_path=f'tmp{sep}img2{sep}000002.jpg'))


# Tracking data sample from the frames with metadata
track_sample = TrackDataSample()
track_sample.video_data_samples = [img, img2]

predictions = []
if isinstance(track_sample, BaseDataElement):
    predictions.append(track_sample.to_dict())
    print(f'predictions = {predictions}')

metric.process(data_batch, predictions)
eval_results = metric.evaluate()

target = {'motchallenge-metric/IDF1': 0.5,
          'motchallenge-metric/MOTA': 0,
          'motchallenge-metric/HOTA': 0.755,
          'motchallenge-metric/IDSW': 0,}

for key in target:
    print(f'eval_results[key] - target[key] = {round(eval_results[key] - target[key], 2)}')

