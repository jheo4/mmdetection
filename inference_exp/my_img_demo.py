from mmengine.logging import print_log
import cv2

# Two different ways to invoke a inference model
from mmdet.apis import DetInferencer
from mmdet.apis import init_detector, inference_detector

from mmdet.registry import VISUALIZERS # reflect inference result to image

project_dir = '/home/jin/mnt/github/mmdetection/'

model_cfg = project_dir + 'configs/rtmdet/rtmdet_l_8xb32-300e_coco.py'
weights = project_dir + 'checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth'
device = 'cuda:0'

inferencer = DetInferencer(model=model_cfg, weights=weights, device=device)
detector   = init_detector(model_cfg, weights, device=device)

image_fn = project_dir + 'demo/demo.jpg'
image_to_infer = cv2.imread(image_fn)
# bgr to rgb
image_to_infer = image_to_infer[:, :, ::-1]

result = inference_detector(detector, image_to_infer)
print("##############################################---------------##################################################")

# Prediction Result Structure
#   - metainfo
#     {'pad_shape': (640, 640), 'img_path': None, 'img_id': 0, 'ori_shape': (427, 640), 'img_shape': (640, 640),
#      'batch_input_shape': (640, 640), 'scale_factor': (1.0, 1.0)}
#   - pred_instances
#      - bboxes
#      - scores
#      - labels
#   - gt_instances
#      - bboxes
#      - labels
pred_meta = result.metainfo
pred_bboxes = result.get('pred_instances').get('bboxes').cpu().numpy()
pred_scores = result.get('pred_instances').get('scores').cpu().numpy()
pred_labels = result.get('pred_instances').get('labels').cpu().numpy()
gt_bboxes = result.get('gt_instances').get('bboxes').cpu().numpy()
gt_labels = result.get('gt_instances').get('labels').cpu().numpy()

print(gt_bboxes)
print(gt_labels)


print("##############################################---------------##################################################")

visualizer_cfg = dict(name='visualizer',
                      type='DetLocalVisualizer',
                      vis_backends=[dict(type='LocalVisBackend'),])
visualizer = VISUALIZERS.build(visualizer_cfg)
visualizer.dataset_meta = detector.dataset_meta

visualizer.add_datasample(name='result',
                          image=image_to_infer,
                          data_sample=result,
                          draw_gt=False,
                          pred_score_thr=0.3,
                          show=True)
image_with_result = visualizer.get_image()
# rgb to bgr
image_with_result = image_with_result[:, :, ::-1]
cv2.imshow('result', image_with_result)
cv2.waitKey(0)
cv2.destroyAllWindows()

