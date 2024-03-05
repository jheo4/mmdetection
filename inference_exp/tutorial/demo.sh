# Single image demo
python ../demo/image_demo.py ../demo/demo.jpg \
    ../configs/rtmdet/rtmdet_l_8xb32-300e_coco.py \
    --weights ../checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth \
    --device cpu

# Video demo
python ../demo/video_demo.py ../demo/demo.mp4 \
    ../configs/rtmdet/rtmdet_l_8xb32-300e_coco.py \
    ../checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth \
    --out result.mp4

# Test model -- evaluation result
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
python tools/test.py rtmdet_tiny_8xb32-300e_coco.py rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth

# Analysis on the evaluation process and GT/Pred format
python ../tests/test_evaluation/test_metrics/test_coco_metric.py # coco format & coco metric
python ../tests/test_evaluation/test_metrics/test_mot_challenge_metrics.py # MOT format & MOT metric

