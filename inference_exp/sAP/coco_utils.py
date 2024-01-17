import torch

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
