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


def jump_data_index_by_execution_time(data_index, data_rate, exec_time_ms):
    timeslot = 1000 / data_rate
    jump = int(exec_time_ms / timeslot) + 1
    # print(f"Jumping from {data_index} to {data_index + jump}")
    return data_index + jump


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
    print(jump_data_index_by_execution_time(0, 30, 130))

