import os
from colorama import Fore, Style

class COCO_Evaluator:
    eval_datapoints = 0
    # https://cocodataset.org/#detection-eval
    mAP = 0
    mAP_50 = 0
    mAP_75 = 0
    mAP_small = 0
    mAP_medium = 0
    mAP_large = 0


    def __init__(self):
        None


    def calculate_mean(self, mAP, cur_mean_mAP):
        return cur_mean_mAP * (self.eval_datapoints / (self.eval_datapoints + 1)) + max(mAP, 0) * (1 / (self.eval_datapoints + 1))


    def update_mAP(self, eval_result):
        self.mAP        = self.calculate_mean(eval_result['coco/bbox_mAP'], self.mAP)
        self.mAP_50     = self.calculate_mean(eval_result['coco/bbox_mAP_50'], self.mAP_50)
        self.mAP_75     = self.calculate_mean(eval_result['coco/bbox_mAP_75'], self.mAP_75)
        self.mAP_small  = self.calculate_mean(eval_result['coco/bbox_mAP_s'], self.mAP_small)
        self.mAP_medium = self.calculate_mean(eval_result['coco/bbox_mAP_m'], self.mAP_medium)
        self.mAP_large  = self.calculate_mean(eval_result['coco/bbox_mAP_l'], self.mAP_large)
        self.eval_datapoints += 1


    def print_eval_result(self):
        print(Fore.CYAN + "COCO Evaluation Results" + Style.RESET_ALL)
        print("\tmAP: ", self.mAP)
        print("\tmAP_50: ", self.mAP_50)
        print("\tmAP_75: ", self.mAP_75)
        print("\tmAP_small: ", self.mAP_small)
        print("\tmAP_medium: ", self.mAP_medium)
        print("\tmAP_large: ", self.mAP_large)


if __name__ == "__main__":
    coco_evaluator = COCO_Evaluator()
    coco_evaluator.update_mAP({'coco/bbox_mAP': 1.0, 'coco/bbox_mAP_50': 1.0, 'coco/bbox_mAP_75': 1.0, 'coco/bbox_mAP_s': -1.0, 'coco/bbox_mAP_m': 1.0, 'coco/bbox_mAP_l': 1.0})
    coco_evaluator.update_mAP({'coco/bbox_mAP': 0.6, 'coco/bbox_mAP_50': 0.6, 'coco/bbox_mAP_75': 0.6, 'coco/bbox_mAP_s': -0.6, 'coco/bbox_mAP_m': 0.6, 'coco/bbox_mAP_l': 0.6})
    coco_evaluator.print_eval_result()

