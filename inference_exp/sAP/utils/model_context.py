from mmdet.apis import init_detector
from utils.coco_evaluator import COCO_Evaluator

'''
Each model has its own context, which includes:
    1. model_cfg: model config file path
    2. model_weight: model weight file path
    3. device: device to run the model (cpu or cuda:0)
    4. model: mmdet detector instance
'''
class Model_Context:
    model_cfg = ''
    model_weight = ''
    device = 'cuda:0'
    model = None
    evaluator = COCO_Evaluator()


    def __init__(self, model_cfg, model_weight, device, evaluator):
        self.model_cfg = model_cfg
        self.model_weight = model_weight
        self.device = device

        # ??? is this the right place to init the model?
        self.model = init_detector(model_cfg, model_weight, device=device) # return nn.Module

        # per-model evaluator
        self.evaluator = evaluator

