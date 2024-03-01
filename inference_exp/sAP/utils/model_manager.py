from utils.model_context import Model_Context
from utils.coco_evaluator import COCO_Evaluator
import os
from colorama import Fore, Style
import json

class Model_Manager:
    # models['model_name'] = ModelContext
    models = {}


    def __init__(self):
        None


    # add model to model manager's list
    def add_model(self, model_name, model_cfg, model_weight, device='cuda:0', evaluator=COCO_Evaluator()):
        self.models[model_name] = Model_Context(model_cfg, model_weight, device, evaluator)


    # get model from model manager's list
    def get_model(self, model_name):
        return self.models[model_name]


    # remove model from model manager's list
    def remove_model(self, model_name):
        del self.models[model_name] # nn.Module deletion: as other python variables, it will be deleted when no reference to it
        # just-in-case: release GPU memory TODO: check if this is necessary
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()


    # add models from json file and print all loaded model cases
    def add_model_cases_from_json(self, json_file):
        # check if file exists
        if not os.path.exists(json_file):
            print(Fore.RED + "Error: json file not found" + Style.RESET_ALL)
            return

        # load json file
        model_configs = json.load(open(json_file))
        model_dir = model_configs['model_dir']
        model_specs = model_configs['models']

        # add model cases
        for model_spec in model_specs:
            model_case    = model_spec['case']
            model_cfg    = os.path.join(model_dir, model_spec['cfg'])
            model_weight = os.path.join(model_dir, model_spec['weight'])
            device = model_spec['device']

            self.add_model(model_case, model_cfg, model_weight, device)

        # print all loaded model cases
        self.print_all_models()


    def print_all_models(self):
        for model_name in self.models:
            print("="*120)
            print(model_name)
            print("\tcfg: " + self.models[model_name].model_cfg)
            print("\tweight: " + self.models[model_name].model_weight)
            print("\tdevice: " + self.models[model_name].device)
        print("="*120)

