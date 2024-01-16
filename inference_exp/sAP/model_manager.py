from model_context import Model_Context

class Model_Manager:
    # models['model_name'] = ModelContext
    models = {}

    def __init__(self):
        None

    # add model to model manager's list
    def add_model(self, model_name, model_cfg, model_weight, device='cuda:0'):
        self.models[model_name] = Model_Context(model_cfg, model_weight, device)

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


    def print_all_models(self):
        for model_name in self.models:
            print("="*120)
            print(model_name)
            print("\tcfg: " + self.models[model_name].model_cfg)
            print("\tweight: " + self.models[model_name].model_weight)
            print("\tdevice: " + self.models[model_name].device)
        print("="*120)

