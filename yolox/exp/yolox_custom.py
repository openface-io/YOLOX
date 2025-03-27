import os

from model.yolox.yolox.exp import Exp as MyExp


class yolox_custom_exp(MyExp):
    def __init__(self, model_name=None):
        super(yolox_custom_exp, self).__init__()

        if model_name=='yolox_s':
            self.depth = 0.33
            self.width = 0.50
        elif model_name=='yolox_m':
            self.depth = 0.67
            self.width = 0.75
        else:
            raise TypeError("Provide a model name")

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
