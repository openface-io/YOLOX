import os

from model.yolox.yolox.exp import Exp as MyExp


class yolox_s_exp(MyExp):
    def __init__(self):
        super(yolox_s_exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
