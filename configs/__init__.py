from .bisenetv1 import cfg as bisenetv1_cfg
# from .bisenetv2 import cfg as bisenetv2_cfg

class cfg_dict(object):
    def __init__(self, d) -> None:
        super().__init__()
        self.__dict__ = d

cfg_factory = dict(
    bisenetv1 = cfg_dict(bisenetv1_cfg)
)