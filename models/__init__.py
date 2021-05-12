from .erfnet import ERFNet
from .roadseg import RoadSeg
from .vietnet import VietNet

model_factory = {
    'erfnet': ERFNet,
    'roadseg': RoadSeg,
}