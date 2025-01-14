import torch
from torch.nn import Conv2d, Sequential, ModuleList, BatchNorm2d
from torch import nn
from ..nn.mobilenet_v2 import MobileNetV2, InvertedResidual, ConvBNReLU

from .ssd import SSD, GraphPath
from .predictor import Predictor
from .config import mobilenetv1_ssd_config as config

class SeparableConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(SeparableConv2d, self).__init__(
            Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                   groups=in_channels, stride=stride, padding=padding),
            BatchNorm2d(in_channels),
            nn.ReLU(),
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        )


def create_mobilenetv2_ssd_lite(num_classes, width_mult=1.0, use_batch_norm=True, is_test=False):
    base_net = MobileNetV2().features

    source_layer_indexes = [
        GraphPath(14, 'conv', 3),
        19,
    ]
    extras = ModuleList([
        InvertedResidual(1280, 512, stride=2, expand_ratio=0.2),
        InvertedResidual(512, 256, stride=2, expand_ratio=0.25),
        InvertedResidual(256, 256, stride=2, expand_ratio=0.5),
        InvertedResidual(256, 64, stride=2, expand_ratio=0.25)
    ])

    regression_headers = ModuleList([
        SeparableConv2d(in_channels=round(576 * width_mult), out_channels=6 * 4,
                        kernel_size=3, padding=1),
        SeparableConv2d(in_channels=1280, out_channels=6 * 4, kernel_size=3, padding=1),
        SeparableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
        SeparableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
        SeparableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=1),
    ])

    classification_headers = ModuleList([
        SeparableConv2d(in_channels=round(576 * width_mult), out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeparableConv2d(in_channels=1280, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeparableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeparableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeparableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=1),
    ])

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)

def fuse_model_mobilenetv2_ssd_lite(net):
    for m in net.modules():
        if type(m) == ConvBNReLU or type(m) == SeparableConv2d:
            torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
        if type(m) == InvertedResidual:
            m.fuse_model()
    return net

def create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=torch.device('cpu')):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          config.image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor
