import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = self.priorbox.forward()
        self.size = size

        # SSD network
        self.mobilenetv2 = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: localization layers, Shape: [batch, num_priors, 4]
                    2: confidence layers, Shape: [batch, num_priors, num_classes]
                    3: priorbox layers, Shape: [num_priors, 4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply MobileNetV2 up to layer15/expand
        for k in range(13):
            x = self.mobilenetv2[k](x)

        s = self.mobilenetv2[14].conv[:3](x)
        sources.append(s)

        # apply MobileNetV2 up to the last layer
        for k in range(13, len(self.mobilenetv2)):
            x = self.mobilenetv2[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(x.data.type())                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext in ('.pkl', '.pth'):
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class InvertedResidual_extra(nn.Module):
    def __init__(self, inp, oup, stride=2):
        super(InvertedResidual_extra, self).__init__()
        hidden_dim = oup // 2

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


def mobilenetv2(cfg, i, width_mult=1.):
    input_channel = _make_divisible(32 * width_mult, 8)
    layers = [conv_3x3_bn(i, input_channel, 2)]
    # building inverted residual blocks
    block = InvertedResidual
    for t, c, n, s in cfg:
        output_channel = _make_divisible(c * width_mult, 8)
        layers.append(block(input_channel, output_channel, s, t))
        input_channel = output_channel
        for i in range(1, n):
            layers.append(block(input_channel, output_channel, 1, t))
            input_channel = output_channel
    # building last several layers
    output_channel = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
    layers.append(conv_1x1_bn(input_channel, output_channel))
    return layers


def add_extras(cfg, i):
    # Extra layers added to MobileNetV2 for feature scaling
    layers = []
    in_channels = i
    block = InvertedResidual_extra
    for v in cfg:
        layers.append(block(in_channels, v))
        in_channels = v
    return layers


def multibox(mobilenetv2, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    mobile_source = [14, -1]
    for k, v in enumerate(mobile_source):
        layer = mobilenetv2[v]
        if hasattr(layer, 'conv'):
            layer = layer.conv
        loc_layers += [nn.Conv2d(layer[0].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(layer[0].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers, 2):
        loc_layers += [nn.Conv2d(v.conv[-3].out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.conv[-3].out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return mobilenetv2, extra_layers, (loc_layers, conf_layers)


base = {
    '320': [
           # t, c, n, s
           [1,  16, 1, 1],
           [6,  24, 2, 2],
           [6,  32, 3, 2],
           [6,  64, 4, 2],
           [6,  96, 3, 1],
           [6, 160, 3, 2],
           [6, 320, 1, 1],
           ], 
    '512': [],
}
extras = {
    '320': [512, 256, 256, 128],
    '512': [],
}
mbox = {
    '320': [3, 6, 6, 6, 6, 6],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(phase, size=320, num_classes=21, width_mult=1.):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 320:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(mobilenetv2(base[str(size)], 3, width_mult),
                                     add_extras(extras[str(size)], _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)
