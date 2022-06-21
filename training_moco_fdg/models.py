import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
# from .utils import load_state_dict_from_url
# note the refinement module

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

#
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
#     'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
#     'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
#     'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
#     'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
# }

"""
ResNet with fc classifier
"""
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=60, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, Inhibition_module=None,
                 deep_stem=False, stem_width=32, avg_down=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer

        self.inplanes = stem_width*2 if deep_stem else 64

        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        if deep_stem:
            self.conv1 = nn.Sequential(
                conv3x3(3, stem_width, stride=2),
                norm_layer(stem_width),
                nn.ReLU(),
                conv3x3(stem_width, stem_width, stride=1),
                norm_layer(stem_width),
                nn.ReLU(),
                conv3x3(stem_width, self.inplanes, stride=1),
            )
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = norm_layer(self.inplanes if not deep_stem else stem_width*2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], avg_down=avg_down)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, avg_down=avg_down,
                                       dilate=replace_stride_with_dilation[0])

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, avg_down=avg_down,
                                       dilate=replace_stride_with_dilation[1])

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, avg_down=avg_down,
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # print(self.modules)

        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, avg_down=False, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if avg_down and stride != 1:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride, count_include_pad=False, ceil_mode=True),
                    conv1x1(self.inplanes, planes * block.expansion, 1),
                    norm_layer(planes * block.expansion)
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                    norm_layer(planes * block.expansion)
                )


        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    def _first_block(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
    def feature_extract(self,x,layer_num):
      if layer_num == 1:
          x = self._first_block(x)
          self.layer1_out = self.layer1(x)
          return self.layer1_out
      if layer_num == 2:
          self.layer2_out = self.layer2(x)
          return self.layer2_out
      if layer_num == 3:
          self.layer3_out = self.layer3(x)
          return self.layer3_out
      if layer_num == 4:
          self.layer4_out = self.layer4(x)
          return self.layer4_out
    def forward_classifier(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
    def forward(self, x):
        return self._forward_impl(x)
def _resnet(arch, block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model
def resnet18(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], **kwargs)
def resnet34(**kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], **kwargs)
def resnet50(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], **kwargs)
def resnet101(**kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], **kwargs)
def resnet152(**kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], **kwargs)
def resnext50_32x4d(**kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], **kwargs)
def resnext101_64x4d(**kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 64
    kwargs['width_per_group'] = 4
    return _resnet('resnext101_64x4d', Bottleneck, [3, 4, 23, 3], **kwargs)
def resnet50d(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    return _resnet('resnet50d', Bottleneck, [3, 4, 6, 3],
                   deep_stem=True, stem_width=32, avg_down=True,
                   **kwargs)


"""
ResNet with other classifiers
"""
class ResNet_no_fc(nn.Module):

    def __init__(self, block, layers, num_classes=60, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, Inhibition_module=None,
                 deep_stem=False, stem_width=32, avg_down=False):
        super(ResNet_no_fc, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer

        self.inplanes = stem_width*2 if deep_stem else 64

        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        if deep_stem:
            self.conv1 = nn.Sequential(
                conv3x3(3, stem_width, stride=2),
                norm_layer(stem_width),
                nn.ReLU(),
                conv3x3(stem_width, stem_width, stride=1),
                norm_layer(stem_width),
                nn.ReLU(),
                conv3x3(stem_width, self.inplanes, stride=1),
            )
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = norm_layer(self.inplanes if not deep_stem else stem_width*2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], avg_down=avg_down)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, avg_down=avg_down,
                                       dilate=replace_stride_with_dilation[0])

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, avg_down=avg_down,
                                       dilate=replace_stride_with_dilation[1])

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, avg_down=avg_down,
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # print(self.modules)

        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, avg_down=False, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if avg_down and stride != 1:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride, count_include_pad=False, ceil_mode=True),
                    conv1x1(self.inplanes, planes * block.expansion, 1),
                    norm_layer(planes * block.expansion)
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                    norm_layer(planes * block.expansion)
                )


        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    def _first_block(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x
    def feature_extract(self,x,layer_num):
      if layer_num == 1:
          x = self._first_block(x)
          self.layer1_out = self.layer1(x)
          return self.layer1_out
      if layer_num == 2:
          self.layer2_out = self.layer2(x)
          return self.layer2_out
      if layer_num == 3:
          self.layer3_out = self.layer3(x)
          return self.layer3_out
      if layer_num == 4:
          self.layer4_out = self.layer4(x)
          return self.layer4_out
    def forward_classifier(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
    def forward(self, x):
        return self._forward_impl(x)

def resnet18_(pretrained=False, **kwargs):  # 512
    model = ResNet_no_fc(BasicBlock, [2, 2, 2, 2], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model
def resnet34_(pretrained=False, **kwargs):  # 512
    model = ResNet_no_fc(BasicBlock, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model
def resnet50_(pretrained=False, **kwargs):  # 2048
    model = ResNet_no_fc(Bottleneck, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model
def resnet101_(pretrained=False, **kwargs):
    model = ResNet_no_fc(Bottleneck, [3, 4, 23, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model
def resnet152_(pretrained=False, **kwargs):
    model = ResNet_no_fc(Bottleneck, [3, 8, 36, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

class Classifier(nn.Module):
    def __init__(self, in_dim=512, num_classes=60, bias=False, scale=1.0, learn_scale=False, cls_type="linear"):
        super(Classifier, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.classifier_type = cls_type

        if self.classifier_type == "linear":
            self.layers = nn.Linear(in_dim, num_classes)

        elif self.classifier_type == "cosine":
            self.layers = CosineClassifier(
                num_channels=self.in_dim,
                num_classes=self.num_classes,
                scale=scale,
                learn_scale=learn_scale,
                bias=bias,
            )

        else:
            raise ValueError(
                "Not implemented / recognized classifier type {}".format(self.classifier_type)
            )

    def forward(self, features):
        scores = self.layers(features)
        return scores
class CosineClassifier(nn.Module):
    def __init__(
        self,
        num_channels,
        num_classes,
        scale=1.0,
        learn_scale=False,
        bias=False,
        normalize_x=True,
        normalize_w=True,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.normalize_x = normalize_x
        self.normalize_w = normalize_w

        weight = torch.FloatTensor(num_classes, num_channels).normal_(
            0.0, np.sqrt(2.0 / num_channels)
        )
        self.weight = nn.Parameter(weight, requires_grad=True)

        if bias:
            bias = torch.FloatTensor(num_classes).fill_(0.0)
            self.bias = nn.Parameter(bias, requires_grad=True)
        else:
            self.bias = None

        scale_cls = torch.FloatTensor(1).fill_(scale)
        self.scale_cls = nn.Parameter(scale_cls, requires_grad=learn_scale)

    def forward(self, x_in):
        assert x_in.dim() == 2
        return cosine_fully_connected_layer(
            x_in,
            self.weight.t(),
            scale=self.scale_cls,
            bias=self.bias,
            normalize_x=self.normalize_x,
            normalize_w=self.normalize_w,
        )

    def extra_repr(self):
        s = "num_channels={}, num_classes={}, scale_cls={} (learnable={})".format(
            self.num_channels,
            self.num_classes,
            self.scale_cls.item(),
            self.scale_cls.requires_grad,
        )
        learnable = self.scale_cls.requires_grad
        s = (
            f"num_channels={self.num_channels}, "
            f"num_classes={self.num_classes}, "
            f"scale_cls={self.scale_cls.item()} (learnable={learnable}), "
            f"normalize_x={self.normalize_x}, normalize_w={self.normalize_w}"
        )

        if self.bias is None:
            s += ", bias=False"
        return s
def cosine_fully_connected_layer(x_in, weight, scale=None, bias=None, normalize_x=True, normalize_w=True):
    # x_in: a 2D tensor with shape [batch_size x num_features_in]
    # weight: a 2D tensor with shape [num_features_in x num_features_out]
    # scale: (optional) a scalar value
    # bias: (optional) a 1D tensor with shape [num_features_out]

    assert x_in.dim() == 2
    assert weight.dim() == 2
    assert x_in.size(1) == weight.size(0)

    if normalize_x:
        x_in = F.normalize(x_in, p=2, dim=1, eps=1e-12)

    if normalize_w:
        weight = F.normalize(weight, p=2, dim=0, eps=1e-12)

    x_out = torch.mm(x_in, weight)

    if scale is not None:
        x_out = x_out * scale.view(1, -1)

    if bias is not None:
        x_out = x_out + bias.view(1, -1)

    return x_out


"""
Label Smoothing
"""
def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y
def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)



import functools
model_dict = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "resnet50d": resnet50d,
    "resnext50_32x4d":resnext50_32x4d,
    "resnext101_64x4d": resnext101_64x4d,
}


def create_net(args):
    net = None

    # srm does not have any input parameters

    kwargs = {}
    kwargs["num_classes"] = 60
    kwargs["Inhibition_module"] = None
    net = model_dict[args.arch.lower()](**kwargs)
    return net