import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.resnet import model_urls, conv3x3, conv1x1, Bottleneck, BasicBlock

def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)


class ModifiedResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, downscale=False):
        super(ModifiedResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
        #                                dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.downscale = downscale

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        return x1, x2, x3

    def forward(self, x):
        x1, x2, x3 = self._forward_impl(x)

        return x1, x2, x3


# class Bottleneckdecoder(nn.Module):
#
#     def __init__(self, ich, och, upsample=False):
#         super(Bottleneckdecoder, self).__init__()
#
#         self.upsample = upsample
#         self.upsampleLayer = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),
#                                            nn.Conv2d(ich, och, kernel_size=1, bias=False),
#                                            nn.BatchNorm2d(och, eps=1e-05, momentum=0.1, affine=True,
#                                                           track_running_stats=True)) if upsample else None
#
#         self.conv1 = nn.Sequential(nn.Conv2d(ich, och, 3, 1, 1, bias=False),
#                                    nn.BatchNorm2d(och, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
#         self.relu = nn.ReLU(True)
#         self.conv2 = nn.Conv2d(och, och, 3, 1, 1, bias=False)
#         self.bn2 = nn.BatchNorm2d(och, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#
#         self.conv3 = nn.Sequential(nn.Conv2d(och, och*4, 1, 1, 1, bias=False),
#                                    nn.BatchNorm2d(och*4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
#
#     def forward(self, x):
#         if self.upsample:
#             identity = self.upsampleLayer(x)
#             out = identity
#         else:
#             out = self.conv1(x)
#             out = self.relu(out)
#             out = self.conv2(out)
#             out = self.bn2(out)
#             out = self.relu(out)
#
#             out = self.conv3(out)
#             out = self.relu(out)
#
#         return out
#
#
# class ModifiedResNetdecoder(nn.Module):
#     def __init__(self):
#         super(ModifiedResNetdecoder, self).__init__()
#
#         self.decoder = nn.Sequential(Bottleneckdecoder(1024, 256),
#                                      Bottleneckdecoder(1024, 256),
#                                      Bottleneckdecoder(1024, 256),
#                                      Bottleneckdecoder(1024, 256),
#                                      Bottleneckdecoder(1024, 256),
#                                      Bottleneckdecoder(1024, 256),
#                                      Bottleneckdecoder(256, 512, True),
#                                      Bottleneckdecoder(512, 128),
#                                      Bottleneckdecoder(512, 128),
#                                      Bottleneckdecoder(512, 128),
#                                      Bottleneckdecoder(512, 128),
#                                      Bottleneckdecoder(128, 256, True),
#                                      Bottleneckdecoder(256, 64),
#                                      Bottleneckdecoder(256, 64),
#                                      Bottleneckdecoder(256, 64),
#                                      Bottleneckdecoder(256, 64, True))
#
#         self.output = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),
#                                     nn.Conv2d(64, 3, kernel_size=1, bias=False),
#                                     nn.Tanh())
#
#     def forward(self, x):
#         x = self.decoder(x)
#         x_re = self.output(x)
#         return x_re


class BasicBlockdecoder(nn.Module):
    def __init__(self, ich, och, upsample=False):
        super(BasicBlockdecoder, self).__init__()

        self.upsample = upsample
        self.upsampleLayer = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),
                                           nn.Conv2d(ich, och, kernel_size=1, bias=False),
                                           nn.BatchNorm2d(och, eps=1e-05, momentum=0.1, affine=True,
                                                          track_running_stats=True)) if upsample else None

        self.conv1 = nn.Sequential(nn.Conv2d(ich, och, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(och, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(och, och, 3, 1, 1, bias=False)
        self.bnrm = nn.BatchNorm2d(och, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        if self.upsample:
            identity = self.upsampleLayer(x)
            out = identity
        else:
            identity = x
            out = self.conv1(x)

        out = self.relu(out)
        out = self.conv2(out)
        out = self.bnrm(out)

        out += identity
        out = self.relu(out)
        return out


class ModifiedResNetdecoder(nn.Module):
    def __init__(self):
        super(ModifiedResNetdecoder, self).__init__()

        self.decoder = nn.Sequential(BasicBlockdecoder(1024, 512, True),
                                     BasicBlockdecoder(512, 512),
                                     BasicBlockdecoder(512, 256, True),
                                     BasicBlockdecoder(256, 256),
                                     BasicBlockdecoder(256, 128, True))

        self.output = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),
                                    nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                    nn.Conv2d(64, 3, kernel_size=1, bias=False),
                                    nn.Tanh())

    def forward(self, x):
        x = self.decoder(x)
        x_re = self.output(x)
        return x_re


class ctResNet(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self):
        super(ctResNet, self).__init__()
        self.encoder = ModifiedResNet(block=Bottleneck, layers=[3, 4, 6, 3])  # resnet50
        # self.encoder = ModifiedResNet(block=Bottleneck, layers=[3, 4, 23, 3])  # resnet101
        # self.encoder = ModifiedResNet(block=Bottleneck, layers=[3, 8, 36, 3])  # resnet152
        self.decoder = ModifiedResNetdecoder()

    def forward(self, x):
        x1, x2, x3 = self.encoder(x)
        # print("x1,x2,x3 shape: ", x1.shape, x2.shape, x3.shape)  # [2, 256, 32, 32] [2, 512, 16, 16] [2, 1024, 8, 8]
        x_re = self.decoder(x3)
        # print("x_re shape: ", x_re.shape)  # [2, 3, 128, 128]
        return x1, x2, x3, x_re


if __name__ == '__main__':
    from torchsummary import summary

    # class Opt:
    #     def __init__(self, nz, ngpu):
    #         # self.isize = isize
    #         self.nz = nz
    #         # self.nc = nc
    #         # self.ngf = ngf
    #         self.ngpu = ngpu
    #         # self.extralayers = extralayers
    model = ctResNet()
    model.cuda()
    summary(model, input_size=(3, 320, 448))



