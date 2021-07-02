from collections import namedtuple

import torch
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, PReLU, ReLU, Sigmoid, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential


def load_resnet_pretrained_weights(model, numOfLayer):
    model_dict = model.state_dict()
    if numOfLayer == 50:
        checkpoint = torch.load('./pretrained_models/msceleb_ckpts/backbone_ir50_ms1m_epoch120.pth')
        indexToLayer = {'0': 'layer1.0.', '1': 'layer1.1.', '2': 'layer1.2.',
                        '3': 'layer2.0.', '4': 'layer2.1.', '5': 'layer2.2.', '6': 'layer2.3.',
                        '7': 'layer3.0.', '8': 'layer3.1.', '9': 'layer3.2.', '10': 'layer3.3.', '11': 'layer3.4.',
                        '12': 'layer3.5.', '13': 'layer3.6.', '14': 'layer3.7.', '15': 'layer3.8.', '16': 'layer3.9.',
                        '17': 'layer3.10.', '18': 'layer3.11.', '19': 'layer3.12.', '20': 'layer3.13.',
                        '21': 'layer4.0.', '22': 'layer4.1.', '23': 'layer4.2.'}
    else:
        checkpoint = torch.load('./pretrained_models/msceleb_ckpts/backbone_IR_18_HeadFC_Softmax_112_512_1.0_Epoch_156_lfw_112_0.994_X4_112_0.990_agedb_30_112_0.949.pth')
        indexToLayer = {'0': 'layer1.0.', '1': 'layer1.1.',
                        '2': 'layer2.0.', '3': 'layer2.1.',
                        '4': 'layer3.0.', '5': 'layer3.1.',
                        '6': 'layer4.0.', '7': 'layer4.1.'}

    newCheckpoint = {}
    for key, value in checkpoint.items():
        subStr = key.split('.', 2)
        if subStr[0] == 'body':
            newKey = indexToLayer[subStr[1]] + subStr[2]
            newCheckpoint[newKey] = value
        elif subStr[0] == 'output_layer':
            continue
        else:
            newCheckpoint[key] = value

    for key, value in model_dict.items():
        subStr = key.split('.', 2)
        if subStr[0] == 'fc' or subStr[0] == 'loc_fc' or \
                subStr[0] == 'Crop_Net' or subStr[0] == 'GCN' or \
                subStr[0] == 'SourceMean' or subStr[0] == 'TargetMean' or \
                subStr[0] == 'SourceBN' or subStr[0] == 'TargetBN' or \
                subStr[0] == 'output_layer':
            newCheckpoint[key] = value

    model.load_state_dict(newCheckpoint)
    return model


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x


class bottleneck_IR(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class bottleneck_IR_SE(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 18:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=2),
            get_block(in_channel=64, depth=128, num_units=2),
            get_block(in_channel=128, depth=256, num_units=2),
            get_block(in_channel=256, depth=512, num_units=2)
        ]
    elif num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]

    return blocks


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
