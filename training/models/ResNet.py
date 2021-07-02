import copy
import torch
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential
from models.ResNet_utils import bottleneck_IR, get_block, get_blocks, init_weights


# Support: ['IR_18', 'IR_50']
class Backbone_Global_Local(nn.Module):
    def __init__(self, numOfLayer, class_num=7):

        super(Backbone_Global_Local, self).__init__()

        unit_module = bottleneck_IR

        self.input_layer = Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            BatchNorm2d(64), PReLU(64))

        blocks = get_blocks(numOfLayer)
        self.layer1 = Sequential(
            *[unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride) for bottleneck in
              blocks[0]])  # get_block(in_channel=64, depth=64, num_units=3)])
        self.layer2 = Sequential(
            *[unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride) for bottleneck in
              blocks[1]])  # get_block(in_channel=64, depth=128, num_units=4)])
        self.layer3 = Sequential(
            *[unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride) for bottleneck in
              blocks[2]])  # get_block(in_channel=128, depth=256, num_units=14)])
        self.layer4 = Sequential(
            *[unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride) for bottleneck in
              blocks[3]])  # get_block(in_channel=256, depth=512, num_units=3)])

        self.output_layer = Sequential(
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))

        cropNet_modules = []
        cropNet_blocks = [get_block(in_channel=128, depth=256, num_units=2),
                          get_block(in_channel=256, depth=512, num_units=2)]
        for block in cropNet_blocks:
            for bottleneck in block:
                cropNet_modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        cropNet_modules += [
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.ReLU()]
        self.Crop_Net = nn.ModuleList([copy.deepcopy(nn.Sequential(*cropNet_modules)) for i in range(5)])
        self.fc = nn.Linear(64 + 320, class_num)
        self.fc.apply(init_weights)

        self.loc_fc = nn.Linear(320, class_num)
        self.loc_fc.apply(init_weights)
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))

    def classify(self, imgs, locations):

        featureMap = self.input_layer(imgs)

        featureMap1 = self.layer1(featureMap)  # Batch * 64 * 56 * 56
        featureMap2 = self.layer2(featureMap1)  # Batch * 128 * 28 * 28
        featureMap3 = self.layer3(featureMap2)  # Batch * 256 * 14 * 14
        featureMap4 = self.layer4(featureMap3)  # Batch * 512 * class_num * class_num

        global_feature = self.output_layer(featureMap4).view(featureMap.size(0), -1)  # Batch * 64
        loc_feature = self.crop_featureMap(featureMap2, locations)  # Batch * 320
        feature = torch.cat((global_feature, loc_feature), 1)  # Batch * (64+320)

        feature = feature.view(feature.size(0), -1).narrow(1, 0, 64 + 320)  # Batch * (64+320)
        loc_feature = feature.narrow(1, 64, 320)  # Batch * 320

        pred = self.fc(feature)  # Batch * class_num
        loc_pred = self.loc_fc(loc_feature)  # Batch * class_num

        return feature, pred, loc_pred

    def forward(self, imgs, locations):

        return self.classify(imgs, locations)

    def output_num(self):
        return 64 * 6

    def get_parameters(self):
        parameter_list = [{"params": self.input_layer.parameters(), "lr_mult": 1, 'decay_mult': 2},
                          {"params": self.layer1.parameters(), "lr_mult": 1, 'decay_mult': 2},
                          {"params": self.layer2.parameters(), "lr_mult": 1, 'decay_mult': 2},
                          {"params": self.layer3.parameters(), "lr_mult": 1, 'decay_mult': 2},
                          {"params": self.layer4.parameters(), "lr_mult": 1, 'decay_mult': 2},
                          {"params": self.output_layer.parameters(), "lr_mult": 10, 'decay_mult': 2},
                          {"params": self.fc.parameters(), "lr_mult": 10, 'decay_mult': 2},
                          {"params": self.loc_fc.parameters(), "lr_mult": 10, 'decay_mult': 2},
                          {"params": self.Crop_Net.parameters(), "lr_mult": 10, 'decay_mult': 2}]
        return parameter_list

    def crop_featureMap(self, featureMap, locations):
        batch_size = featureMap.size(0)
        map_ch = featureMap.size(1)
        map_len = featureMap.size(2)

        grid_ch = map_ch
        grid_len = 7  # 14, 6, 4

        feature_list = []
        for i in range(5):
            grid_list = []
            for j in range(batch_size):
                w_min = locations[j, i, 0] - int(grid_len / 2)
                w_max = locations[j, i, 0] + int(grid_len / 2)
                h_min = locations[j, i, 1] - int(grid_len / 2)
                h_max = locations[j, i, 1] + int(grid_len / 2)

                map_w_min = max(0, w_min)
                map_w_max = min(map_len - 1, w_max)
                map_h_min = max(0, h_min)
                map_h_max = min(map_len - 1, h_max)

                grid_w_min = max(0, 0 - w_min)
                grid_w_max = grid_len + min(0, map_len - 1 - w_max)
                grid_h_min = max(0, 0 - h_min)
                grid_h_max = grid_len + min(0, map_len - 1 - h_max)

                grid = torch.zeros(grid_ch, grid_len, grid_len)
                if featureMap.is_cuda:
                    grid = grid.cuda()

                grid[:, grid_h_min:grid_h_max + 1, grid_w_min:grid_w_max + 1] = featureMap[j, :,
                                                                                map_h_min:map_h_max + 1,
                                                                                map_w_min:map_w_max + 1]

                grid_list.append(grid)

            feature = torch.stack(grid_list, dim=0)
            feature_list.append(feature)

        # feature list: 5 * [ batch_size * channel * 3 * 3 ]
        output_list = []
        for i in range(5):
            output = self.Crop_Net[i](feature_list[i])
            output = self.GAP(output)
            output_list.append(output)

        loc_feature = torch.stack(output_list, dim=1)  # batch_size * 5 * 64 * 1 * 1
        loc_feature = loc_feature.view(batch_size, -1)  # batch_size * 320

        return loc_feature


class Backbone_onlyGlobal(nn.Module):
    def __init__(self, numOfLayer, class_num):
        super(Backbone_onlyGlobal, self).__init__()

        unit_module = bottleneck_IR

        self.input_layer = Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            BatchNorm2d(64), PReLU(64))

        blocks = get_blocks(numOfLayer)
        self.layer1 = Sequential(
            *[unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride) for bottleneck in
              blocks[0]])  # get_block(in_channel=64, depth=64, num_units=3)])
        self.layer2 = Sequential(
            *[unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride) for bottleneck in
              blocks[1]])  # get_block(in_channel=64, depth=128, num_units=4)])
        self.layer3 = Sequential(
            *[unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride) for bottleneck in
              blocks[2]])  # get_block(in_channel=128, depth=256, num_units=14)])
        self.layer4 = Sequential(
            *[unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride) for bottleneck in
              blocks[3]])  # get_block(in_channel=256, depth=512, num_units=3)])

        self.output_layer = Sequential(
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))

        self.fc = nn.Linear(64, class_num)
        self.fc.apply(init_weights)

    def classify(self, imgs):
        featureMap = self.input_layer(imgs)

        featureMap1 = self.layer1(featureMap)  # Batch * 64 * 56 * 56
        featureMap2 = self.layer2(featureMap1)  # Batch * 128 * 28 * 28
        featureMap3 = self.layer3(featureMap2)  # Batch * 256 * 14 * 14
        featureMap4 = self.layer4(featureMap3)  # Batch * 512 * 7 * 7

        feature = self.output_layer(featureMap4).view(featureMap.size(0), -1)  # Batch * 64

        pred = self.fc(feature)  # Batch * 7
        loc_pred = None

        return feature, pred, loc_pred

    def forward(self, imgs, locations):
        return self.classify(imgs)

    def output_num(self):
        return 64

    def get_parameters(self):
        parameter_list = [{"params": self.input_layer.parameters(), "lr_mult": 1, 'decay_mult': 2},
                          {"params": self.layer1.parameters(), "lr_mult": 1, 'decay_mult': 2},
                          {"params": self.layer2.parameters(), "lr_mult": 1, 'decay_mult': 2},
                          {"params": self.layer3.parameters(), "lr_mult": 1, 'decay_mult': 2},
                          {"params": self.layer4.parameters(), "lr_mult": 1, 'decay_mult': 2},
                          {"params": self.output_layer.parameters(), "lr_mult": 10, 'decay_mult': 2},
                          {"params": self.fc.parameters(), "lr_mult": 10, 'decay_mult': 2},
                          ]
        return parameter_list


def IR_global_local(numOfLayer, class_num):
    """Constructs a ir-18/ir-50 model."""

    model = Backbone_Global_Local(numOfLayer, class_num)

    return model


def IR_global(numOfLayer, class_num):
    """Constructs a ir-18/ir-50 model."""

    model = Backbone_onlyGlobal(numOfLayer, class_num)

    return model
