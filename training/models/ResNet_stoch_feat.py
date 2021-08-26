import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential
from torch.utils import data
from models.ResNet_utils import bottleneck_IR, get_block, get_blocks, init_weights
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class GradReverse(Function):
    @staticmethod
    def forward(self, x):
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        return (grad_output * -1.0)


def grad_reverse(x):
    return GradReverse().apply(x)
      
class Stochastic_Features_cls(nn.Module):
    def __init__(self, args, input_dim=100):
        super(Stochastic_Features_cls, self).__init__()

        self.cls = nn.Linear(input_dim, args.class_num)
        self.cls.apply(init_weights)

    def forward(self, x):
        out = self.cls(x)
        return out



# Support: ['IR_18', 'IR_50']
class Backbone_Global_Local_stoch_feat(nn.Module):
    def __init__(self, numOfLayer, feature_dim=384):

        super(Backbone_Global_Local_stoch_feat, self).__init__()
        self.feature_dim = feature_dim
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
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_mean = Sequential(nn.Linear(384, self.feature_dim), nn.ReLU())
        self.fc_sigma = nn.Linear(384, self.feature_dim)

        self.fc_mean.apply(init_weights)
        self.fc_sigma.apply(init_weights)

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
        return feature

    def forward(self, imgs, locations, sample=False):

        x = self.classify(imgs, locations)
        feature = self.fc_mean(x)
        sigma = nn.Softplus()(self.fc_sigma(x))
        return feature, sigma

    def output_num(self):
        return self.feature_dim #100 #64 * 6

    def get_parameters(self):
        parameter_list = [{'params':list(self.input_layer.parameters())+list(self.layer1.parameters())+list(self.layer2.parameters())+
                            list(self.layer3.parameters())+list(self.layer4.parameters()), 'lr_mult':1, 'decay_mult':2},
                          {'params':list(self.output_layer.parameters())+list(self.Crop_Net.parameters())+list(self.fc_mean.parameters())+
                          list(self.fc_sigma.parameters()), 'lr_mult':10, 'decay_mult':2}]
        return parameter_list

    def get_parameters_update(self):
        parameter_list = list(self.output_layer.parameters())+list(self.Crop_Net.parameters())+list(self.fc_mean.parameters())+list(self.fc_sigma.parameters())
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


def IR_global_local_stoch_feat(numOfLayer, feature_dim=384):
    """Constructs a ir-18/ir-50 model."""

    model = Backbone_Global_Local_stoch_feat(numOfLayer, feature_dim=feature_dim)

    return model

class Backbone_Global_Local_stoch_feat_384(nn.Module):
    def __init__(self, numOfLayer, feature_dim=384):
        super(Backbone_Global_Local_stoch_feat_384, self).__init__()
        self.feature_dim = feature_dim
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
            nn.ReLU())
            
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))

        self.global_layer_sigma = Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7, 7), stride=(1, 1), padding=(1, 1)),
            nn.Softplus())

        cropNet_modules = []
        cropNet_blocks = [get_block(in_channel=128, depth=256, num_units=2),
                          get_block(in_channel=256, depth=512, num_units=2)]
        for block in cropNet_blocks:
            for bottleneck in block:
                cropNet_modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))

        # cropNet_mean_module = Sequential(nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #                                 nn.ReLU(),
        #                                 nn.AdaptiveAvgPool2d((1, 1)))
        # cropNet_sigma_module = Sequential(nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
        #                                 nn.AdaptiveAvgPool2d((1, 1)),
        #                                 nn.Softplus())

        cropNet_mean_module = Sequential(nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        nn.ReLU())
                                        
        cropNet_sigma_module = Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1)), 
                                        nn.Softplus())

        self.Crop_Net = nn.ModuleList([copy.deepcopy(nn.Sequential(*cropNet_modules)) for i in range(5)])
        self.Crop_Net_mean = nn.ModuleList([copy.deepcopy(nn.Sequential(*cropNet_mean_module)) for i in range(5)])
        self.Crop_Net_sigma = nn.ModuleList([copy.deepcopy(nn.Sequential(*cropNet_sigma_module)) for i in range(5)])

        # self.global_layer_mean.apply(init_weights)
        self.global_layer_sigma.apply(init_weights)

        self.Crop_Net_mean.apply(init_weights)
        self.Crop_Net_sigma.apply(init_weights)

    def forward(self, imgs, locations):

        featureMap = self.input_layer(imgs)

        featureMap1 = self.layer1(featureMap)  # Batch * 64 * 56 * 56
        featureMap2 = self.layer2(featureMap1)  # Batch * 128 * 28 * 28
        featureMap3 = self.layer3(featureMap2)  # Batch * 256 * 14 * 14
        featureMap4 = self.layer4(featureMap3)  # Batch * 512 * class_num * class_num

        pre_global_feature = self.output_layer(featureMap4)
        global_feature = self.GAP(pre_global_feature).view(featureMap.size(0), -1)
        global_feature_sigma = self.global_layer_sigma(pre_global_feature).view(featureMap.size(0), -1)

        # global_feature = self.global_layer_mean(featureMap4).view(featureMap.size(0), -1)  # Batch * 64
        # global_feature_sigma = self.global_layer_sigma(featureMap4).view(featureMap.size(0), -1)  # Batch * 64

        loc_feature, loc_feature_sigma = self.crop_featureMap(featureMap2, locations)  # Batch * 320

        feature = torch.cat((global_feature, loc_feature), 1)  # Batch * (64+320)
        feature = feature.view(feature.size(0), -1).narrow(1, 0, 64 + 320)  # Batch * (64+320)

        feature_sigma = torch.cat((global_feature_sigma, loc_feature_sigma), 1)  # Batch * (64+320)
        feature_sigma = feature_sigma.view(feature_sigma.size(0), -1).narrow(1, 0, 64 + 320)  # Batch * (64+320)
        return feature, feature_sigma

    def output_num(self):
        return self.feature_dim #100 #64 * 6

    def get_parameters(self):
        parameter_list = [{'params':list(self.input_layer.parameters())+list(self.layer1.parameters())+list(self.layer2.parameters())+
                            list(self.layer3.parameters())+list(self.layer4.parameters()), 'lr_mult':1, 'decay_mult':2},
                          {'params':list(self.output_layer.parameters())+list(self.global_layer_sigma.parameters())+
                          list(self.Crop_Net.parameters())+list(self.Crop_Net_mean.parameters())+list(self.Crop_Net_sigma.parameters()), 'lr_mult':10, 'decay_mult':2}]
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
        output_mean_list = []
        output_sigma_list = []
        for i in range(5):
            output = self.Crop_Net[i](feature_list[i])
            output = self.Crop_Net_mean[i](output)
            output_mean = self.GAP(output)
            output_sigma = self.Crop_Net_sigma[i](output)
            # output_mean = self.Crop_Net_mean[i](output)
            # output_sigma = self.Crop_Net_sigma[i](output)
            output_mean_list.append(output_mean)
            output_sigma_list.append(output_sigma)

        loc_feature = torch.stack(output_mean_list, dim=1)  # batch_size * 5 * 64 * 1 * 1
        loc_feature = loc_feature.view(batch_size, -1)  # batch_size * 320

        loc_feature_sigma = torch.stack(output_sigma_list, dim=1)  # batch_size * 5 * 64 * 1 * 1
        loc_feature_sigma = loc_feature.view(batch_size, -1)  # batch_size * 320

        return loc_feature, loc_feature_sigma


def IR_global_local_stoch_feat_384(numOfLayer):
    """Constructs a ir-18/ir-50 model."""

    model = Backbone_Global_Local_stoch_feat_384(numOfLayer)

    return model


class Backbone_onlyResNet50_stoch(nn.Module):
    def __init__(self, numOfLayer):
        super(Backbone_onlyResNet50_stoch, self).__init__()

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

        self.pool_layer = nn.AdaptiveAvgPool2d((1, 1))

        self.sigma_layer = nn.Conv2d(512,512,(7,7))
        self.sigma_layer.apply(init_weights)

    def forward(self, imgs, locations):
        featureMap = self.input_layer(imgs)

        featureMap1 = self.layer1(featureMap)  # Batch * 64 * 56 * 56
        featureMap2 = self.layer2(featureMap1)  # Batch * 128 * 28 * 28
        featureMap3 = self.layer3(featureMap2)  # Batch * 256 * 14 * 14
        featureMap4 = self.layer4(featureMap3)  # Batch * 512 * 7 * 7

        feature = torch.flatten(self.pool_layer(featureMap4),1)
        feature = feature.view(featureMap.size(0), -1)  # Batch * 512

        sigma = nn.Softplus()(torch.flatten(self.sigma_layer(featureMap4),1))
        return feature, sigma

    def output_num(self):
        return 512

    def get_parameters(self):
        parameter_list = [{"params": self.input_layer.parameters(), "lr_mult": 1, 'decay_mult': 2},
                          {"params": self.layer1.parameters(), "lr_mult": 1, 'decay_mult': 2},
                          {"params": self.layer2.parameters(), "lr_mult": 1, 'decay_mult': 2},
                          {"params": self.layer3.parameters(), "lr_mult": 1, 'decay_mult': 2},
                          {"params": self.layer4.parameters(), "lr_mult": 1, 'decay_mult': 2},
                          {"params": self.sigma_layer.parameters(), "lr_mult": 10, 'decay_mult': 2}, 
                          ]
        return parameter_list


def IR_onlyResNet50_stoch(numOfLayer):
    """Constructs a ir-18/ir-50 model."""

    model = Backbone_onlyResNet50_stoch(numOfLayer)

    return model
