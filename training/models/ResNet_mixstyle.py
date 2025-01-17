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
from models.mixstyle import MixStyle, MixStyle_Cls

class GradReverse(Function):
    @staticmethod
    def forward(self, x):
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        return (grad_output * -1.0)


def grad_reverse(x):
    return GradReverse().apply(x)


class ResClassifier(nn.Module):
    def __init__(self, num_classes=7,num_layer=2,num_unit=384,prob=0.5,middle=100):
        super(ResClassifier, self).__init__()
        layers = []
        if num_layer == 1:
            layers.append(nn.Linear(num_unit, num_classes))
        
        if num_layer > 1:
            layers.append(nn.Dropout(p=prob))
            layers.append(nn.Linear(num_unit,middle))
            layers.append(nn.BatchNorm1d(middle,affine=True))
            layers.append(nn.ReLU(inplace=True))

            for i in range(num_layer-1):
                layers.append(nn.Dropout(p=prob))
                layers.append(nn.Linear(middle,middle))
                layers.append(nn.BatchNorm1d(middle,affine=True))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(middle,num_classes))
        self.classifier = nn.Sequential(*layers)
        self.classifier.apply(init_weights)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x)
        x = self.classifier(x)
        return x

class StochasticClassifier(nn.Module):
    def __init__(self, args, input_dim=384):
        super(StochasticClassifier, self).__init__()
        
        self.stoch_bias=args.use_stoch_bias
        self.classifiers = None
        self.weight = torch.zeros((args.class_num, input_dim))

        self.weight_mu = Parameter(torch.empty_like(self.weight, requires_grad=True))
        self.weight_rho = Parameter(torch.empty_like(self.weight, requires_grad=True))

        if self.stoch_bias:
            print('Using stochastic bias')
            self.bias = torch.zeros(args.class_num)
            self.bias_mu = Parameter(torch.empty_like(self.bias, requires_grad=True))
            self.bias_rho = Parameter(torch.empty_like(self.bias, requires_grad=True))
        else:
            self.bias = Parameter(torch.zeros(args.class_num))

        self.fc.apply(init_weights)
        nn.init.xavier_normal_(self.weight_mu)
        nn.init.constant_(self.weight_rho, -args.var_rho)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_rho, -args.var_rho)


    def reparameterize(self, sample=False):
        weight_std =  torch.log(1+torch.exp(self.weight_rho)) 
        bias_std = torch.log(1+torch.exp(self.bias_rho)) 

        if self.training or sample:
            weight_eps = torch.randn_like(weight_std)
            bias_eps = torch.randn_like(bias_std)
        else:
            weight_eps = torch.zeros_like(weight_std)
            bias_eps = torch.zeros_like(bias_std)

        self.weight = self.weight_mu + weight_eps * weight_std
        
        if self.stoch_bias:
            self.bias = self.bias_mu + bias_eps * bias_std
        return

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x)
        self.reparameterize()
        out = F.linear(x, self.weight, self.bias)
        return out

    def eval_n(self, x, n=5):

        if self.classifiers is None:
            self.classifiers = []
            for i in range(n):
                self.reparameterize(sample=True)
                self.classifiers.append({'weight':self.weight, 'bias':self.bias})
            # print(self.weight_mu, self.bias_mu)
            # print(self.classifiers)
        
        preds = []
        ent = []
        for i in range(n):
            out = F.linear(x, self.classifiers[i]['weight'], self.classifiers[i]['bias'])
            probs = torch.softmax(out[0], dim=-1).cpu().data.numpy()
            ent.append(-np.sum(probs * np.log(probs)))
            out = out.cpu().data.numpy()
            pred = np.argmax(out, axis=1)
            preds.append(pred[0])
            # print(i, (probs*100).astype(int), pred)
        return preds, ent
        
# Support: ['IR_18', 'IR_50']
class Backbone_Global_Local_feat_mixstyle(nn.Module):
    def __init__(self, numOfLayer):

        super(Backbone_Global_Local_feat_mixstyle, self).__init__()

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
        self.fc = nn.Sequential(nn.Linear(384,384), nn.ReLU())
        self.mixstyle = MixStyle_Cls(p=0.5, alpha=0.1, mix='sourcedomain')


    def classify(self, imgs, locations, labels):

        featureMap = self.input_layer(imgs)

        featureMap1 = self.layer1(featureMap)  # Batch * 64 * 56 * 56
        featureMap1 = self.mixstyle(featureMap1, labels)
        featureMap2 = self.layer2(featureMap1)  # Batch * 128 * 28 * 28
        featureMap2 = self.mixstyle(featureMap2, labels)
        featureMap3 = self.layer3(featureMap2)  # Batch * 256 * 14 * 14
        featureMap4 = self.layer4(featureMap3)  # Batch * 512 * class_num * class_num

        global_feature = self.output_layer(featureMap4).view(featureMap.size(0), -1)  # Batch * 64
        loc_feature = self.crop_featureMap(featureMap2, locations)  # Batch * 320
        feature = torch.cat((global_feature, loc_feature), 1)  # Batch * (64+320)

        feature = feature.view(feature.size(0), -1).narrow(1, 0, 64 + 320)  # Batch * (64+320)
        feature = self.fc(feature)
        return feature

    def forward(self, imgs, locations, labels, useClassify=True):

        return self.classify(imgs, locations, labels)

    def output_num(self):
        return 64 * 6

    def get_parameters(self):
        parameter_list = [{'params':list(self.input_layer.parameters())+list(self.layer1.parameters())+list(self.layer2.parameters())+
                            list(self.layer3.parameters())+list(self.layer4.parameters()), 'lr_mult':0.1, 'decay_mult':2},
                          {'params':list(self.output_layer.parameters())+list(self.Crop_Net.parameters()), 'lr_mult':1, 'decay_mult':2}]
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


def IR_global_local_feat_mixstyle(numOfLayer):
    """Constructs a ir-18/ir-50 model."""

    model = Backbone_Global_Local_feat_mixstyle(numOfLayer)

    return model


class Backbone_onlyResNet50(nn.Module):
    def __init__(self, numOfLayer):
        super(Backbone_onlyResNet50, self).__init__()

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

    def forward(self, imgs, locations):
        featureMap = self.input_layer(imgs)

        featureMap1 = self.layer1(featureMap)  # Batch * 64 * 56 * 56
        featureMap2 = self.layer2(featureMap1)  # Batch * 128 * 28 * 28
        featureMap3 = self.layer3(featureMap2)  # Batch * 256 * 14 * 14
        featureMap4 = self.layer4(featureMap3)  # Batch * 512 * 7 * 7

        feature = torch.flatten(self.pool_layer(featureMap4),1)
        feature = feature.view(featureMap.size(0), -1)  # Batch * 512
        return feature

    def output_num(self):
        return 512

    def get_parameters(self):
        parameter_list = [{"params": self.input_layer.parameters(), "lr_mult": 1, 'decay_mult': 2},
                          {"params": self.layer1.parameters(), "lr_mult": 1, 'decay_mult': 2},
                          {"params": self.layer2.parameters(), "lr_mult": 1, 'decay_mult': 2},
                          {"params": self.layer3.parameters(), "lr_mult": 1, 'decay_mult': 2},
                          {"params": self.layer4.parameters(), "lr_mult": 1, 'decay_mult': 2},
                          ]
        return parameter_list


def IR_onlyResNet50(numOfLayer):
    """Constructs a ir-18/ir-50 model."""

    model = Backbone_onlyResNet50(numOfLayer)

    return model