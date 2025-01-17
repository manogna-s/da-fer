import torch
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential
import copy

from models.GraphConvolutionNetwork import GCNwithIntraAndInterMatrix
from models.GCN_utils import CountMeanOfFeature, CountMeanAndCovOfFeature, CountMeanOfFeatureInCluster
from models.ResNet_utils import bottleneck_IR, get_block, get_blocks, init_weights


# Support: ['IR_18', 'IR_50']
class Backbone_GCN(nn.Module):
    def __init__(self, numOfLayer, useIntraGCN=True, useInterGCN=True, useRandomMatrix=False, useAllOneMatrix=False, useCov=False, useCluster=False, class_num = 7):   

        super(Backbone_GCN, self).__init__()

        self.useIntraGCN = useIntraGCN
        self.useInterGCN = useInterGCN
        unit_module = bottleneck_IR
        
        self.input_layer = Sequential(Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1,1), padding=(1,1), bias=False),BatchNorm2d(64), PReLU(64))

        blocks = get_blocks(numOfLayer)
        self.layer1 = Sequential(*[unit_module(bottleneck.in_channel,bottleneck.depth,bottleneck.stride) for bottleneck in blocks[0]]) #get_block(in_channel=64, depth=64, num_units=3)])
        self.layer2 = Sequential(*[unit_module(bottleneck.in_channel,bottleneck.depth,bottleneck.stride) for bottleneck in blocks[1]]) #get_block(in_channel=64, depth=128, num_units=4)])
        self.layer3 = Sequential(*[unit_module(bottleneck.in_channel,bottleneck.depth,bottleneck.stride) for bottleneck in blocks[2]]) #get_block(in_channel=128, depth=256, num_units=14)])
        self.layer4 = Sequential(*[unit_module(bottleneck.in_channel,bottleneck.depth,bottleneck.stride) for bottleneck in blocks[3]]) #get_block(in_channel=256, depth=512, num_units=3)])

        self.output_layer = Sequential(nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)), 
                                       nn.ReLU(),
                                       nn.AdaptiveAvgPool2d((1,1)))

        cropNet_modules = []
        cropNet_blocks = [get_block(in_channel=128, depth=256, num_units=2), get_block(in_channel=256, depth=512, num_units=2)]
        for block in cropNet_blocks:
            for bottleneck in block:
                cropNet_modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        cropNet_modules+=[nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)), nn.ReLU()]
        self.Crop_Net = nn.ModuleList([ copy.deepcopy(nn.Sequential(*cropNet_modules)) for i in range(5) ])

        self.fc = nn.Linear(64 + 320, class_num)
        self.fc.apply(init_weights)

        self.loc_fc = nn.Linear(320, class_num)
        self.loc_fc.apply(init_weights)

        self.GAP = nn.AdaptiveAvgPool2d((1,1))

        if self.useIntraGCN and self.useInterGCN:
            self.GCN = GCNwithIntraAndInterMatrix(64, 128, 64, useIntraGCN=useIntraGCN, useInterGCN=useInterGCN, useRandomMatrix=useRandomMatrix, useAllOneMatrix=useAllOneMatrix)

            self.SourceMean = (CountMeanAndCovOfFeature(64+320) if useCov else CountMeanOfFeature(64+320)) if not useCluster else CountMeanOfFeatureInCluster(64+320, class_num=class_num)
            self.TargetMean = (CountMeanAndCovOfFeature(64+320) if useCov else CountMeanOfFeature(64+320)) if not useCluster else CountMeanOfFeatureInCluster(64+320, class_num=class_num)
 
            self.SourceBN = BatchNorm1d(64+320)
            self.TargetBN = BatchNorm1d(64+320)

    def classify(self, imgs, locations):

        featureMap = self.input_layer(imgs)

        featureMap1 = self.layer1(featureMap)  # Batch * 64 * 56 * 56
        featureMap2 = self.layer2(featureMap1) # Batch * 128 * 28 * 28
        featureMap3 = self.layer3(featureMap2) # Batch * 256 * 14 * 14
        featureMap4 = self.layer4(featureMap3) # Batch * 512 * class_num * class_num

        global_feature = self.output_layer(featureMap4).view(featureMap.size(0), -1) # Batch * 64
        loc_feature = self.crop_featureMap(featureMap2, locations)                   # Batch * 320
        feature = torch.cat((global_feature, loc_feature), 1)                        # Batch * (64+320)    
        
        # GCN
        if self.useIntraGCN and self.useInterGCN:
            if self.training:
                feature = self.SourceMean(feature)
            feature = torch.cat( ( self.SourceBN(feature), self.TargetBN(self.TargetMean.getSample(feature.detach())) ), 1) # Batch * (64+320 + 64+320)
            feature = self.GCN(feature.view(feature.size(0), 12, -1))                                                       # Batch * 12 * 64

        feature = feature.view(feature.size(0), -1).narrow(1, 0, 64+320) # Batch * (64+320)
        loc_feature = feature.narrow(1, 64, 320)                         # Batch * 320

        pred = self.fc(feature)              # Batch * class_num
        loc_pred = self.loc_fc(loc_feature)  # Batch * class_num

        return feature, pred, loc_pred

    def transfer(self, imgs, locations, domain='Target'):

        assert domain in ['Source', 'Target'], 'Parameter domain should be Source or Target.'

        featureMap = self.input_layer(imgs)

        featureMap1 = self.layer1(featureMap)  # Batch * 64 * 56 * 56
        featureMap2 = self.layer2(featureMap1) # Batch * 128 * 28 * 28
        featureMap3 = self.layer3(featureMap2) # Batch * 256 * 14 * 14
        featureMap4 = self.layer4(featureMap3) # Batch * 512 * class_num * class_num

        global_feature = self.output_layer(featureMap4).view(featureMap.size(0), -1)  # Batch * 64
        loc_feature = self.crop_featureMap(featureMap2, locations)                    # Batch * 320
        feature = torch.cat((global_feature, loc_feature), 1)                         # Batch * (64+320)

        if self.useIntraGCN and self.useInterGCN:
            if self.training:

            # Compute Feature
                SourceFeature = feature.narrow(0, 0, feature.size(0)//2)                  # Batch/2 * (64+320)
                TargetFeature = feature.narrow(0, feature.size(0)//2, feature.size(0)//2) # Batch/2 * (64+320)

                SourceFeature = self.SourceMean(SourceFeature) # Batch/2 * (64+320)
                TargetFeature = self.TargetMean(TargetFeature) # Batch/2 * (64+320)

                SourceFeature = self.SourceBN(SourceFeature)   # Batch/2 * (64+320)
                TargetFeature = self.TargetBN(TargetFeature)   # Batch/2 * (64+320)

                # Compute Mean
                SourceMean = self.SourceMean.getSample(TargetFeature.detach()) # Batch/2 * (64+320)
                TargetMean = self.TargetMean.getSample(SourceFeature.detach()) # Batch/2 * (64+320)

                SourceMean = self.SourceBN(SourceMean) # Batch/2 * (64+320)
                TargetMean = self.TargetBN(TargetMean) # Batch/2 * (64+320)

                # GCN
                feature = torch.cat( ( torch.cat((SourceFeature,TargetMean), 1), torch.cat((SourceMean,TargetFeature), 1) ), 0) # Batch * (64+320 + 64+320)
                feature = self.GCN(feature.view(feature.size(0), 12, -1))                                                       # Batch * 12 * 64

                feature = feature.view(feature.size(0), -1)                                                                     # Batch * (64+320 + 64+320)
                feature = torch.cat( (feature.narrow(0, 0, feature.size(0)//2).narrow(1, 0, 64+320), feature.narrow(0, feature.size(0)//2, feature.size(0)//2).narrow(1, 64+320, 64+320) ), 0) # Batch * (64+320)
                loc_feature = feature.narrow(1, 64, 320)                                                                        # Batch * 320

                pred = self.fc(feature)             # Batch * class_num
                loc_pred = self.loc_fc(loc_feature) # Batch * class_num

                return feature, pred, loc_pred

            # Inference
            if domain=='Source':
                SourceFeature = feature                                         # Batch * (64+320)
                TargetMean = self.TargetMean.getSample(SourceFeature.detach())  # Batch * (64+320)

                SourceFeature = self.SourceBN(SourceFeature)                    # Batch * (64+320)
                TargetMean = self.TargetBN(TargetMean)                          # Batch * (64+320)
            
                feature = torch.cat((SourceFeature,TargetMean), 1)              # Batch * (64+320 + 64+320)
                feature = self.GCN(feature.view(feature.size(0), 12, -1))       # Batch * 12 * 64

            elif domain=='Target':
                TargetFeature = feature                                         # Batch * (64+320)
                SourceMean = self.SourceMean.getSample(TargetFeature.detach())  # Batch * (64+320)

                SourceMean = self.SourceBN(SourceMean)                          # Batch * (64+320)
                TargetFeature = self.TargetBN(TargetFeature)                    # Batch * (64+320)

                feature = torch.cat((SourceMean,TargetFeature), 1)              # Batch * (64+320 + 64+320)
                feature = self.GCN(feature.view(feature.size(0), 12, -1))       # Batch * 12 * 64
            
            feature = feature.view(feature.size(0), -1)# Batch * (64+320 + 64+320)
            if domain=='Source':
                feature = feature.narrow(1, 0, 64+320)# Batch * (64+320)
            elif domain=='Target':   
                feature = feature.narrow(1, 64+320, 64+320)  # Batch * (64+320)
            loc_feature = feature.narrow(1, 64, 320)                         # Batch * 320

            pred = self.fc(feature)             # Batch * class_num
            loc_pred = self.loc_fc(loc_feature) # Batch * class_num

            return feature, pred, loc_pred

        feature = feature.view(feature.size(0), -1).narrow(1, 0, 64+320) # Batch * (64+320)
        loc_feature = feature.narrow(1, 64, 320)                         # Batch * 320

        pred = self.fc(feature)             # Batch * class_num
        loc_pred = self.loc_fc(loc_feature) # Batch * class_num

        return feature, pred, loc_pred

    def forward(self, imgs, locations, flag=True, domain='Target'):
        
        if flag:
            return self.classify(imgs, locations)

        return self.transfer(imgs, locations, domain)

    def output_num(self):
        return 64*6

    def get_parameters(self):
        if self.useIntraGCN and self.useInterGCN:
            parameter_list = [{"params":self.input_layer.parameters(), "lr_mult":1, 'decay_mult':2}, {"params":self.layer1.parameters(), "lr_mult":1, 'decay_mult':2}, {"params":self.layer2.parameters(), "lr_mult":1, 'decay_mult':2}, {"params":self.layer3.parameters(), "lr_mult":1, 'decay_mult':2}, {"params":self.layer4.parameters(), "lr_mult":1, 'decay_mult':2}, {"params":self.output_layer.parameters(), "lr_mult":10, 'decay_mult':2}, {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}, {"params":self.loc_fc.parameters(), "lr_mult":10, 'decay_mult':2}, {"params":self.Crop_Net.parameters(), "lr_mult":10, 'decay_mult':2}, {"params":self.GCN.parameters(), "lr_mult":10, 'decay_mult':2}, {"params":self.SourceBN.parameters(), "lr_mult":10, 'decay_mult':2}, {"params":self.TargetBN.parameters(), "lr_mult":10, 'decay_mult':2}]
        else:
            parameter_list = [{"params":self.input_layer.parameters(), "lr_mult":1, 'decay_mult':2}, {"params":self.layer1.parameters(), "lr_mult":1, 'decay_mult':2}, {"params":self.layer2.parameters(), "lr_mult":1, 'decay_mult':2}, {"params":self.layer3.parameters(), "lr_mult":1, 'decay_mult':2}, {"params":self.layer4.parameters(), "lr_mult":1, 'decay_mult':2}, {"params":self.output_layer.parameters(), "lr_mult":10, 'decay_mult':2}, {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}, {"params":self.loc_fc.parameters(), "lr_mult":10, 'decay_mult':2}, {"params":self.Crop_Net.parameters(), "lr_mult":10, 'decay_mult':2}]
        return parameter_list

    def crop_featureMap(self, featureMap, locations):
        batch_size = featureMap.size(0)
        map_ch = featureMap.size(1)
        map_len = featureMap.size(2)

        grid_ch = map_ch
        grid_len = 7 # 14, 6, 4

        feature_list = []
        for i in range(5):
            grid_list = []
            for j in range(batch_size):
                w_min = locations[j,i,0]-int(grid_len/2)
                w_max = locations[j,i,0]+int(grid_len/2)
                h_min = locations[j,i,1]-int(grid_len/2)
                h_max = locations[j,i,1]+int(grid_len/2)
                
                map_w_min = max(0, w_min)
                map_w_max = min(map_len-1, w_max)
                map_h_min = max(0, h_min)
                map_h_max = min(map_len-1, h_max)
                
                grid_w_min = max(0, 0-w_min)
                grid_w_max = grid_len + min(0, map_len-1-w_max)
                grid_h_min = max(0, 0-h_min)
                grid_h_max = grid_len + min(0, map_len-1-h_max)
                
                grid = torch.zeros(grid_ch, grid_len, grid_len)
                if featureMap.is_cuda:
                    grid = grid.cuda()

                grid[:, grid_h_min:grid_h_max+1, grid_w_min:grid_w_max+1] = featureMap[j, :, map_h_min:map_h_max+1, map_w_min:map_w_max+1] 

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
        loc_feature = loc_feature.view(batch_size, -1) # batch_size * 320 

        return loc_feature


def IR_GCN(numOfLayer, useIntraGCN, useInterGCN, useRandomMatrix, useAllOneMatrix, useCov, useCluster, class_num):
    """Constructs a ir-18/ir-50 model."""

    model = Backbone_GCN(numOfLayer, useIntraGCN, useInterGCN, useRandomMatrix, useAllOneMatrix, useCov, useCluster, class_num)

    return model
