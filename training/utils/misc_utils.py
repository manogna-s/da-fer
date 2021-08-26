import argparse
import os
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase


class MarkerHandler(HandlerBase):
    def create_artists(self, legend, tup,xdescent, ydescent,
                        width, height, fontsize,trans):
        return [plt.Line2D([width/2], [height/2.],ls="",
                       marker=tup[1],color=tup[0], transform=trans)]

class AverageMeter(object):
    '''Computes and stores the sum, count and average'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        self.val = val
        self.sum += val
        self.count += count

        if self.count == 0:
            self.avg = 0
        else:
            self.avg = float(self.sum) / self.count


def str2bool(input):
    if isinstance(input, bool):
        return input
    if input.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif input.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def Set_Param_Optim(args, model):
    """Set Parameters for optimization."""

    if isinstance(model, nn.DataParallel):
        return model.module.get_parameters()

    return model.get_parameters()


def Set_Optimizer(args, parameter_list, lr=0.001, weight_decay=0.0005, momentum=0.9):
    """Set Optimizer."""

    return optim.SGD(parameter_list, lr=lr, weight_decay=weight_decay, momentum=momentum)


def lr_scheduler(optimizer, iter_num, gamma, power, lr=0.001, weight_decay=0.0005):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""

    lr = lr * (1 + gamma * iter_num) ** (-power)

    for param_group in optimizer.param_groups:

        if 'lr_mult' in param_group:
            param_group['lr'] = lr * param_group['lr_mult']
        else:
            param_group['lr'] = lr

        if 'decay_mult' in param_group:
            param_group['weight_decay'] = weight_decay * param_group['decay_mult']
        else:
            param_group['weight_decay'] = weight_decay

    return optimizer, lr


def lr_scheduler_withoutDecay(optimizer, lr=0.001, weight_decay=0.0005):
    """Learning rate without Decay."""

    for param_group in optimizer.param_groups:

        if 'lr_mult' in param_group:
            param_group['lr'] = lr * param_group['lr_mult']
        else:
            param_group['lr'] = lr

        if 'decay_mult' in param_group:
            param_group['weight_decay'] = weight_decay * param_group['decay_mult']
        else:
            param_group['weight_decay'] = weight_decay

    return optimizer, lr


def Compute_Accuracy(args, pred, target, acc, prec, recall):
    '''Compute the accuracy of all samples, the accuracy of positive samples, the recall of positive samples.'''

    pred = pred.cpu().data.numpy()
    pred = np.argmax(pred, axis=1)
    target = target.cpu().data.numpy()

    pred = pred.astype(np.int32).reshape(pred.shape[0], )
    target = target.astype(np.int32).reshape(target.shape[0], )

    for i in range(args.class_num):
        TP = np.sum((pred == i) * (target == i))
        TN = np.sum((pred != i) * (target != i))

        # Compute Accuracy of All --> TP+TN / All
        acc[i].update(np.sum(pred == target), pred.shape[0])

        # Compute Precision of Positive --> TP/(TP+FP)
        prec[i].update(TP, np.sum(pred == i))

        # Compute Recall of Positive --> TP/(TP+FN)
        recall[i].update(TP, np.sum(target == i))


def Show_Accuracy(acc, prec, recall, class_num=7):
    """Compute average of accuaracy/precision/recall/f1"""

    # Compute F1 value
    f1 = [AverageMeter() for i in range(class_num)]
    for i in range(class_num):
        if prec[i].avg == 0 or recall[i].avg == 0:
            f1[i].avg = 0
            continue
        f1[i].avg = 2 * prec[i].avg * recall[i].avg / (prec[i].avg + recall[i].avg)

    # Compute average of accuaracy/precision/recall/f1
    acc_avg, prec_avg, recall_avg, f1_avg = 0, 0, 0, 0

    for i in range(class_num):
        acc_avg += acc[i].avg
        prec_avg += prec[i].avg
        recall_avg += recall[i].avg
        f1_avg += f1[i].avg

    acc_avg, prec_avg, recall_avg, f1_avg = acc_avg / class_num, prec_avg / class_num, recall_avg / class_num, f1_avg / class_num

    # Log Accuracy Infomation
    Accuracy_Info = ''

    Accuracy_Info += '    Accuracy'
    for i in range(class_num):
        Accuracy_Info += ' {:.4f}'.format(acc[i].avg)
    Accuracy_Info += '\n'

    Accuracy_Info += '    Precision'
    for i in range(class_num):
        Accuracy_Info += ' {:.4f}'.format(prec[i].avg)
    Accuracy_Info += '\n'

    Accuracy_Info += '    Recall'
    for i in range(class_num):
        Accuracy_Info += ' {:.4f}'.format(recall[i].avg)
    Accuracy_Info += '\n'

    Accuracy_Info += '    F1'
    for i in range(class_num):
        Accuracy_Info += ' {:.4f}'.format(f1[i].avg)
    Accuracy_Info += '\n'

    LoggerInfo = Accuracy_Info
    LoggerInfo += '''    Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f}''' \
        .format(acc_avg, prec_avg, recall_avg, f1_avg)
    print(LoggerInfo)

    return Accuracy_Info, acc_avg, prec_avg, recall_avg, f1_avg


def Visualization(args, figName, model, dataloader, useClassify=True, domain='Source'):
    '''Feature Visualization in Source/Target Domain.'''

    assert useClassify in [True, False], 'useClassify should be bool.'
    assert domain in ['Source', 'Target'], 'domain should be source or target.'

    model.eval()

    Feature, Label = [], []

    # Get Cluster
    for i in range(7):
        if domain == 'Source':
            Feature.append(model.SourceMean.running_mean[i].cpu().data.numpy())
        elif domain == 'Target':
            Feature.append(model.TargetMean.running_mean[i].cpu().data.numpy())
    Label.append(np.array([7 for i in range(7)]))

    # Get Feature and Label
    for step, (input, landmark, label) in enumerate(dataloader):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        with torch.no_grad():
            feature, output, loc_output = model(input, landmark, useClassify, domain)
        Feature.append(feature.cpu().data.numpy())
        Label.append(label.cpu().data.numpy())

    Feature = np.vstack(Feature)
    Label = np.concatenate(Label)

    # Using T-SNE
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=50, early_exaggeration=3)
    embedding = tsne.fit_transform(Feature)

    # Draw Visualization of Feature
    colors = {0: 'red', 1: 'blue', 2: 'olive', 3: 'green', 4: 'orange', 5: 'purple', 6: 'darkslategray', 7: 'black'}
    # labels = {0:'Surprised', 1:'Fear', 2:'Disgust',  3:'Happy',  4:'Sad',  5:'Angry',  6:'Neutral', 7:'Cluster'}
    labels = {0: '惊讶', 1: '恐惧', 2: '厌恶', 3: '开心', 4: '悲伤', 5: '愤怒', 6: '平静', 7: '聚类中心'}

    data_min, data_max = np.min(embedding, 0), np.max(embedding, 0)
    data_norm = (embedding - data_min) / (data_max - data_min)

    fig = plt.figure()
    ax = plt.subplot(111)

    for i in range(7):
        data_x, data_y = data_norm[Label == i][:, 0], data_norm[Label == i][:, 1]
        scatter = plt.scatter(data_x, data_y, c='', edgecolors=colors[i], s=5, label=labels[i], marker='^', alpha=0.6)
    scatter = plt.scatter(data_norm[Label == 7][:, 0], data_norm[Label == 7][:, 1], c=colors[7], s=20, label=labels[7],
                          marker='^', alpha=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])

    plt.legend(handles=[mpatches.Patch(color=colors[i], label="{:s}".format(labels[i])) for i in range(8)],
               loc='upper left',
               # prop = {'size':8},
               prop=matplotlib.font_manager.FontProperties(fname='./simhei.ttf'),
               bbox_to_anchor=(1.05, 0.85),
               borderaxespad=0)
    plt.savefig(fname='{}'.format(figName), format="pdf", bbox_inches='tight')


def VisualizationForTwoDomain(args, figName, model, source_dataloader, target_dataloader, useClassify=True,
                              showClusterCenter=True):
    '''Feature Visualization in Source and Target Domain.'''

    model.eval()

    Feature_Source, Label_Source, Feature_Target, Label_Target = [], [], [], []

    # Get Feature and Label in Source Domain
    if showClusterCenter:
        for i in range(7):
            Feature_Source.append(model.SourceMean.running_mean[i].cpu().data.numpy())
        Label_Source.append(np.array([7 for i in range(7)]))

    for step, (input, landmark, label) in enumerate(source_dataloader):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        with torch.no_grad():
            feature, output, loc_output = model(input, landmark, useClassify, domain='Source')

        Feature_Source.append(feature.cpu().data.numpy())
        Label_Source.append(label.cpu().data.numpy())

    Feature_Source = np.vstack(Feature_Source)
    Label_Source = np.concatenate(Label_Source)

    # Get Feature and Label in Target Domain
    if showClusterCenter:
        for i in range(7):
            Feature_Target.append(model.TargetMean.running_mean[i].cpu().data.numpy())
        Label_Target.append(np.array([7 for i in range(7)]))

    for step, (input, landmark, label) in enumerate(target_dataloader):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        with torch.no_grad():
            feature, output, loc_output = model(input, landmark, useClassify, domain='Target')

        Feature_Target.append(feature.cpu().data.numpy())
        Label_Target.append(label.cpu().data.numpy())

    Feature_Target = np.vstack(Feature_Target)
    Label_Target = np.concatenate(Label_Target)

    # Sampling from Source Domain
    Feature_Temple, Label_Temple = [], []
    for i in range(8):
        num_source = np.sum(Label_Source == i)
        num_target = np.sum(Label_Target == i)

        num = num_source if num_source <= num_target else num_target

        Feature_Temple.append(Feature_Source[Label_Source == i][:num])
        Label_Temple.append(Label_Source[Label_Source == i][:num])

    Feature_Source = np.vstack(Feature_Temple)
    Label_Source = np.concatenate(Label_Temple)

    Label_Target += 8

    Feature = np.vstack((Feature_Source, Feature_Target))
    Label = np.concatenate((Label_Source, Label_Target))

    # Using T-SNE
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=50, early_exaggeration=3)
    embedding = tsne.fit_transform(Feature)

    # Draw Visualization of Feature
    colors = {0: 'firebrick', 1: 'aquamarine', 2: 'goldenrod', 3: 'cadetblue', 4: 'saddlebrown', 5: 'yellowgreen',
              6: 'navy'}
    labels = {0: 'Surprised', 1: 'Fear', 2: 'Disgust', 3: 'Happy', 4: 'Sad', 5: 'Angry', 6: 'Neutral'}

    data_min, data_max = np.min(embedding, 0), np.max(embedding, 0)
    data_norm = (embedding - data_min) / (data_max - data_min)

    fig = plt.figure()
    ax = plt.subplot(111)

    for i in range(7):

        data_source_x, data_source_y = data_norm[Label == i][:, 0], data_norm[Label == i][:, 1]
        source_scatter = plt.scatter(data_source_x, data_source_y, color="none", edgecolor=colors[i], s=20,
                                     label=labels[i], marker="o", alpha=0.4, linewidth=0.5)

        data_target_x, data_target_y = data_norm[Label == (i + 8)][:, 0], data_norm[Label == (i + 8)][:, 1]
        target_scatter = plt.scatter(data_target_x, data_target_y, color=colors[i], edgecolor="none", s=30,
                                     label=labels[i], marker="x", alpha=0.6, linewidth=0.2)

        if i == 0:
            source_legend = source_scatter
            target_legend = target_scatter

    if showClusterCenter:
        source_cluster = plt.scatter(data_norm[Label == 7][:, 0], data_norm[Label == 7][:, 1], c='black', s=20,
                                     label='Source Cluster Center', marker='^', alpha=1)
        target_cluster = plt.scatter(data_norm[Label == 15][:, 0], data_norm[Label == 15][:, 1], c='black', s=20,
                                     label='Target Cluster Center', marker='s', alpha=1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])
    plt.savefig(fname='{}.pdf'.format(figName), format="pdf", bbox_inches='tight')


def VizFeatures(args, epoch, model, dataloaders):
    Visualization('{}_Source.pdf'.format(epoch), model, dataloaders['train_source'], useClassify=False,
                  domain='Source')
    Visualization('{}_Target.pdf'.format(epoch), model, dataloaders['train_target'], useClassify=False,
                  domain='Target')

    VisualizationForTwoDomain('{}_train'.format(epoch), model, dataloaders['train_source'], dataloaders['train_target'],
                              useClassify=args.useClassify, showClusterCenter=False)
    VisualizationForTwoDomain('{}_test'.format(epoch), model, dataloaders['test_source'], dataloaders['test_target'],
                              useClassify=args.useClassify, showClusterCenter=False)
    return

def viz_tsne(args, Feature, Label):
    # Using T-SNE
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=30, early_exaggeration=10)
    embedding = tsne.fit_transform(Feature)

    # Draw Visualization of Feature
    colors = {0: 'firebrick', 1: 'aquamarine', 2: 'goldenrod', 3: 'cadetblue', 4: 'saddlebrown', 5: 'yellowgreen',
              6: 'navy'}
    labels = {0: 'Surprised', 1: 'Fear', 2: 'Disgust', 3: 'Happy', 4: 'Sad', 5: 'Angry', 6: 'Neutral'}

    data_min, data_max = np.min(embedding, 0), np.max(embedding, 0)
    data_norm = (embedding - data_min) / (data_max - data_min)

    fig = plt.figure()
    ax = plt.subplot(111)

    for i in range(7):
        data_source_x, data_source_y = data_norm[Label == i+14][:, 0], data_norm[Label == i+14][:, 1]
        source_scatter = plt.scatter(data_source_x, data_source_y, color="none", edgecolor=colors[i], s=5,
                                     label=labels[i], marker="o", alpha=0.4, linewidth=0.5)

    for i in range(7):
        data_source_x, data_source_y = data_norm[Label == i][:, 0], data_norm[Label == i][:, 1]
        source_test_scatter = plt.scatter(data_source_x, data_source_y, color=colors[i], edgecolor='black', s=15,
                                     label=labels[i], marker="X", alpha=0.4, linewidth=0.5)
    ax.legend(bbox_to_anchor = (1.05, 0.6))
    
    for i in range(7):
        data_target_x, data_target_y = data_norm[Label == (i + 7)][:, 0], data_norm[Label == (i + 7)][:, 1]
        target_scatter = plt.scatter(data_target_x, data_target_y, color=colors[i], edgecolor='black', s=15,
                                     label=labels[i], marker="D", alpha=0.6, linewidth=0.3)

    plt.title(f'{args.log}_tsne')

    list_color  = ["c", "gold", "crimson"]
    list_mak    = ["o","x","D"]
    list_lab    = ['source train','source test','target']

    ax.legend(list(zip(list_color,list_mak)), list_lab, 
          handler_map={tuple:MarkerHandler()}, bbox_to_anchor = (1.05, 0.6))

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])
    plt.savefig(fname=os.path.join(args.out, f'{args.log}_tsne.pdf'), format="pdf", bbox_inches='tight')
    return