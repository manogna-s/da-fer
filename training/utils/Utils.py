import os
import pandas as pd
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from models.AdversarialNetwork import RandomLayer, AdversarialNetwork
from models.ResNet import IR_global_local, IR_global
from models.ResNet_GCN import IR_GCN
from models.ResNet_utils import load_resnet_pretrained_weights
from utils.Dataset import MyDataset
from utils.misc_utils import *


def BuildModel(args):
    """Bulid Model."""

    if args.net == 'ResNet18':
        numOfLayer = 18
    elif args.net == 'ResNet50':
        numOfLayer = 50
    else:
        print('Add the model you want!')

    if args.use_gcn:
        model = IR_GCN(numOfLayer, args.intra_gcn, args.inter_gcn, args.rand_mat, args.all1_mat, args.use_cov,
                       args.use_cluster, args.class_num)
    else:
        if args.local_feat:
            model = IR_global_local(numOfLayer, args.class_num)
        else:
            model = IR_global(numOfLayer, args.class_num)

    if args.pretrained:
        model = load_resnet_pretrained_weights(model, numOfLayer)
    else:
        print('No Resume Model')

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.cuda()
    return model


def BuildAdversarialNetwork(args, model_output_num, class_num=7):
    """Bulid Adversarial Network."""

    if args.rand_layer:
        random_layer = RandomLayer([model_output_num, class_num], 1024)
        ad_net = AdversarialNetwork(1024, 512)
        random_layer.cuda()

    else:
        random_layer = None
        if args.dan_method == 'DANN' or args.dan_method == 'MME':
            ad_net = AdversarialNetwork(model_output_num, 128)
        else:
            ad_net = AdversarialNetwork(model_output_num * class_num, 512)

    ad_net.cuda()

    return random_layer, ad_net


def BuildDataloader(args, flag1='train', flag2='source', max_samples=-1):
    """Bulid data loader."""

    assert flag1 in ['train', 'test'], 'Function BuildDataloader : function parameter flag1 wrong.'
    assert flag2 in ['source', 'target'], 'Function BuildDataloader : function parameter flag2 wrong.'

    # Set Transform
    trans = transforms.Compose([
        transforms.Resize((args.face_scale, args.face_scale)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    target_trans = None

    dataPath_prefix = '../Dataset'

    data_imgs, data_labels, data_bboxs, data_landmarks = [], [], [], []
    if flag1 == 'train':
        if flag2 == 'source':
            if args.source == 'RAF':
                list_patition_label = pd.read_csv(dataPath_prefix + '/%s/lists/image_list.txt' % (args.source),
                                                  header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)
                n_samples = list_patition_label.shape[0]
                n_class = {0: 0, 1: 0}
                if max_samples == -1:
                    max_samples = n_samples
                for index in range(n_samples):
                    if list_patition_label[index, 0][:5] == "train":
                        if not os.path.exists(
                                dataPath_prefix + '/%s/boundingbox/' % (args.source) + list_patition_label[index, 0][
                                                                                       :-4] + '_boundingbox' + '.txt'):
                            continue
                        if not os.path.exists(
                                dataPath_prefix + '/%s/landmarks_5/' % (args.source) + list_patition_label[index, 0][
                                                                                       :-4] + '.txt'):
                            continue
                        bbox = np.loadtxt(
                            dataPath_prefix + '/%s/boundingbox/' % (args.source) + list_patition_label[index, 0][
                                                                                   :-4] + '_boundingbox.txt').astype(
                            np.int)
                        landmark = np.loadtxt(
                            dataPath_prefix + '/%s/landmarks_5/' % (args.source) + list_patition_label[index, 0][
                                                                                   :-3] + 'txt').astype(np.int)
                        label = list_patition_label[index, 1]
                        if n_class[label] < max_samples:
                            data_imgs.append(
                                dataPath_prefix + '/%s/images/' % (args.source) + list_patition_label[index, 0])
                            data_labels.append(label)
                            data_bboxs.append(bbox)
                            data_landmarks.append(landmark)
                            n_class[label] += 1

            if args.source == 'RAF_7class':
                list_patition_label = pd.read_csv(
                    dataPath_prefix + '/RAF_7class/basic/EmoLabel/list_patition_label.txt', header=None,
                    delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)
                for index in range(list_patition_label.shape[0]):
                    if list_patition_label[index, 0][:5] == "train":
                        if not os.path.exists(
                                dataPath_prefix + '/RAF_7class/basic/Annotation/boundingbox/' + list_patition_label[
                                                                                                    index, 0][
                                                                                                :-4] + '_boundingbox' + '.txt'):
                            continue
                        if not os.path.exists(
                                dataPath_prefix + '/RAF_7class/basic/Annotation/Landmarks_5/' + list_patition_label[
                                                                                                    index, 0][
                                                                                                :-4] + '.txt'):
                            continue
                        bbox = np.loadtxt(
                            dataPath_prefix + '/RAF_7class/basic/Annotation/boundingbox/' + list_patition_label[
                                                                                                index, 0][
                                                                                            :-4] + '_boundingbox.txt').astype(
                            np.int)
                        landmark = np.loadtxt(
                            dataPath_prefix + '/RAF_7class/basic/Annotation/Landmarks_5/' + list_patition_label[
                                                                                                index, 0][
                                                                                            :-3] + 'txt').astype(np.int)

                        data_imgs.append(
                            dataPath_prefix + '/RAF_7class/basic/Image/original/' + list_patition_label[index, 0])
                        data_labels.append(list_patition_label[index, 1] - 1)
                        data_bboxs.append(bbox)
                        data_landmarks.append(landmark)

        if flag2 == 'target':
            if args.target == 'AISIN':
                list_patition_label = pd.read_csv(dataPath_prefix + '/%s/lists/image_list.txt' % (args.target),
                                                  header=None,
                                                  delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)
                n_class = {0: 0, 1: 0}
                n_samples = list_patition_label.shape[0]
                if max_samples == -1:
                    max_samples = n_samples
                for index in range(n_samples):
                    if list_patition_label[index, 0][:5] == "train":
                        if not os.path.exists(
                                dataPath_prefix + '/%s/landmarks_5/' % (args.target) + list_patition_label[index, 0][
                                                                                       :-3] + 'txt'):
                            continue
                        img = Image.open(
                            dataPath_prefix + '/%s/images/' % (args.target) + list_patition_label[index, 0]).convert(
                            'RGB')
                        ori_img_w, ori_img_h = img.size
                        landmark = np.loadtxt(
                            dataPath_prefix + '/%s/landmarks_5/' % (args.target) + list_patition_label[index, 0][
                                                                                   :-3] + 'txt').astype(np.int)
                        label = list_patition_label[index, 1]
                        if n_class[label] < max_samples:
                            data_imgs.append(
                                dataPath_prefix + '/%s/images/' % (args.target) + list_patition_label[index, 0])
                            data_labels.append(label)
                            data_bboxs.append((0, 0, ori_img_w, ori_img_h))
                            data_landmarks.append(landmark)
                            n_class[label]+=1

            if args.target == 'JAFFE':
                list_patition_label = pd.read_csv(dataPath_prefix + '/JAFFE/list/list_putao.txt', header=None,
                                                  delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):

                    if not os.path.exists(
                            dataPath_prefix + '/JAFFE/annos/bbox/' + list_patition_label[index, 0][:-4] + 'txt'):
                        continue
                    if not os.path.exists(
                            dataPath_prefix + '/JAFFE/annos/landmark_5/' + list_patition_label[index, 0][:-4] + 'txt'):
                        continue

                    bbox = np.loadtxt(
                        dataPath_prefix + '/JAFFE/annos/bbox/' + list_patition_label[index, 0][:-4] + 'txt').astype(
                        np.int)
                    landmark = np.loadtxt(
                        dataPath_prefix + '/JAFFE/annos/landmark_5/' + list_patition_label[index, 0][
                                                                       :-4] + 'txt').astype(
                        np.int)

                    data_imgs.append(dataPath_prefix + '/JAFFE/images/' + list_patition_label[index, 0])
                    data_labels.append(list_patition_label[index, 1])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)

    elif flag1 == 'test':
        if flag2 == 'source':
            if args.source == 'RAF':
                list_patition_label = pd.read_csv(dataPath_prefix + '/%s/lists/image_list.txt' % (args.source),
                                                  header=None, delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)
                for index in range(list_patition_label.shape[0]):
                    if list_patition_label[index, 0][:4] == "test":
                        if not os.path.exists(
                                dataPath_prefix + '/%s/boundingbox/' % (args.source) + list_patition_label[index, 0][
                                                                                       :-4] + '_boundingbox.txt'):
                            continue
                        if not os.path.exists(
                                dataPath_prefix + '/%s/landmarks_5/' % (args.source) + list_patition_label[index, 0][
                                                                                       :-3] + 'txt'):
                            continue

                        bbox = np.loadtxt(
                            dataPath_prefix + '/%s/boundingbox/' % (args.source) + list_patition_label[index, 0][
                                                                                   :-4] + '_boundingbox.txt').astype(
                            np.int)
                        landmark = np.loadtxt(
                            dataPath_prefix + '/%s/landmarks_5/' % (args.source) + list_patition_label[index, 0][
                                                                                   :-3] + 'txt').astype(np.int)
                        data_imgs.append(
                            dataPath_prefix + '/%s/images/' % (args.source) + list_patition_label[index, 0])
                        data_labels.append(list_patition_label[index, 1])
                        data_bboxs.append(bbox)
                        data_landmarks.append(landmark)

            if args.source == 'RAF_7class':
                list_patition_label = pd.read_csv(
                    dataPath_prefix + '/RAF_7class/basic/EmoLabel/list_patition_label.txt', header=None,
                    delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)
                for index in range(list_patition_label.shape[0]):
                    if list_patition_label[index, 0][:4] == "test":
                        if not os.path.exists(
                                dataPath_prefix + '/RAF_7class/basic/Annotation/boundingbox/' + list_patition_label[
                                                                                                    index, 0][
                                                                                                :-4] + '_boundingbox' + '.txt'):
                            continue
                        if not os.path.exists(
                                dataPath_prefix + '/RAF_7class/basic/Annotation/Landmarks_5/' + list_patition_label[
                                                                                                    index, 0][
                                                                                                :-4] + '.txt'):
                            continue
                        bbox = np.loadtxt(
                            dataPath_prefix + '/RAF_7class/basic/Annotation/boundingbox/' + list_patition_label[
                                                                                                index, 0][
                                                                                            :-4] + '_boundingbox.txt').astype(
                            np.int)
                        landmark = np.loadtxt(
                            dataPath_prefix + '/RAF_7class/basic/Annotation/Landmarks_5/' + list_patition_label[
                                                                                                index, 0][
                                                                                            :-3] + 'txt').astype(np.int)

                        data_imgs.append(
                            dataPath_prefix + '/RAF_7class/basic/Image/original/' + list_patition_label[index, 0])
                        data_labels.append(list_patition_label[index, 1] - 1)
                        data_bboxs.append(bbox)
                        data_landmarks.append(landmark)

        elif flag2 == 'target':
            if args.target == 'AISIN':
                list_patition_label = pd.read_csv(dataPath_prefix + '/%s/lists/image_list.txt' % (args.target),
                                                  header=None,
                                                  delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)
                for index in range(list_patition_label.shape[0]):
                    if list_patition_label[index, 0][:4] == "test":
                        if not os.path.exists(
                                dataPath_prefix + '/%s/landmarks_5/' % (args.target) + list_patition_label[index, 0][
                                                                                       :-3] + 'txt'):
                            continue
                        img = Image.open(
                            dataPath_prefix + '/%s/images/' % (args.target) + list_patition_label[index, 0]).convert(
                            'RGB')
                        ori_img_w, ori_img_h = img.size
                        landmark = np.loadtxt(
                            dataPath_prefix + '/%s/landmarks_5/' % (args.target) + list_patition_label[index, 0][
                                                                                   :-3] + 'txt').astype(np.int)

                        data_imgs.append(
                            dataPath_prefix + '/%s/images/' % (args.target) + list_patition_label[index, 0])
                        data_labels.append(list_patition_label[index, 1])
                        data_bboxs.append((0, 0, ori_img_w, ori_img_h))
                        data_landmarks.append(landmark)

            if args.target == 'JAFFE':
                list_patition_label = pd.read_csv(dataPath_prefix + '/JAFFE/list/list_putao.txt', header=None,
                                                  delim_whitespace=True)
                list_patition_label = np.array(list_patition_label)

                for index in range(list_patition_label.shape[0]):
                    if not os.path.exists(
                            dataPath_prefix + '/JAFFE/annos/bbox/' + list_patition_label[index, 0][:-4] + 'txt'):
                        continue
                    if not os.path.exists(
                            dataPath_prefix + '/JAFFE/annos/landmark_5/' + list_patition_label[index, 0][:-4] + 'txt'):
                        continue

                    bbox = np.loadtxt(
                        dataPath_prefix + '/JAFFE/annos/bbox/' + list_patition_label[index, 0][:-4] + 'txt').astype(
                        np.int)
                    landmark = np.loadtxt(
                        dataPath_prefix + '/JAFFE/annos/landmark_5/' + list_patition_label[index, 0][
                                                                       :-4] + 'txt').astype(
                        np.int)

                    data_imgs.append(dataPath_prefix + '/JAFFE/images/' + list_patition_label[index, 0])
                    data_labels.append(list_patition_label[index, 1])
                    data_bboxs.append(bbox)
                    data_landmarks.append(landmark)

    # DataSet Distribute
    distribute_ = np.array(data_labels)
    print(' %s %s dataset qty: %d' % (flag1, flag2, len(data_imgs)))
    dataset_dist = []
    for i in range(args.class_num):
        dataset_dist.append(np.sum(distribute_ == i))

    print("Dataset Distribution for %s classes is: " % (args.class_num), dataset_dist)

    # DataSet
    data_set = MyDataset(data_imgs, data_labels, data_bboxs, data_landmarks, flag1, trans, target_trans)

    # DataLoader
    if flag1 == 'train':
        data_loader = data.DataLoader(dataset=data_set, batch_size=args.train_batch, shuffle=True, num_workers=8,
                                      drop_last=True)
    elif flag1 == 'test':
        data_loader = data.DataLoader(dataset=data_set, batch_size=args.test_batch, shuffle=False, num_workers=8,
                                      drop_last=False)

    return data_loader



