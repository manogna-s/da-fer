import os
import pandas as pd
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from models.AdversarialNetwork import RandomLayer, AdversarialNetwork
from models.ResNet import IR_global_local, IR_global
from models.ResNet_GCN import IR_GCN
from models.ResNet_feat import IR_global_local_feat, IR_onlyResNet50
from models.ResNet_mixstyle import IR_global_local_feat_mixstyle
from models.ResNet_stoch_feat import IR_global_local_stoch_feat, IR_onlyResNet50_stoch, IR_global_local_stoch_feat_384
from models.ResNet_utils import load_resnet_pretrained_weights
from utils.Dataset import MyDataset
from utils.misc_utils import *
from utils.randaugument import * 
import json

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
    elif args.use_mcd or args.use_star:
        if args.local_feat:
            if args.use_mixstyle:
                print('Using mixstyle')
                model = IR_global_local_feat_mixstyle(numOfLayer)
            else:
                model = IR_global_local_feat(numOfLayer)
        else:
            print('MCD with only global feat not yet added')
    elif args.use_stoch_feats:
        if args.n_hidden == 384:
            print('384 model')
            model = IR_global_local_stoch_feat_384(numOfLayer) 
        elif args.local_feat:
            model = IR_global_local_stoch_feat(numOfLayer) 
        else:
            print('Stochastic feats with global features only')
            model = IR_onlyResNet50_stoch(numOfLayer)
    elif args.use_scn_attention:
        model = IR_onlyResNet50(numOfLayer)
    else:
        if args.local_feat:
            model = IR_global_local(numOfLayer, args.class_num)
        else:
            model = IR_global(numOfLayer, args.class_num)

    if args.pretrained:
        print('Loading MSCeleb pretrained weights')
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


def get_dataset(split='train', dataset='RAF', max_samples=-1):
    annotations_file = os.path.join('../Dataset', dataset, 'annotations', split+'_annotations.json')
    with open(annotations_file, 'r') as fp:
        annotations = json.load(fp)[split]
    df = pd.DataFrame.from_dict(annotations)
    if max_samples>0:
        df = df.head(max_samples)
    data = {}
    data['img_paths']=df['img_path'].tolist()
    data['bboxs']=df['bbox'].tolist()
    data['labels']=df['label'].tolist()
    data['landmarks']=df['landmarks'].tolist()
    return data


def BuildDataloader(args, split='train', domain='source', max_samples=-1, use_aug=False):
    """Bulid data loader."""

    assert split in ['train', 'test'], 'Function BuildDataloader : function parameter flag1 wrong.'
    assert domain in ['source', 'target'], 'Function BuildDataloader : function parameter flag2 wrong.'

    # Set Transform
    trans = transforms.Compose([
        transforms.Resize((args.face_scale, args.face_scale)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if use_aug:
        print('Using color jitter augmentations')
        aug_train_transform = transforms.Compose([
                transforms.Resize((args.face_scale, args.face_scale)),
                RandAugment_ColorJitter(n=1, m=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        trans = aug_train_transform

    if domain == 'source':
        dataset = args.source
    elif domain == 'target':
        dataset = args.target
    data_dict = get_dataset(split=split, dataset = dataset, max_samples=max_samples)

    # DataSet Distribute
    distribute_ = np.array(data_dict['labels'])
    print(' %s %s dataset qty: %d' % (split, domain, len(data_dict['img_paths'])))
    dataset_dist = []
    for i in range(args.class_num):
        dataset_dist.append(np.sum(distribute_ == i))

    print("Dataset Distribution for %s classes is: " % (args.class_num), dataset_dist)

    # DataSet

    # DataLoader
    if split == 'train':
        data_set = MyDataset(data_dict['img_paths'], data_dict['labels'], data_dict['bboxs'], data_dict['landmarks'], 
                            split, domain, trans, class_num=args.class_num, class_count=args.train_class_count)
        # data_loader = data.DataLoader(dataset=data_set, batch_size=args.train_batch, shuffle=True, num_workers=8,
        #                               drop_last=True)

        train_sampler = ImbalancedDatasetSampler(data_set)
        data_loader = data.DataLoader(dataset=data_set, sampler=train_sampler, batch_size=args.train_batch, num_workers=8,
                                      drop_last=True)
    elif split == 'test':
        data_set = MyDataset(data_dict['img_paths'], data_dict['labels'], data_dict['bboxs'], data_dict['landmarks'], split, domain, trans)

        data_loader = data.DataLoader(dataset=data_set, batch_size=args.test_batch, shuffle=False, num_workers=8,
                                      drop_last=False)

    return data_loader


def Build_AugDataloader(args, split='train', domain='source', max_samples=-1, aug_transform='Contrast'):
    """Bulid data loader."""

    assert split in ['train', 'test'], 'Function BuildDataloader : function parameter flag1 wrong.'
    assert domain in ['source', 'target'], 'Function BuildDataloader : function parameter flag2 wrong.'

    # Set Transform
    trans = transforms.Compose([
        transforms.Resize((args.face_scale, args.face_scale)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    print('Using color jitter augmentations')

    transform_dict={'Autocontrast':RandAugment_AutoContrast(n=1,m=10), 
                    'Brightness':RandAugment_Brightness(n=1,m=10),
                    'Color':RandAugment_Color(n=1,m=10),
                    'Contrast':RandAugment_Contrast(n=1,m=10)}

    aug_transform = transforms.Compose([
                transforms.Resize((args.face_scale, args.face_scale)),
                transform_dict[aug_transform],
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    trans = aug_transform

    if domain == 'source':
        dataset = args.source
    elif domain == 'target':
        dataset = args.target
    data_dict = get_dataset(split=split, dataset = dataset, max_samples=max_samples)

    # DataSet Distribute
    distribute_ = np.array(data_dict['labels'])
    print(' %s %s dataset qty: %d' % (split, domain, len(data_dict['img_paths'])))
    dataset_dist = []
    for i in range(args.class_num):
        dataset_dist.append(np.sum(distribute_ == i))

    print("Dataset Distribution for %s classes is: " % (args.class_num), dataset_dist)

    # DataSet

    # DataLoader
    if split == 'train':
        data_set = MyDataset(data_dict['img_paths'], data_dict['labels'], data_dict['bboxs'], data_dict['landmarks'], 
                            split, domain, trans, class_num=args.class_num, class_count=args.train_class_count)
        data_loader = data.DataLoader(dataset=data_set, batch_size=args.train_batch, shuffle=True, num_workers=8,
                                      drop_last=True)
    elif split == 'test':
        data_set = MyDataset(data_dict['img_paths'], data_dict['labels'], data_dict['bboxs'], data_dict['landmarks'], split, domain, trans)

        data_loader = data.DataLoader(dataset=data_set, batch_size=args.test_batch, shuffle=False, num_workers=8,
                                      drop_last=False)

    return data_loader


