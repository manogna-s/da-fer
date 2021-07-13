import os
import pandas as pd
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from models.AdversarialNetwork import RandomLayer, AdversarialNetwork
from models.ResNet import IR_global_local, IR_global
from models.ResNet_GCN import IR_GCN
from models.ResNet_feat import IR_global_local_feat
from models.ResNet_utils import load_resnet_pretrained_weights
from utils.Dataset import MyDataset
from utils.misc_utils import *
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
            model = IR_global_local_feat(numOfLayer)
        else:
            print('MCD with only global feat not yet added')
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


def get_dataset(split='train', domain='RAF'):
    annotations_file = os.path.join('../Dataset', domain, 'annotations', split+'_annotations.json')
    with open(annotations_file, 'r') as fp:
        annotations = json.load(fp)[split]

    df = pd.DataFrame.from_dict(annotations)
    data = {}
    data['img_paths']=df['img_path'].tolist()
    data['bboxs']=df['bbox'].tolist()
    data['labels']=df['label'].tolist()
    data['landmarks']=df['landmarks'].tolist()
    return data


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

    if flag2 == 'source':
        domain = args.source
    elif flag2 == 'target':
        domain = args.target
    data_dict = get_dataset(split=flag1, domain = domain)

    # DataSet Distribute
    distribute_ = np.array(data_dict['labels'])
    print(' %s %s dataset qty: %d' % (flag1, flag2, len(data_dict['img_paths'])))
    dataset_dist = []
    for i in range(args.class_num):
        dataset_dist.append(np.sum(distribute_ == i))

    print("Dataset Distribution for %s classes is: " % (args.class_num), dataset_dist)

    # DataSet
    data_set = MyDataset(data_dict['img_paths'], data_dict['labels'], data_dict['bboxs'], data_dict['landmarks'], flag1, trans, target_trans)

    # DataLoader
    if flag1 == 'train':
        data_loader = data.DataLoader(dataset=data_set, batch_size=args.train_batch, shuffle=True, num_workers=8,
                                      drop_last=True)
    elif flag1 == 'test':
        data_loader = data.DataLoader(dataset=data_set, batch_size=args.test_batch, shuffle=False, num_workers=8,
                                      drop_last=False)

    return data_loader



