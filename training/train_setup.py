from torch.utils.tensorboard import SummaryWriter
import json
from models.GCN_utils import init_gcn
from utils.Utils import *

parser = argparse.ArgumentParser(description='Domain adaptation for Expression Classification')

parser.add_argument('--log', type=str, help='Log Name')
parser.add_argument('--out', type=str, help='Output Path')
parser.add_argument('--net', type=str, default='ResNet50', choices=['ResNet18', 'ResNet50', 'VGGNet', 'MobileNet'])
parser.add_argument('--pretrained', type=str2bool, help='pretrained', default=True)
parser.add_argument('--GPU_ID', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--save_checkpoint', type=str2bool, default=False, help='whether to save checkpoint')

# Dataset args
parser.add_argument('--source', type=str, default='RAF', choices=['RAF', 'RAF_7class'])
parser.add_argument('--target', type=str, default='AISIN',
                    choices=['JAFFE', 'AISIN'])
# set maximum no. of samples per class. Use to create balanced no. of samples.
parser.add_argument('--source_labeled', type=int, default=-1,
                    help='number of unlabeled samples (default: -1 == all samples)')  # 2465 for RAF
parser.add_argument('--target_unlabeled', type=int, default=-1,
                    help='number of unlabeled samples (default: -1 == all samples)')  # 1700 for AISIN

parser.add_argument('--local_feat', type=str2bool, default=True, help='whether to use Local Feature')
parser.add_argument('--class_num', type=int, default=7, help='number of class (default: 7)')

parser.add_argument('--lamda', type=float, default=0.1, help='weight for DANN/MME loss')

# DANN variants
parser.add_argument('--use_dan', type=str2bool, default=False, help='whether to use DAN Loss')
parser.add_argument('--dan_method', type=str, default='CDAN-E', choices=['CDAN', 'CDAN-E', 'DANN'])

# MME
parser.add_argument('--use_mme', type=str2bool, default=False, help='whether to use MME loss')

# MCD
parser.add_argument('--use_mcd', type=str2bool, default=False, help='whether to use MCD')
parser.add_argument('--use_grl', type=str2bool, default=False, help='whether to use one step grl')
parser.add_argument('--use_stoch_cls', type=str2bool, default=False, help='whether to use stochastic classifier')

# Feature norm based DA methods
parser.add_argument('--use_afn', type=str2bool, default=False, help='whether to use AFN Loss')
parser.add_argument('--afn_method', type=str, default='SAFN', choices=['HAFN', 'SAFN'])
parser.add_argument('--r', type=float, default=25.0, help='radius of HAFN (default: 25.0)')
parser.add_argument('--dr', type=float, default=1.0, help='radius of SAFN (default: 1.0)')
parser.add_argument('--w_l2', type=float, default=0.05, help='weight L2 norm of AFN (default: 0.05)')

# Training params
parser.add_argument('--face_scale', type=int, default=112, help='Scale of face (default: 112)')
parser.add_argument('--train_batch', type=int, default=32, help='input batch size for training (default: 32)')
parser.add_argument('--test_batch', type=int, default=32, help='input batch size for testing (default: 32)')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--lr_ad', type=float, default=0.001)

parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 10)')
parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='SGD weight decay (default: 0.0005)')

# GCN params
parser.add_argument('--use_gcn', type=str2bool, default=False, help='whether to use GCN')
parser.add_argument('--useClassify', type=str2bool, default=True, help='whether training on source')
parser.add_argument('--intra_gcn', type=str2bool, default=False, help='whether to use Intra-GCN')
parser.add_argument('--inter_gcn', type=str2bool, default=False, help='whether to use Inter-GCN')
parser.add_argument('--rand_mat', type=str2bool, default=False, help='whether to use Random Matrix')
parser.add_argument('--all1_mat', type=str2bool, default=False, help='whether to use All One Matrix')
parser.add_argument('--use_cov', type=str2bool, default=False, help='whether to use Cov')
parser.add_argument('--rand_layer', type=str2bool, default=False, help='whether to use random')
parser.add_argument('--use_cluster', type=str2bool, default=False, help='whether to use Cluster')
parser.add_argument('--method', type=str, default="CADA", help='Choose the method of the experiment')


parser.add_argument('--isTest', type=str2bool, default=False, help='whether to test model')
parser.add_argument('--show_feat', type=str2bool, default=False, help='whether to show feature')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

# Parse Argument
args = parser.parse_args()


def print_experiment_info(args):
    print('Log Name: %s' % args.log)
    print('Output Path: %s' % args.out)
    print('Backbone: %s' % args.net)
    print('Resume Model: %s' % args.pretrained)
    print('CUDA_VISIBLE_DEVICES: %s' % args.GPU_ID)
    print('================================================')

    print('Use {} * {} Image'.format(args.face_scale, args.face_scale))
    print('SourceDataset: %s' % args.source)
    print('TargetDataset: %s' % args.target)
    print('Train Batch Size: %d' % args.train_batch)
    print('Test Batch Size: %d' % args.test_batch)
    print('================================================')

    if args.show_feat:
        print('Show Visualization Result of Feature.')

    if args.isTest:
        print('Test Model.')
    else:
        print('Train Epoch: %d' % args.epochs)
        print('Learning Rate: %f' % args.lr)
        print('Momentum: %f' % args.momentum)
        print('Weight Decay: %f' % args.weight_decay)

        if args.use_afn:
            print('Use AFN Loss: %s' % args.afn_method)
            if args.afn_method == 'HAFN':
                print('Radius of HAFN Loss: %f' % args.r)
            else:
                print('Delta Radius of SAFN Loss: %f' % args.dr)
            print('Weight L2 nrom of AFN Loss: %f' % args.w_l2)

        if args.use_dan:
            print('Use DAN Loss: %s' % args.dan_method)
            print('Learning Rate(Adversarial Network): %f' % args.lr_ad)

        if args.use_mme:
            print('Using MME for domain adaptation')

        if args.use_mcd:
            print('Using MCD for domain adaptation')

    print('================================================')

    print('Number of classes : %d' % args.class_num)
    if not args.local_feat:
        print('Only use global feature.')
    else:
        print('Use global feature and local feature.')
        if args.use_gcn:
            if args.intra_gcn:
                print('Use Intra GCN.')
            if args.inter_gcn:
                print('Use Inter GCN.')

            if args.rand_mat and args.useAllOneMatrix:
                print('Wrong : Use RandomMatrix and AllOneMatrix both!')
                return None
            elif args.rand_mat:
                print('Use Random Matrix in GCN.')
            elif args.all1_mat:
                print('Use All One Matrix in GCN.')

            if args.use_cov and args.use_cluster:
                print('Wrong : Use Cov and Cluster both!')
                return None
            else:
                if args.use_cov:
                    print('Use Mean and Cov.')
                else:
                    print('Use Mean.') if not args.use_cluster else print('Use Mean in Cluster.')

    print('================================================')
    return


def train_setup(args):
    # Build Dataloader
    print("Building Train and Test Dataloader...")
    dataloaders = {'train_source': BuildDataloader(args, flag1='train', flag2='source', max_samples=args.source_labeled),
                   'train_target': BuildDataloader(args, flag1='train', flag2='target', max_samples=args.target_unlabeled),
                   'test_source': BuildDataloader(args, flag1='test', flag2='source'),
                   'test_target': BuildDataloader(args, flag1='test', flag2='target')}
    print('Done!')

    print('================================================')

    # Bulid Model
    print('Building Model...')
    model = BuildModel(args)
    print('Done!')
    print('================================================')

    # Init Mean if using GCN
    if args.use_gcn:
        init_gcn(args, dataloaders['train_source'], dataloaders['train_target'], model)

    # Set Optimizer
    print('Building Optimizer...')
    param_optim = Set_Param_Optim(args, model)
    optimizer = optim.SGD(param_optim, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    print('Done!')

    print('================================================')

    writer = SummaryWriter(os.path.join(args.out, args.log))

    # save arguments used
    with open(os.path.join(args.out,'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    os.makedirs(os.path.join(args.out, 'ckpts'), exist_ok=True)

    return dataloaders, model, optimizer, writer
