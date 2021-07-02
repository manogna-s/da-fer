from utils.Utils import *

parser = argparse.ArgumentParser(description='Domain adaptation for Expression Classification')

parser.add_argument('--log', type=str, help='Log Name')
parser.add_argument('--out', type=str, help='Output Path')
parser.add_argument('--net', type=str, default='ResNet50', choices=['ResNet18', 'ResNet50', 'VGGNet', 'MobileNet'])
parser.add_argument('--pretrained', type=str, help='pretrained', default='None')
parser.add_argument('--dev', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

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