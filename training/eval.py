from models.ResNet_feat import ResClassifier
from utils.Utils import *
from train_setup import *
from models.ResNet_stoch_feat import IR_global_local_stoch_feat, IR_onlyResNet50_stoch
from models.ResNet_stoch_feat import *

label2exp = {0:'Surprised', 1:'Fear', 2:'Disgust', 3:'Happy', 4:'Sad', 5:'Anger', 6:'Neutral'}

def test_MCD(args, splits=None):
    if splits is None:  # evaluate on test splits by default
        splits = ['test_source', 'test_target']
    # args.train_batch = 1
    # args.test_batch = 1

    # Build Dataloader
    print("Building Train and Test Dataloader...")
    dataloaders = {'train_source': BuildDataloader(args, split='train', domain='source', max_samples=args.source_labeled),
                   'train_target': BuildDataloader(args, split='train', domain='target', max_samples=args.target_unlabeled),
                   'test_source': BuildDataloader(args, split='test', domain='source'),
                   'test_target': BuildDataloader(args, split='test', domain='target')}
    print('Done!')

    if args.use_mcd:
        G = IR_global_local_feat(50) 
        print(G)
        G_ckpt = torch.load(os.path.join(args.out,'ckpts', 'MCD_G.pkl'))
        G.load_state_dict(G_ckpt)


        F1 = ResClassifier(num_classes=args.class_num, num_layer=1)
        F1_ckpt = torch.load(os.path.join(args.out,'ckpts', 'MCD_F2.pkl'))
        F1.load_state_dict(F1_ckpt)

 
    G.cuda()
    F1.cuda()
    G.eval()
    F1.eval()

    Features = []
    Labels = []
    results = []
    for split in splits:
        print(f'\n[{split}]')
        iter_dataloader = iter(dataloaders[split])
        acc, prec, recall = [AverageMeter() for i in range(args.class_num)], \
                            [AverageMeter() for i in range(args.class_num)], \
                            [AverageMeter() for i in range(args.class_num)]
        for batch_index, (input, landmark, label) in enumerate(iter_dataloader):
            input, landmark, label = input.cuda(), landmark.cuda(), label

            with torch.no_grad():
                feature = G(input, landmark)
                output = F1(feature)

            Compute_Accuracy(args, output, label, acc, prec, recall)

            Features.append(feature.cpu().data.numpy())
            Label = label.cpu().data.numpy()
            if split == 'test_target' or split =='train_target':
                Label+=7
            elif split == 'train_source':
                Label+=14
            Labels.append(Label)

        AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)

    if args.show_feat:
        Features = np.vstack(Features)
        Labels = np.concatenate(Labels)
        viz_tsne(args, Features, Labels)
    return


def test_stoch_MCD(args, splits=None):
    if splits is None:  # evaluate on test splits by default
        splits = ['test_source', 'test_target']
    args.train_batch = 1
    args.test_batch = 1

    # Build Dataloader
    print("Building Train and Test Dataloader...")
    dataloaders = {'train_source': BuildDataloader(args, split='train', domain='source', max_samples=args.source_labeled),
                   'train_target': BuildDataloader(args, split='train', domain='target', max_samples=args.target_unlabeled),
                   'test_source': BuildDataloader(args, split='test', domain='source'),
                   'test_target': BuildDataloader(args, split='test', domain='target')}
    print('Done!')


    G = IR_global_local_stoch_feat(50,feature_dim=384) 
    print(G)
    G_ckpt = torch.load(os.path.join(args.out,'ckpts', 'Stoch_MCD_G.pkl'))
    G.load_state_dict(G_ckpt)

    F1 = Stochastic_Features_cls(args, input_dim=G.output_num())
    F1_ckpt = torch.load(os.path.join(args.out,'ckpts', 'Stoch_MCD_F2.pkl'))
    F1.load_state_dict(F1_ckpt)
 
    G.cuda()
    F1.cuda()
    G.eval()
    F1.eval()

    Features = []
    Labels = []
    results = []
    for split in splits:
        print(f'\n[{split}]')
        iter_dataloader = iter(dataloaders[split])
        acc, prec, recall = [AverageMeter() for i in range(args.class_num)], \
                            [AverageMeter() for i in range(args.class_num)], \
                            [AverageMeter() for i in range(args.class_num)]
        for batch_index, (input, landmark, label) in enumerate(iter_dataloader):
            input, landmark, label = input.cuda(), landmark.cuda(), label

            with torch.no_grad():
                feature, sigma = G(input, landmark)
                output = F1(feature)

            Compute_Accuracy(args, output, label, acc, prec, recall)

            Features.append(feature.cpu().data.numpy())
            Label = label.cpu().data.numpy()
            if split == 'test_target':
                Label+=7
            elif split == 'train_source':
                Label+=14
            Labels.append(Label)

        AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)

    if args.show_feat:
        Features = np.vstack(Features)
        Labels = np.concatenate(Labels)
        viz_tsne(args, Features, Labels)
    return

def main():
    if args.use_stoch_feats:
        test_stoch_MCD(args, splits = ['train_target','train_source'])
    elif args.use_mcd:
        test_MCD(args, splits = ['test_target','train_source'])
    return

if __name__ == '__main__':
    main()