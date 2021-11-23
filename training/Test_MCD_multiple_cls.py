import torch.nn.functional as F
from torch.autograd import Variable

from models.ResNet_feat import ResClassifier
from train_setup import *
from utils.Loss import *
import copy

eta = 1.0
num_k = 2 #4

def Test_MCD_cls_tsne(args, G, F_dict, dataloaders, epoch, n_cls, splits=None):
    if splits is None:  # evaluate on test splits by default
        splits = ['test_source', 'test_target']
    G.eval()
    for i in F_dict.keys():
        F_dict[i]['cls'].eval()


    for split in splits:
        print(f'\n[{split}]')

        iter_dataloader = iter(dataloaders[split])
        acc1, prec1, recall1 = [AverageMeter() for i in range(args.class_num)], \
                               [AverageMeter() for i in range(args.class_num)], \
                               [AverageMeter() for i in range(args.class_num)]
        
        acc2, prec2, recall2 = [AverageMeter() for i in range(args.class_num)], \
                               [AverageMeter() for i in range(args.class_num)], \
                               [AverageMeter() for i in range(args.class_num)]
        
        acc3, prec3, recall3 = [AverageMeter() for i in range(args.class_num)], \
                               [AverageMeter() for i in range(args.class_num)], \
                               [AverageMeter() for i in range(args.class_num)]

        for batch_index, (input, landmark, label) in enumerate(iter_dataloader):
            input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
            with torch.no_grad():
                feat = G(input, landmark)
                output = {}
                for i in range(n_cls):
                    output[i] = F.softmax(F_dict[i]['cls'](feat))
                # combined_output = (output[0] + output[1] + output[2])/3.0

            Compute_Accuracy(args, output[0], label, acc1, prec1, recall1)
            Compute_Accuracy(args, output[1], label, acc2, prec2, recall2)
            Compute_Accuracy(args, output[2], label, acc3, prec3, recall3)

        print('Cls1')
        AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc1, prec1, recall1, args.class_num)
        print('Cls2')
        AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc2, prec2, recall2, args.class_num)
        print('Cls3')
        AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc3, prec3, recall3, args.class_num)

    return



def main():
    """Main."""
    torch.manual_seed(args.seed)

    # Experiment Information
    print_experiment_info(args)

    dataloaders, G, optimizer_g, writer = train_setup(args)
    optimizer_g, lr = lr_scheduler_withoutDecay(optimizer_g, lr=args.lr)
    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=20, gamma=0.5, verbose=True)
    

    F_dict = {}
    n_cls=3
    for i in range(n_cls):
        Fi = {'cls': ResClassifier(num_classes=args.class_num, num_layer=1)}
        Fi['cls'].cuda()
        Fi['optimizer'] = optim.SGD(Fi['cls'].parameters(), momentum=0.9, lr=0.001, weight_decay=0.0005)
        Fi['scheduler'] = optim.lr_scheduler.StepLR(Fi['optimizer'], step_size=20, gamma=0.1, verbose=True)
        F_dict[i] = Fi

    # Running Experiment
    print("Run Experiment...")
    for epoch in range(1, args.epochs + 1):
        G_ckpt = torch.load(os.path.join(args.out, f'ckpts/MCD_G_{epoch}.pkl'))
        G.load_state_dict(G_ckpt)
        for i in range(n_cls):
            F_ckpt = torch.load(os.path.join(args.out, f'ckpts/F{i}_{epoch}.pkl'))
            F_dict[i]['cls'].load_state_dict(F_ckpt)

        print('\nEvaluation ...')
        Test_MCD_cls_tsne(args, G, F_dict, dataloaders, epoch, n_cls, splits=['test_target'])
    writer.close()


if __name__ == '__main__':
    main()
