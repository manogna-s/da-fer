from models.AdversarialNetwork import init_weights
import torch.nn.functional as F
from torch.autograd import Variable
from train_setup import *
from models.ResNet_utils import init_weights
from utils.Loss import FocalLoss, WeightedFocalLoss
import torchvision.models as models


class Res18Feature(nn.Module):
    def __init__(self, pretrained = True, num_classes = 7, drop_rate = 0):
        super(Res18Feature, self).__init__()
        self.drop_rate = drop_rate
        resnet  = models.resnet18(pretrained)
        # self.feature = nn.Sequential(*list(resnet.children())[:-2]) # before avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-1]) # after avgpool 512x1

        fc_in_dim = list(resnet.children())[-1].in_features # original fc layer's in dimention 512

        self.fc = nn.Linear(fc_in_dim, num_classes) # new fc layer 512x7

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

class Classifier(nn.Module):
    def __init__(self, feature_dim = 512, num_classes=7):
        super(Classifier, self).__init__()
        self.cls = nn.Linear(feature_dim, num_classes)        
        self.cls.apply(init_weights)

    def forward(self, x):
        logits = self.cls(x)
        return logits


def train (args, G, F, train_dataloader, optimizer_g, optimizer_f, epoch, writer, criterion) :
    G.train ()
    F.train ()

    torch.autograd.set_detect_anomaly (True)

    total_loss = AverageMeter ()
    train_dataloader = iter (train_dataloader)

    for batch_index, (data, landmark, label) in enumerate (train_dataloader) :
        data, landmark, label = data.cuda(), landmark.cuda(), label.cuda()
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        
        output = G (data)
        output = F (output)
        
        loss = criterion (output, label)
        loss.backward()
        optimizer_g.step()
        optimizer_f.step()
        total_loss.update (float (loss.cpu().data.item()))
        print(f'Ep : {epoch},  loss: {loss.data.item()}')
    print(f'Train Epoch : Total avg loss {total_loss.avg}')
    return

def test (args, G, F, train_loader, test_loader, epoch) :
    G.eval ()
    F.eval ()
    
    Features = []
    Labels = []
    print ('Evaluating Training Dataset')

    acc, prec, recall = [AverageMeter() for i in range(args.class_num)], \
                            [AverageMeter() for i in range(args.class_num)], \
                            [AverageMeter() for i in range(args.class_num)]

    train_loader = iter (train_loader)
    for batch_index, (data, landmark, label) in enumerate (train_loader) :
        data, landmark, label = data.cuda(), landmark.cuda(), label.cuda()

        with torch.no_grad () :
            output = G (data)
            output = F (output)
        
        Features.append (output.cpu().data.numpy())
        Labels.append (label.cpu().data.numpy()+14)
        Compute_Accuracy(args, output, label, acc, prec, recall)

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f_avg = Show_Accuracy(acc, prec, recall, args.class_num)

    print ('Evaluating Testing Dataset')
    acc, prec, recall = [AverageMeter() for i in range(args.class_num)], \
                        [AverageMeter() for i in range(args.class_num)], \
                        [AverageMeter() for i in range(args.class_num)]

    test_loader = iter (test_loader)
    for batch_index, (data, landmark, label) in enumerate (test_loader) :
        data, landmark, label = data.cuda(), landmark.cuda(), label.cuda()

        with torch.no_grad () :
            output = G (data, landmark)
            output = F (output)

        Features.append (output.cpu().data.numpy())
        Labels.append (label.cpu().data.numpy()+7)
        Compute_Accuracy(args, output, label, acc, prec, recall)

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f_avg = Show_Accuracy(acc, prec, recall, args.class_num)

    Features = np.vstack(Features)
    Labels = np.concatenate(Labels)
    viz_tsne(args, Features, Labels, epoch=epoch)

def main ():
    torch.manual_seed (args.seed)
    print_experiment_info(args)

    dataloaders, _, optimizer_g, writer = train_setup (args)

    G = Res18Feature(num_classes=args.class_num)

    optimizer_g, lr = lr_scheduler_withoutDecay (optimizer_g, lr=args.lr)
    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=20, gamma=0.1, verbose=True)

    F = Classifier(feature_dim=G.output_num(), num_classes=args.class_num)
    F.cuda ()
    optimizer_f = optim.SGD(list(F.parameters()), momentum=0.9, lr=0.001, weight_decay=0.0005)
    scheduler_f = optim.lr_scheduler.StepLR(optimizer_f, step_size=20, gamma=0.1, verbose=True)

    if args.criterion == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.criterion == 'focal':
        criterion = FocalLoss(gamma=1)
    elif args.criterion == 'weighted_focal':
        cls_num_list= np.array([1259, 262, 713, 4705, 1885, 682, 2465])

        beta = 0.9999
        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        class_weights = torch.FloatTensor(per_cls_weights).cuda()
        criterion = FocalLoss(weight=class_weights, gamma=1)
    print(f'Using {args.criterion} loss')

    if args.show_feat :
        G.load_state_dict (torch.load (os.path.join (args.out, 'ckpts/FE.pkl')))
        F.load_state_dict (torch.load (os.path.join (args.out, 'ckpts/Cls.pkl')))
        test(args, G, F, dataloaders['train_source'], dataloaders['test_source'], 31)
        return
    else :
        print("Run Experiment...")
        for epoch in range(1, args.epochs + 1):
            print(f'Epoch : {epoch}')
            train(args, G, F, dataloaders['train_source'], optimizer_g, optimizer_f, epoch, writer, criterion)
            scheduler_g.step()
            scheduler_f.step()
            print('\nEvaluation ...')
            test(args, G, F, dataloaders['train_source'], dataloaders['test_source'], epoch)
            if args.save_checkpoint:
                torch.save(G.state_dict(), os.path.join(args.out, f'ckpts/FE.pkl'))
                torch.save(F.state_dict(), os.path.join(args.out, f'ckpts/Cls.pkl'))

if __name__ == '__main__':
    main()