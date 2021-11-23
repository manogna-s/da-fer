from models.AdversarialNetwork import init_weights
import torch.nn.functional as F
from torch.autograd import Variable
from train_setup import *
from models.ResNet_utils import init_weights
from utils.Loss import FocalLoss, LDAMLoss
from models.mixstyle import activate_mixstyle, deactivate_mixstyle
from utils.Utils import BuildDataloader

class Classifier(nn.Module):
    def __init__(self, feature_dim = 512, num_classes=7):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(nn.Linear (feature_dim, 384), nn.ReLU())
        self.cls = nn.Linear(384, num_classes)
        
        self.cls.apply(init_weights)
        self.fc.apply(init_weights)

    def forward(self, x):
        logits = self.cls(x)
        # logits = self.cls(self.fc(x))
        return logits

def train (args, G, F, train_dataloader1, train_dataloader2, optimizer_g, optimizer_f, epoch, writer, criterion) :
    G.train ()
    F.train ()

    torch.autograd.set_detect_anomaly (True)

    total_loss = AverageMeter ()
    iter_source_dataloader1 = iter (train_dataloader1)
    iter_source_dataloader2 = iter (train_dataloader2)
    num_iter = len(train_dataloader1) if (len(train_dataloader1) > len(train_dataloader2)) else len(
        train_dataloader2)

    try:
        data2, landmark2, label2 = train_dataloader2.next()
    except:
        iter_source_dataloader2 = iter(train_dataloader2)
        data2, landmark2, label2 = iter_source_dataloader2.next()

    # for batch_index, (data, landmark, label) in enumerate (train_dataloader1) :
    for i in range(num_iter):
        try:
            data1, landmark1, label1 = train_dataloader1.next()
        except:
            iter_source_dataloader1 = iter(train_dataloader1)
            data1, landmark1, label1 = iter_source_dataloader1.next()
        # try:
        #     data2, landmark2, label2 = train_dataloader2.next()
        # except:
        #     iter_source_dataloader2 = iter(train_dataloader2)
        #     data2, landmark2, label2 = iter_source_dataloader2.next()

        data = Variable(torch.cat((data1, data2), 0))
        landmark = Variable(torch.cat((landmark1, landmark2), 0))
        label = Variable(torch.cat((label1, label2), 0))
        # perm = torch.randperm(args.train_batch)
        # data = data[perm]
        # landmark = landmark[perm]
        # label = label[perm]        

        B= args.train_batch
        perm1 = torch.randperm(B//2)
        perm = torch.arange(B)
        perm[:B//2]=perm1
        perm = perm[:38]
        data = data[perm]
        landmark = landmark[perm]
        label = label[perm]  


        data, landmark, label = data.cuda(), landmark.cuda(), label.cuda()
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        G.apply(activate_mixstyle)
        output = G (data[:B], landmark[:B])
        output = F (output)
        
        loss = criterion (output, label)
        loss.backward()
        optimizer_g.step()
        optimizer_f.step()
        total_loss.update (float (loss.cpu().data.item()))
        # print(f'Ep : {epoch},  loss: {loss.data.item()}')
    print(f'Train Epoch : Total avg loss {total_loss.avg}')
    return

def test (args, G, F, train_loader, test_loader, epoch) :
    G.eval ()
    F.eval ()
    G.apply(deactivate_mixstyle)
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
            output = G (data, landmark)
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
        Labels.append (label.cpu().data.numpy())
        Compute_Accuracy(args, output, label, acc, prec, recall)

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f_avg = Show_Accuracy(acc, prec, recall, args.class_num)

    Features = np.vstack(Features)
    Labels = np.concatenate(Labels)
    viz_tsne(args, Features, Labels, epoch=epoch)

def test_target(args, G, F, test_loader, epoch) :
    G.eval ()
    F.eval ()
    G.apply(deactivate_mixstyle)

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
        Compute_Accuracy(args, output, label, acc, prec, recall)

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f_avg = Show_Accuracy(acc, prec, recall, args.class_num)
    return

def main ():
    torch.manual_seed (args.seed)
    print_experiment_info(args)



    dataloaders, G, optimizer_g, writer = train_setup (args)
    
    dataloaders['train_source1'] = dataloaders['train_source']
    args.source='CK+'
    dataloaders['train_source2'] = BuildDataloader(args, split='train', domain='source', max_samples=args.source_labeled)
    args.source='RAF'

    optimizer_g, lr = lr_scheduler_withoutDecay (optimizer_g, lr=args.lr)
    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=5, gamma=0.5, verbose=True)
    # print(G)

    F = Classifier(feature_dim=G.output_num(), num_classes=args.class_num)
    F.cuda ()
    optimizer_f = optim.SGD(list(F.parameters()), momentum=0.9, lr=0.001, weight_decay=0.0005)
    scheduler_f = optim.lr_scheduler.StepLR(optimizer_f, step_size=40, gamma=0.1, verbose=True)
    
    if args.show_feat:
        G_ckpt= os.path.join(args.out, f'ckpts/FE.pkl')
        if os.path.exists(G_ckpt):
            checkpoint = torch.load (G_ckpt, map_location='cuda')
            G.load_state_dict (checkpoint, strict=False)

        F1_ckpt= os.path.join(args.out, f'ckpts/Cls.pkl')
        if os.path.exists(F1_ckpt):
            checkpoint = torch.load (F1_ckpt, map_location='cuda')
            F.load_state_dict (checkpoint, strict=False)

        test_target(args, G, F, dataloaders['test_target'], 'last')
        return

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
    elif args.criterion == 'ldam':
        if args.source == 'RAF_balanced':
            cls_num_list= np.array([713, 262, 713, 713, 713, 682, 713])
        else: #RAF
            cls_num_list= np.array([1259, 262, 713, 4705, 1885, 682, 2465])
        idx = 0
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], cls_num_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = [1.75, 3.0, 2.0, 1.0, 1.5, 2.0, 1.25]
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
    print(f'Using {args.criterion} loss')

    if args.show_feat :
        G.load_state_dict (torch.load (os.path.join (args.out, 'ckpts/FE.pkl')))
        F.load_state_dict (torch.load (os.path.join (args.out, 'ckpts/Cls.pkl')))
        test(args, G, F, dataloaders['train_source'], dataloaders['test_source'], 31)
        return
    else :
        print("Run Experiment...")
        for epoch in range(1, args.epochs + 1):
            if args.criterion=='ldam':
                if epoch >3:
                    per_cls_weights = [1.75, 3.0, 2.0, 1.0, 1.5, 2.0, 1.25]
                else: 
                    per_cls_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                # per_cls_weights = get_drw_weights(args, epoch, cls_num_list)
                print(f'Epoch: {epoch}, per cls weights: {per_cls_weights}')
                per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
                criterion = LDAMLoss(cls_num_list, weight=per_cls_weights)

            print(f'Epoch : {epoch}')
            train(args, G, F, dataloaders['train_source1'], dataloaders['train_source2'], optimizer_g, optimizer_f, epoch, writer, criterion)
            scheduler_g.step()
            scheduler_f.step()
            print('\nEvaluation ...')
            test(args, G, F, dataloaders['train_source'], dataloaders['test_source'], epoch)
            test_target(args, G, F, dataloaders['test_target'], epoch)
            if args.save_checkpoint:
                torch.save(G.state_dict(), os.path.join(args.out, f'ckpts/FE.pkl'))
                torch.save(F.state_dict(), os.path.join(args.out, f'ckpts/Cls.pkl'))

if __name__ == '__main__':
    main()