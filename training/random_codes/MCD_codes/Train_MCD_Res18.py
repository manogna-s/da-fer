import torch.nn.functional as F
from torch.autograd import Variable

from models.ResNet_feat import ResClassifier
from train_setup import *
from models.ResNet_utils import init_weights
from models.Res18 import *
from utils.Loss import *

criterion = nn.CrossEntropyLoss()
eta = 1.0
num_k = 4

class Classifier(nn.Module):
    def __init__(self, args, input_dim=100):
        super(Classifier, self).__init__()

        self.cls = nn.Linear(input_dim, args.class_num)
        self.cls.apply(init_weights)

    def forward(self, x):
        out = self.cls(x)
        return out


def Train_MCD(args, G, F1, F2, train_source_dataloader, train_target_dataloader, optimizer_g, optimizer_f, epoch,
              writer, criterion):
    """Train."""
    G.train()
    F1.train()
    F2.train()
    torch.autograd.set_detect_anomaly(True)
    batch_size = args.train_batch

    m_total_loss, m_loss1, m_loss2, m_loss_dis, m_entropy_loss = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    # Get Source/Target Dataloader iterator
    iter_source_dataloader = iter(train_source_dataloader)
    iter_target_dataloader = iter(train_target_dataloader)

    num_iter = len(train_source_dataloader) if (len(train_source_dataloader) > len(train_target_dataloader)) else len(
        train_target_dataloader)

    for batch_index in range(num_iter):
        try:
            data_source, landmark_source, label_source = iter_source_dataloader.next()
        except:
            iter_source_dataloader = iter(train_source_dataloader)
            data_source, landmark_source, label_source = iter_source_dataloader.next()

        try:
            data_target, landmark_target, label_target = iter_target_dataloader.next()
        except:
            iter_target_dataloader = iter(train_target_dataloader)
            data_target, landmark_target, label_target = iter_target_dataloader.next()

        data_source, landmark_source, label_source = data_source.cuda(), landmark_source.cuda(), label_source.cuda()
        data_target, landmark_target, label_target = data_target.cuda(), landmark_target.cuda(), label_target.cuda()

        # Forward Propagation

        data = Variable(torch.cat((data_source, data_target), 0))
        landmark = Variable(torch.cat((landmark_source, landmark_target), 0))
        label_source = Variable(label_source)

        output = G(data)
        output1 = F1(output)
        output2 = F2(output)

        output_s1 = output1[:batch_size, :]
        output_s2 = output2[:batch_size, :]
        output_t1 = output1[batch_size:, :]
        output_t2 = output2[batch_size:, :]
        output_t1 = F.softmax(output_t1)
        output_t2 = F.softmax(output_t2)

        entropy_loss = - torch.mean(torch.log(torch.mean(output_t1, 0) + 1e-6))
        entropy_loss -= torch.mean(torch.log(torch.mean(output_t2, 0) + 1e-6))

        target1 = label_source
        loss1 = criterion(output_s1, target1)
        loss2 = criterion(output_s2, target1)
        all_loss = loss1 + loss2 + args.lamda_ent * entropy_loss
        all_loss.backward()
        optimizer_g.step()
        optimizer_f.step()
        
        # Step B train classifier to maximize discrepancy
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

        output = G(data)
        output1 = F1(output)
        output2 = F2(output)
        output_s1 = output1[:batch_size, :]
        output_s2 = output2[:batch_size, :]
        output_t1 = output1[batch_size:, :]
        output_t2 = output2[batch_size:, :]
        output_t1 = F.softmax(output_t1)
        output_t2 = F.softmax(output_t2)
        loss1 = criterion(output_s1, target1)
        loss2 = criterion(output_s2, target1)
        entropy_loss = - torch.mean(torch.log(torch.mean(output_t1, 0) + 1e-6))
        entropy_loss -= torch.mean(torch.log(torch.mean(output_t2, 0) + 1e-6))
        loss_dis = torch.mean(torch.abs(output_t1 - output_t2))
        F_loss = loss1 + loss2 - eta * loss_dis + args.lamda_ent * entropy_loss
        
        F_loss.backward()
        optimizer_f.step()
        # Step C train generator to minimize discrepancy
        for i in range(num_k):
            optimizer_g.zero_grad()
            output = G(data)
            output1 = F1(output)
            output2 = F2(output)

            output_s1 = output1[:batch_size, :]
            output_s2 = output2[:batch_size, :]
            output_t1 = output1[batch_size:, :]
            output_t2 = output2[batch_size:, :]

            loss1 = criterion(output_s1, target1)
            loss2 = criterion(output_s2, target1)
            output_t1 = F.softmax(output_t1)
            output_t2 = F.softmax(output_t2)
            loss_dis = torch.mean(torch.abs(output_t1 - output_t2))
            entropy_loss = -torch.mean(torch.log(torch.mean(output_t1, 0) + 1e-6))
            entropy_loss -= torch.mean(torch.log(torch.mean(output_t2, 0) + 1e-6))

            loss_dis.backward()
            optimizer_g.step() 

        print('Train Ep: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}\t Dis: {:.6f} Entropy: {:.6f}'.format(
            epoch, batch_index * batch_size, 12000,
                   100. * batch_index / num_iter, loss1.data.item(), loss2.data.item(), loss_dis.data.item(),
            args.lamda_ent * entropy_loss.data.item()))

        # Log loss
        m_total_loss.update(float(F_loss.cpu().data.item()))
        m_loss1.update(float(loss1.cpu().data.item()))
        m_loss2.update(float(loss2.cpu().data.item()))
        m_loss_dis.update(float(loss_dis.cpu().data.item()))
        m_entropy_loss.update(float(entropy_loss.cpu().data.item()))

    LoggerInfo = '''
    [Train source]:
    Epoch {0}
    Learning Rate {1}\n
    '''.format(epoch, args.lr)

    LoggerInfo += '''    Total Loss {loss:.4f} Cls1 Loss {loss1:.4f} Cls2 Loss {loss2:.4f} Discrepancy Loss {dis_loss:.4f} Entropy loss {ent_loss}''' \
        .format(loss=m_total_loss.avg, loss1=m_loss1.avg, loss2=m_loss2.avg, dis_loss=m_loss_dis.avg,
                ent_loss=m_entropy_loss.avg)

    print(LoggerInfo)

    return


def Test_MCD(args, G, F1, F2, dataloaders, epoch, splits=None):
    if splits is None:  # evaluate on test splits by default
        splits = ['test_source', 'test_target']
    G.eval()
    F1.eval()
    F2.eval()
    Features = []
    Labels = []
    for split in splits:
        print(f'\n[{split}]')

        iter_dataloader = iter(dataloaders[split])
        acc1, prec1, recall1 = [AverageMeter() for i in range(args.class_num)], \
                               [AverageMeter() for i in range(args.class_num)], \
                               [AverageMeter() for i in range(args.class_num)]
        acc2, prec2, recall2 = [AverageMeter() for i in range(args.class_num)], \
                               [AverageMeter() for i in range(args.class_num)], \
                               [AverageMeter() for i in range(args.class_num)]
        for batch_index, (input, landmark, label) in enumerate(iter_dataloader):
            input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
            with torch.no_grad():
                feat = G(input)
                output1 = F1(feat)
                output2 = F2(feat)
                
                Features.append (feat.cpu().data.numpy())
                if split == 'train_source':    
                    Labels.append (label.cpu().data.numpy()+14)
                if split == 'test_source':    
                    Labels.append (label.cpu().data.numpy())
                if split == 'test_target':    
                    Labels.append (label.cpu().data.numpy()+7)
                if split == 'train_target':    
                    Labels.append (label.cpu().data.numpy()+7)

            Compute_Accuracy(args, output1, label, acc1, prec1, recall1)
            Compute_Accuracy(args, output2, label, acc2, prec2, recall2)

        print('Classifier 1')
        AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc1, prec1, recall1, args.class_num)

        print('Classifier 2')
        AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc2, prec2, recall2, args.class_num)


    Features = np.vstack(Features)
    Labels = np.concatenate(Labels)
    viz_tsne(args, Features, Labels, epoch=f'{splits[1]}_{epoch}')
    return


def main():
    """Main."""
    torch.manual_seed(args.seed)

    # Experiment Information
    print_experiment_info(args)

    dataloaders, _, _, writer = train_setup(args)
    G = ResNet18(pretrained=True)
    G.cuda()

    # optimizer_g = torch.optim.Adam(G.parameters(),weight_decay = 1e-4)
    optimizer_g = optim.SGD(G.parameters(), momentum=0.9, lr=0.01, weight_decay=0.0005)
    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=20, gamma=0.1, verbose=True)

    F1 = Classifier(args, input_dim=G.output_num())
    F2 = Classifier(args, input_dim=G.output_num())
    F1.cuda()
    F2.cuda()

    optimizer_f = optim.Adam(list(F1.parameters()) + list(F2.parameters()), weight_decay=0.0005)
    optimizer_f = optim.SGD(list(F1.parameters()) + list(F2.parameters()), momentum=0.9, lr=0.01, weight_decay=0.0005)
    scheduler_f = optim.lr_scheduler.StepLR(optimizer_f, step_size=20, gamma=0.1, verbose=True)

    if args.criterion == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.criterion == 'focal':
        criterion = FocalLoss(gamma=1)
    elif args.criterion == 'weighted_focal':
        if args.source == 'RAF_balanced':
            cls_num_list= np.array([713, 262, 713, 713, 713, 682, 713])
        else: #RAF
            cls_num_list= np.array([1259, 262, 713, 4705, 1885, 682, 2465])
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        print(per_cls_weights)
        class_weights = torch.FloatTensor(per_cls_weights).cuda()
        criterion = FocalLoss(weight=class_weights, gamma=1)
    print(f'Using {args.criterion} loss')

    if args.show_feat :
        G.load_state_dict (torch.load (os.path.join (args.out, 'ckpts/MCD_G.pkl')), strict=False)
        F1.load_state_dict (torch.load (os.path.join (args.out, 'ckpts/MCD_F1.pkl')))
        F2.load_state_dict (torch.load (os.path.join (args.out, 'ckpts/MCD_F2.pkl')))

        Test_MCD(args, G, F1, F2, dataloaders, 30, splits=['train_source', 'test_source'])
        Test_MCD(args, G, F1, F2, dataloaders, 30, splits=['train_source', 'test_target'])
        return

    # Running Experiment
    print("Run Experiment...")
    for epoch in range(1, args.epochs + 1):
        print(f'Epoch : {epoch}')

        Train_MCD(args, G, F1, F2, dataloaders['train_source'], dataloaders['train_target'], optimizer_g, optimizer_f,
                  epoch, writer, criterion)
        scheduler_g.step()
        scheduler_f.step()
        print('\nEvaluation ...')
        Test_MCD(args, G, F1, F2, dataloaders, epoch, splits=['train_source', 'test_source'])
        Test_MCD(args, G, F1, F2, dataloaders, epoch, splits=['train_source', 'train_target'])
        Test_MCD(args, G, F1, F2, dataloaders, epoch, splits=['train_source', 'test_target'])

        if args.save_checkpoint:
            torch.save(G.state_dict(), os.path.join(args.out, f'ckpts/MCD_G.pkl'))
            torch.save(F1.state_dict(), os.path.join(args.out, f'ckpts/MCD_F1.pkl'))
            torch.save(F2.state_dict(), os.path.join(args.out, f'ckpts/MCD_F2.pkl'))
    writer.close()


if __name__ == '__main__':
    main()
