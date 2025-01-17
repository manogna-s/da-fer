import torch.nn.functional as F
from torch.autograd import Variable

from models.ResNet_feat import ResClassifier
from train_setup import *
from utils.Loss import *
import copy
import torch.distributions.normal as normal

eta = 1.0
num_k = 2 #4

class StochasticClassifier_mine(nn.Module):
    def __init__(self, num_features, num_classes, rho=-4):
        super(StochasticClassifier_mine, self).__init__()

        bottleneck_dim = 128
        self.fc1=nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
        )
        num_features =bottleneck_dim
        print(f'Using rho {rho} for softplus')

        # self.weight = torch.zeros((num_classes, num_features))

        self.weight_mu = nn.Parameter(torch.randn(num_classes, num_features)) #nn.Parameter(torch.empty_like(self.weight, requires_grad=True))
        self.weight_rho = nn.Parameter(torch.zeros(num_classes, num_features)) #nn.Parameter(torch.empty_like(self.weight, requires_grad=True))

        self.bias = nn.Parameter(torch.zeros(num_classes))

        nn.init.xavier_normal_(self.weight_mu)
        nn.init.constant_(self.weight_rho, rho)

    def reparameterize(self, sample=False):
        weight_std = torch.log(1 + torch.exp(self.weight_rho))

        if self.training or sample:
            weight_eps = torch.randn_like(weight_std)
        else:
            weight_eps = torch.zeros_like(weight_std)

        self.weight = self.weight_mu + weight_eps * weight_std
        return

    def forward(self, x):
        self.reparameterize()
        x=self.fc1(x)
        out = F.linear(x, self.weight, self.bias)
        return out

class StochasticClassifier_fscil(nn.Module):
    def __init__(self, num_features, num_classes, temp=0.05):
        super().__init__()
        self.mu = nn.Parameter(0.01 * torch.randn(num_classes, num_features))
        self.sigma = nn.Parameter(torch.zeros(num_classes, num_features))
        self.bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.kaiming_uniform_(self.mu)

    def forward(self, x, stochastic=True):
        mu = self.mu
        sigma = self.sigma
        if stochastic and self.training:
            sigma = F.softplus(sigma - 4) # when sigma=0, softplus(sigma-4)=0.0181
            weight = sigma * torch.randn_like(mu) + mu
        else:
            weight = mu
        score = F.linear(x, weight, self.bias)
        return score

class StochasticClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        bottleneck_dim = 128
        self.fc1=nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(num_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(bottleneck_dim, bottleneck_dim),
            # nn.BatchNorm1d(bottleneck_dim),
            # nn.ReLU(),
        )

        num_features =bottleneck_dim
        self.mu = nn.Parameter(torch.randn(num_classes, num_features))
        self.sigma = nn.Parameter(torch.zeros(num_classes, num_features))
        self.bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.kaiming_uniform_(self.mu)


    def forward(self, x, stochastic=True):
        x=self.fc1(x)
        if stochastic and self.training:
            sigma = F.softplus(self.sigma - 2) # when sigma=0, softplus(sigma-4)=0.0181
            distribution = normal.Normal(self.mu, sigma)
            weight = distribution.rsample()
        else:
            weight = self.sigma
        score = F.linear(x, weight, self.bias)
        return score

class StochasticClassifier_basic(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.mu = nn.Parameter(torch.randn(num_classes, num_features))
        self.sigma = nn.Parameter(torch.zeros(num_classes, num_features))
        self.bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.kaiming_uniform_(self.mu)

    def forward(self, x, stochastic=True):
        if stochastic and self.training:
            sigma = F.softplus(self.sigma - 4) # when sigma=0, softplus(sigma-4)=0.0181, softplus(sigma-2)=0.1269
            distribution = normal.Normal(self.mu, sigma)
            weight = distribution.rsample()
        else:
            weight = self.sigma
        score = F.linear(x, weight, self.bias)
        return score


def Train_MCD(args, G, F1, F2, train_source_dataloader, train_target_dataloader, optimizer_g, optimizer_f, epoch,
              writer, criterion):
    """Train."""
    G.train()
    F1.train()
    F2.train()
    # torch.autograd.set_detect_anomaly(True)
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

        output = G(data, landmark)
        output1 = F1(output)
        output2 = F2(output)

        output_s1 = output1[:batch_size, :]
        output_s2 = output2[:batch_size, :]
        output_t1 = output1[batch_size:, :]
        output_t2 = output2[batch_size:, :]
        output_t1 = F.softmax(output_t1, dim=1)
        output_t2 = F.softmax(output_t2, dim=1)

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

        output = G(data, landmark)
        output1 = F1(output)
        output2 = F2(output)
        output_s1 = output1[:batch_size, :]
        output_s2 = output2[:batch_size, :]
        output_t1 = output1[batch_size:, :]
        output_t2 = output2[batch_size:, :]
        output_t1 = F.softmax(output_t1, dim=1)
        output_t2 = F.softmax(output_t2, dim=1)
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
            output = G(data, landmark)
            output1 = F1(output)
            output2 = F2(output)

            output_s1 = output1[:batch_size, :]
            output_s2 = output2[:batch_size, :]
            output_t1 = output1[batch_size:, :]
            output_t2 = output2[batch_size:, :]

            loss1 = criterion(output_s1, target1)
            loss2 = criterion(output_s2, target1)
            output_t1 = F.softmax(output_t1, dim=1)
            output_t2 = F.softmax(output_t2, dim=1)
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



def Test_MCD_tsne(args, G, F1, F2, dataloaders, epoch, splits=None):
    if splits is None:  # evaluate on test splits by default
        splits = ['test_source', 'test_target']
    G.eval()
    F1.eval()
    F2.eval()
    Features = []
    Labels = []


    if True:
        iter_dataloader = iter(dataloaders['train_source'])
        acc1, prec1, recall1 = [AverageMeter() for i in range(args.class_num)], \
                               [AverageMeter() for i in range(args.class_num)], \
                               [AverageMeter() for i in range(args.class_num)]
        acc2, prec2, recall2 = [AverageMeter() for i in range(args.class_num)], \
                               [AverageMeter() for i in range(args.class_num)], \
                               [AverageMeter() for i in range(args.class_num)]
        for batch_index, (input, landmark, label) in enumerate(iter_dataloader):
            input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
            with torch.no_grad():
                feat = G(input, landmark)
                output1 = F1(feat)
                output2 = F2(feat)
                Features.append (feat.cpu().data.numpy())
                Labels.append (label.cpu().data.numpy()+14)
            Compute_Accuracy(args, output1, label, acc1, prec1, recall1)
            Compute_Accuracy(args, output2, label, acc2, prec2, recall2)

        print('Classifier 1')
        AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc1, prec1, recall1, args.class_num)

        print('Classifier 2')
        AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc2, prec2, recall2, args.class_num)

    Features_src_test=copy.deepcopy(Features)
    Labels_src_test=copy.deepcopy(Labels)

    Features_tar_train=copy.deepcopy(Features)
    Labels_tar_train=copy.deepcopy(Labels)

    Features_tar_test=copy.deepcopy(Features)
    Labels_tar_test=copy.deepcopy(Labels)

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
                feat = G(input, landmark)
                output1 = F1(feat)
                output2 = F2(feat)
                
                if split == 'test_source':
                    Features_src_test.append (feat.cpu().data.numpy())    
                    Labels_src_test.append (label.cpu().data.numpy())
                if split == 'test_target':    
                    Features_tar_test.append (feat.cpu().data.numpy())    
                    Labels_tar_test.append (label.cpu().data.numpy()+7)
                if split == 'train_target':  
                    Features_tar_train.append (feat.cpu().data.numpy())    
                    Labels_tar_train.append (label.cpu().data.numpy()+7)

            Compute_Accuracy(args, output1, label, acc1, prec1, recall1)
            Compute_Accuracy(args, output2, label, acc2, prec2, recall2)

        print('Classifier 1')
        AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc1, prec1, recall1, args.class_num)

        print('Classifier 2')
        AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc2, prec2, recall2, args.class_num)


    Features_src_test = np.vstack(Features_src_test)
    Labels_src_test = np.concatenate(Labels_src_test)
    viz_tsne(args, Features_src_test, Labels_src_test, epoch=f'test_source_{epoch}')

    Features_tar_train = np.vstack(Features_tar_train)
    Labels_tar_train = np.concatenate(Labels_tar_train)
    viz_tsne(args, Features_tar_train, Labels_tar_train, epoch=f'train_target_{epoch}')

    Features_tar_test = np.vstack(Features_tar_test)
    Labels_tar_test = np.concatenate(Labels_tar_test)
    viz_tsne(args, Features_tar_test, Labels_tar_test, epoch=f'test_target_{epoch}')
    return

def main():
    """Main."""
    torch.manual_seed(args.seed)

    # Experiment Information
    print_experiment_info(args)

    dataloaders, G, optimizer_g, writer = train_setup(args)
    optimizer_g, lr = lr_scheduler_withoutDecay(optimizer_g, lr=args.lr)
    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=10, gamma=0.5, verbose=True)
    
    F1 = StochasticClassifier_mine(num_features = G.output_num(), num_classes=args.class_num)
    F1.cuda()
    print(F1)

    # F2 = StochasticClassifier(num_features = G.output_num(), num_classes=args.class_num)
    # F2.cuda()
    optimizer_f = optim.SGD(F1.parameters(), momentum=0.9, lr=0.001, weight_decay=0.0005)
    scheduler_f = optim.lr_scheduler.StepLR(optimizer_f, step_size=10, gamma=0.5, verbose=True)

    print(f'Using {args.criterion} loss')

    # Running Experiment
    print("Run Experiment...")
    for epoch in range(1, args.epochs + 1):
        if args.criterion=='ldam':
            if epoch >5:
                per_cls_weights = [1.75, 3.0, 2.0, 1.0, 1.5, 2.0, 1.25]
            else: 
                per_cls_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            print(f'Epoch: {epoch}, per cls weights: {per_cls_weights}')
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
            criterion = LDAMLoss(cls_num_list, weight=per_cls_weights)

        print(f'Epoch : {epoch}')

        # checkpoint = torch.load ('/home/manogna/da-fer/training/logs_STAR/sfew_fc_ft_star/ckpts/MCD_G_1.pkl', map_location='cuda')
        # G.load_state_dict (checkpoint, strict=False)
        # checkpoint = torch.load ('/home/manogna/da-fer/training/logs_STAR/sfew_fc_ft_star/ckpts/MCD_F1_1.pkl', map_location='cuda')
        # F1.load_state_dict (checkpoint, strict=False)

        Train_MCD(args, G, F1, F1, dataloaders['train_source'], dataloaders['train_target'], optimizer_g, optimizer_f,
                  epoch, writer, criterion)
        scheduler_g.step()
        scheduler_f.step()
        print('\nEvaluation ...')
        Test_MCD_tsne(args, G, F1, F1, dataloaders, epoch, splits=['test_source', 'train_target', 'test_target'])

        if args.save_checkpoint and epoch%5:
            torch.save(G.state_dict(), os.path.join(args.out, f'ckpts/MCD_G_{epoch}.pkl'))
            torch.save(F1.state_dict(), os.path.join(args.out, f'ckpts/MCD_F1_{epoch}.pkl'))
    writer.close()


if __name__ == '__main__':
    main()
