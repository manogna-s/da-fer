import torch.nn.functional as F
from torch.autograd import Variable

from models.ResNet_stoch_feat import *
from train_setup import *
from torch.distributions.multivariate_normal import MultivariateNormal

criterion = nn.CrossEntropyLoss()
weighted_criterion = nn.CrossEntropyLoss(reduction='none')
eta = 1.0
num_k = 2
n_samples = 5

sigma_avg = 3 
threshold = 2 * np.log(7)

def get_sample_loss(G, F1, F2, data, landmark, target):
    sample_loss = 0
    feat, sigma = G(data, landmark)
    mvn = MultivariateNormal(feat, scale_tril=torch.diag_embed(sigma))

    pred_entropy=0
    for i in range(n_samples):
        feat = mvn.rsample()
        output1 = F1(feat)
        sample_loss += criterion(output1, target)
        output2 = F2(feat)
        sample_loss += criterion(output2, target)

        probs_1= F.softmax(output1)
        probs_2= F.softmax(output2)
        pred_entropy += -torch.sum(probs_1 * torch.log(probs_1), 1)
        pred_entropy += -torch.sum(probs_2 * torch.log(probs_2), 1)
    
    pred_entropy/=n_samples
    weights = nn.ReLU()(threshold - pred_entropy)
    weights = weights/torch.sum(weights)
    print(pred_entropy)
    print(weights)
    sample_loss = sample_loss/n_samples
    return sample_loss, weights

def get_weighted_sample_loss(G, F1, F2, data, landmark, target):
    sample_loss = 0
    feat, sigma = G(data, landmark)
    mvn = MultivariateNormal(feat, scale_tril=torch.diag_embed(sigma))

    pred_entropy=0
    for i in range(n_samples):
        feat = mvn.rsample()
        output1 = F1(feat)
        sample_loss += weighted_criterion(output1, target)
        output2 = F2(feat)
        sample_loss += weighted_criterion(output2, target)

        probs_1= F.softmax(output1)
        probs_2= F.softmax(output2)
        pred_entropy += -torch.sum(probs_1 * torch.log(probs_1), 1)
        pred_entropy += -torch.sum(probs_2 * torch.log(probs_2), 1)
    
    pred_entropy/=n_samples
    weights = nn.ReLU()(threshold - pred_entropy)
    weights = weights/torch.sum(weights)
    print(pred_entropy)
    print(weights)
    sample_loss = (sample_loss * weights).sum()
    sample_loss = sample_loss/n_samples
    return sample_loss, weights

def Train_Stoch_Feat_MCD(args, G, F1, F2, train_source_dataloader, train_target_dataloader, optimizer_g, optimizer_f, epoch,
              writer):
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

        output, _ = G(data, landmark)
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

        sample_loss = 0
        loss_fu = 0
        if epoch>3 and epoch<=5:
            sample_loss, weights = get_sample_loss(G, F1, F2, data_source, landmark_source, label_source)
            loss1 = criterion(output_s1, target1)
            loss2 = criterion(output_s2, target1)
        elif epoch>5:
            sample_loss, weights = get_weighted_sample_loss(G, F1, F2, data_source, landmark_source, label_source)  
            loss1 = (weighted_criterion(output_s1, target1) * weights).sum()
            loss2 = (weighted_criterion(output_s2, target1) * weights).sum()
        else: 
            loss1 = criterion(output_s1, target1)
            loss2 = criterion(output_s2, target1)
        all_loss = loss1 + loss2 + args.lamda_ent * entropy_loss + 0.5 * sample_loss 

        all_loss.backward()
        optimizer_g.step()
        optimizer_f.step()

        # Step B train classifier to maximize discrepancy
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

        output, _ = G(data, landmark)
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
        loss_dis = torch.mean(torch.abs(output_t1 - output_t2))

        sample_loss = 0
        if epoch>3 and epoch<=5:
            sample_loss, weights = get_sample_loss(G, F1, F2, data_source, landmark_source, label_source)
            loss1 = criterion(output_s1, target1)
            loss2 = criterion(output_s2, target1)
        elif epoch>5:
            sample_loss, weights = get_weighted_sample_loss(G, F1, F2, data_source, landmark_source, label_source) 
            loss1 = (weighted_criterion(output_s1, target1) * weights).sum()
            loss2 = (weighted_criterion(output_s2, target1) * weights).sum()
        else: 
            loss1 = criterion(output_s1, target1)
            loss2 = criterion(output_s2, target1)

        F_loss = loss1 + loss2 - eta * loss_dis + args.lamda_ent * entropy_loss
        F_loss += 0.5 * sample_loss

        print(f'Sample loss: {0.5 * sample_loss}, Feature uncertainty loss: {0.005 * loss_fu}')

        F_loss.backward()
        optimizer_f.step()
        # Step C train generator to minimize discrepancy
        for i in range(num_k):
            optimizer_g.zero_grad()
            output, _ = G(data, landmark)
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

        print('Train Ep: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}\tDis: {:.6f} Entropy: {:.6f}'.format(
            epoch, batch_index * batch_size, 12000,
                   100. * batch_index / num_iter, loss1.data.item(), loss2.data.item(), loss_dis.data.item(),
            entropy_loss.data.item()))

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


def Test_Stoch_Feat_MCD(args, G, F1, F2, dataloaders, splits=None):
    if splits is None:  # evaluate on test splits by default
        splits = ['test_source', 'test_target']
    G.eval()
    F1.eval()
    F2.eval()
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
                feat, _ = G(input, landmark)
                output1 = F1(feat)
                output2 = F2(feat)
                
            Compute_Accuracy(args, output1, label, acc1, prec1, recall1)
            Compute_Accuracy(args, output2, label, acc2, prec2, recall2)

        print('Classifier 1')
        AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc1, prec1, recall1, args.class_num)

        print('Classifier 2')
        AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc2, prec2, recall2, args.class_num)
    return


def main():
    """Main."""
    torch.manual_seed(args.seed)

    # Experiment Information
    print_experiment_info(args)

    dataloaders, G, optimizer_g, writer = train_setup(args)
    optimizer_g, lr = lr_scheduler_withoutDecay(optimizer_g, lr=args.lr)
    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=20, gamma=0.1, verbose=True)
    # print(G)
    F1 = Stochastic_Features_cls(args, input_dim=G.output_num())
    F2 = Stochastic_Features_cls(args, input_dim=G.output_num())
    F1.cuda()
    F2.cuda()
    print(F1)


    G_ckpt= os.path.join(args.out, f'ckpts/Stoch_MCD_G.pkl')
    if os.path.exists(G_ckpt):
        checkpoint = torch.load (G_ckpt, map_location='cuda')
        G.load_state_dict (checkpoint, strict=False)

    F1_ckpt= os.path.join(args.out, f'ckpts/Stoch_MCD_F1.pkl')
    if os.path.exists(F1_ckpt):
        checkpoint = torch.load (F1_ckpt, map_location='cuda')
        F1.load_state_dict (checkpoint, strict=False)

    F2_ckpt= os.path.join(args.out, f'ckpts/Stoch_MCD_F2.pkl')
    if os.path.exists(F2_ckpt):
        checkpoint = torch.load (F2_ckpt, map_location='cuda')
        F2.load_state_dict (checkpoint, strict=False)


    optimizer_f = optim.SGD(list(F1.parameters())+list(F2.parameters()), momentum=0.9, lr=0.001, weight_decay=0.0005)
    scheduler_f = optim.lr_scheduler.StepLR(optimizer_f, step_size=20, gamma=0.1, verbose=True)

    optimizer_g_update = optim.SGD(G.get_parameters_update(), momentum=0.9, lr=0.0001, weight_decay=0.0005)
    scheduler_g_update = optim.lr_scheduler.StepLR(optimizer_g_update, step_size=20, gamma=0.1, verbose=True)

    optimizer_f_update = optim.SGD(list(F1.parameters())+list(F2.parameters()), momentum=0.9, lr=0.0001, weight_decay=0.0005)
    scheduler_f_update = optim.lr_scheduler.StepLR(optimizer_f_update, step_size=20, gamma=0.1, verbose=True)



    # Running Experiment
    print("Run Experiment...")
    for epoch in range(1, args.epochs + 1):
        print(f'Epoch : {epoch}')
        # if epoch > 5:
        #     optimizer_g = optimizer_g_update
        #     scheduler_g = scheduler_g_update
        #     optimizer_f = optimizer_f_update
        #     scheduler_f = scheduler_f_update
        Train_Stoch_Feat_MCD(args, G, F1, F2, dataloaders['train_source'], dataloaders['train_target'], optimizer_g, optimizer_f,
                  epoch, writer)
        scheduler_g.step()
        scheduler_f.step()
        print('\nEvaluation ...')
        Test_Stoch_Feat_MCD(args, G, F1, F2, dataloaders, splits=['train_source', 'train_target', 'test_source', 'test_target'])
        if args.save_checkpoint:
            torch.save(G.state_dict(), os.path.join(args.out, f'ckpts/Stoch_MCD_G.pkl'))
            torch.save(F1.state_dict(), os.path.join(args.out, f'ckpts/Stoch_MCD_F1.pkl'))
            torch.save(F2.state_dict(), os.path.join(args.out, f'ckpts/Stoch_MCD_F2.pkl'))

    writer.close()


if __name__ == '__main__':
    main()
