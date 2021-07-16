import torch.nn.functional as F
from torch.autograd import Variable

from models.ResNet_feat import StochasticClassifier
from train_setup import *

criterion = nn.CrossEntropyLoss()
eta = 1.0
num_k = 4


def Train_STAR_grl(args, G, F_cls, train_source_dataloader, train_target_dataloader, optimizer_g, optimizer_f,
                      epoch, writer):
    """Train."""
    G.train()
    F_cls.train()
    F_cls.train()
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

        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        # Forward Propagation
        data = Variable(torch.cat((data_source, data_target), 0))
        landmark = Variable(torch.cat((landmark_source, landmark_target), 0))
        label_source = Variable(label_source)

        output = G(data, landmark)
        output1 = F_cls(output)
        output2 = F_cls(output)

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
        all_loss = loss1 + loss2 + 0.01 * entropy_loss

        all_loss.backward(retain_graph=True)
        optimizer_g.step()
        optimizer_f.step()

        # Step B and C using grl: min max discrepancy
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

        for i in range(1):
            feat_t = G(data_target, landmark_target)
            output_t1 = F_cls(feat_t, reverse=True)
            output_t2 = F_cls(feat_t, reverse=True)
            loss_dis = -torch.mean(torch.abs(F.softmax(output_t1) - F.softmax(output_t2)))
            loss_dis.backward()
            optimizer_f.step()
            optimizer_g.step()

        print('Train Ep: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}\t Dis: {:.6f} Entropy: {:.6f}'.format(
            epoch, batch_index * batch_size, 12000,
                   100. * batch_index / num_iter, loss1.data.item(), loss2.data.item(), loss_dis.data.item(),
            entropy_loss.data.item()))

        # Log loss
        m_total_loss.update(float(all_loss.cpu().data.item()))
        m_loss1.update(float(loss1.cpu().data.item()))
        m_loss2.update(float(loss2.cpu().data.item()))
        m_loss_dis.update(float(loss_dis.cpu().data.item()))
        m_entropy_loss.update(float(entropy_loss.cpu().data.item()))

    LoggerInfo = '''Epoch {0}
    Learning Rate {1}\n
    '''.format(epoch, args.lr)

    LoggerInfo += '''Total Loss {loss:.4f} Cls1 Loss {loss1:.4f} Cls2 Loss {loss2:.4f} Discrepancy Loss {dis_loss:.4f} Entropy loss {ent_loss}''' \
        .format(loss=m_total_loss.avg, loss1=m_loss1.avg, loss2=m_loss2.avg, dis_loss=m_loss_dis.avg,
                ent_loss=m_entropy_loss.avg)

    print(LoggerInfo)

    return


def Train_STAR(args, G, F_cls, train_source_dataloader, train_target_dataloader, optimizer_g, optimizer_f, epoch,
              writer):
    """Train."""
    G.train()
    F_cls.train()
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

        output = G(data, landmark)
        output1 = F_cls(output)
        output2 = F_cls(output)

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
        all_loss = loss1 + loss2 + 0.0 * entropy_loss
        all_loss.backward()
        optimizer_g.step()
        optimizer_f.step()

        # Step B train classifier to maximize discrepancy
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

        output = G(data, landmark)
        output1 = F_cls(output)
        output2 = F_cls(output)
        output_s1 = output1[:batch_size, :]
        output_s2 = output2[:batch_size, :]
        output_t1 = output1[batch_size:, :]
        output_t2 = output2[batch_size:, :]
        output_t1 = F.softmax(output_t1)
        output_t2 = F.softmax(output_t2)
        loss1 = criterion(output_s1, target1)
        loss2 = criterion(output_s2, target1)
        entropy_loss =- torch.mean(torch.log(torch.mean(output_t1, 0) + 1e-6))
        entropy_loss -= torch.mean(torch.log(torch.mean(output_t2, 0) + 1e-6))
        loss_dis = torch.mean(torch.abs(output_t1 - output_t2))
        F_loss = loss1 + loss2 - eta * loss_dis + 0.0 * entropy_loss
        F_loss.backward()
        optimizer_f.step()
        # Step C train generator to minimize discrepancy
        for i in range(num_k):
            optimizer_g.zero_grad()
            output = G(data, landmark)
            output1 = F_cls(output)
            output2 = F_cls(output)

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

        # print('Train Ep: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}\t Dis: {:.6f} Entropy: {:.6f}'.format(
        #     epoch, batch_index * batch_size, 12000,
        #            100. * batch_index / num_iter, loss1.data.item(), loss2.data.item(), loss_dis.data.item(),
        #     entropy_loss.data.item()))

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


def Test_STAR(args, G, F_cls, dataloaders, splits=None):
    if splits is None:  # evaluate on test splits by default
        splits = ['test_source', 'test_target']
    G.eval()
    F_cls.eval()
    for split in splits:
        print(f'\n[{split}]')
        iter_dataloader = iter(dataloaders[split])
        acc1, prec1, recall1 = [AverageMeter() for i in range(args.class_num)], \
                               [AverageMeter() for i in range(args.class_num)], \
                               [AverageMeter() for i in range(args.class_num)]
        for batch_index, (input, landmark, label) in enumerate(iter_dataloader):
            input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
            with torch.no_grad():
                feat = G(input, landmark)
                output1 = F_cls(feat)

            Compute_Accuracy(args, output1, label, acc1, prec1, recall1)

        AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc1, prec1, recall1, args.class_num)

    # print('Weight mean')
    # print(F_cls.weight_mu)
    # print('Weight sigma')
    # print(torch.log(1+torch.exp(F_cls.weight_rho)))

    # print('Bias mean')
    # print(F_cls.bias_mu)
    # print('Bias sigma')
    # print(torch.log(1+torch.exp(F_cls.bias_rho)))
    return


def main():
    """Main."""
    torch.manual_seed(args.seed)

    # Experiment Information
    print_experiment_info(args)

    dataloaders, G, optimizer_g, writer = train_setup(args)
    optimizer_g, lr = lr_scheduler_withoutDecay(optimizer_g, lr=args.lr)
    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=20, gamma=0.1, verbose=True)

    F_cls = StochasticClassifier(num_classes=args.class_num, stoch_bias=args.use_stoch_bias)
    F_cls.cuda()
    optimizer_f = optim.SGD(F_cls.parameters(), momentum=0.9, lr=0.001, weight_decay=0.0005)
    scheduler_f = optim.lr_scheduler.StepLR(optimizer_f, step_size=20, gamma=0.1, verbose=True)
    
    # Running Experiment
    print("Run Experiment...")
    for epoch in range(1, args.epochs + 1):
        print(f'Epoch : {epoch}')
        if args.use_grl:
            Train_STAR_grl(args, G, F_cls, dataloaders['train_source'], dataloaders['train_target'], optimizer_g, optimizer_f,
                  epoch, writer)
        else:
            Train_STAR(args, G, F_cls, dataloaders['train_source'], dataloaders['train_target'], optimizer_g, optimizer_f,
                  epoch, writer)
        scheduler_g.step()
        scheduler_f.step()
        print('\nEvaluation ...')
        Test_STAR(args, G, F_cls, dataloaders, splits=['train_source', 'train_target', 'test_source', 'test_target'])
        if args.save_checkpoint:
            torch.save(G.state_dict(), os.path.join(args.out, f'ckpts/star_G.pkl'))
            torch.save(F_cls.state_dict(), os.path.join(args.out, f'ckpts/star_F.pkl'))
    writer.close()


if __name__ == '__main__':
    main()
