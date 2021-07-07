'''
Run using $sh mme_train.sh gpu_id exp_name local_feat
$sh mme_train.sh 0 mme_global False
'''

import time
from utils.Loss import *
from test import Test
from train_setup import *
from models.ResNet_MCD import ResClassifier
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


criterion = nn.CrossEntropyLoss()
eta = 1.0
num_k = 4

def Train_MCD_onestep(args, G, F1, F2, train_source_dataloader, train_target_dataloader, optimizer_g, optimizer_f, epoch, writer):
    """Train."""
    G.train()
    F1.train()
    F2.train()
    torch.autograd.set_detect_anomaly(True)
    batch_size = args.train_batch

    acc_S, prec_S, recall_S = [AverageMeter() for i in range(args.class_num)], \
                              [AverageMeter() for i in range(args.class_num)], \
                              [AverageMeter() for i in range(args.class_num)]

    acc_T, prec_T, recall_T = [AverageMeter() for i in range(args.class_num)], \
                              [AverageMeter() for i in range(args.class_num)], \
                              [AverageMeter() for i in range(args.class_num)]

    m_total_loss, m_loss1, m_loss2, m_loss_dis, m_entropy_loss = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    # Decay Learn Rate per Epoch
    if epoch <= 20:
        args.lr = 0.001
    elif epoch <= 40:
        args.lr = 0.0001
    else:
        args.lr = 0.00001

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
        data = Variable(torch.cat((data_source,data_target),0))
        landmark = Variable(torch.cat((landmark_source,landmark_target),0))
        label_source = Variable(label_source)
        
        output = G(data, landmark)
        output1 = F1(output)
        output2 = F2(output)

        batch_size = args.train_batch
        output_s1 = output1[:batch_size,:]
        output_s2 = output2[:batch_size,:]
        output_t1 = output1[batch_size:,:]
        output_t2 = output2[batch_size:,:]
        output_t1 = F.softmax(output_t1)
        output_t2 = F.softmax(output_t2)

        entropy_loss = - torch.mean(torch.log(torch.mean(output_t1,0)+1e-6))
        entropy_loss -= torch.mean(torch.log(torch.mean(output_t2,0)+1e-6))

        target1 = label_source
        loss1 = criterion(output_s1, target1)
        loss2 = criterion(output_s2, target1)
        all_loss = loss1 + loss2 + 0.0 * entropy_loss

        all_loss.backward(retain_graph=True)
        optimizer_g.step()
        optimizer_f.step()

        #Step B train classifier to maximize discrepancy
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

        for i in range(num_k):
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()
            feat_t = G(data_target, landmark_target)
            output_t1 = F1(feat_t, reverse=True)
            output_t2 = F2(feat_t, reverse=True)
            loss_dis = -torch.mean(torch.abs(output_t1-output_t2))
            loss_dis.backward()
            optimizer_f.step()
            optimizer_g.step()

        print('Train Ep: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}\t Dis: {:.6f} Entropy: {:.6f}'.format(
                    epoch, batch_index * batch_size, 12000,
                    100. * batch_index / num_iter, loss1.data.item(),loss2.data.item(),loss_dis.data.item(),entropy_loss.data.item()))

        # Log loss
        m_total_loss.update(float(all_loss.cpu().data.item()))
        m_loss1.update(float(loss1.cpu().data.item()))
        m_loss2.update(float(loss2.cpu().data.item()))
        m_loss_dis.update(float(loss_dis.cpu().data.item()))
        m_entropy_loss.update(float(entropy_loss.cpu().data.item()))

        # Compute accuracy, precision and recall
        Compute_Accuracy(args, output_s1, label_source, acc_S, prec_S, recall_S)
        Compute_Accuracy(args, output_t1, label_target, acc_T, prec_T, recall_T)



    AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc_S, prec_S, recall_S, args.class_num)

    LoggerInfo = '''
    [Train source]:
    Epoch {0}
    Learning Rate {1}\n'''.format(epoch, args.lr)

    LoggerInfo += AccuracyInfo

    LoggerInfo += '''    Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f}
    Total Loss {loss:.4f} Cls1 Loss {loss1:.4f} Cls2 Loss {loss2:.4f} Discrepancy Loss {dis_loss:.4f} Entropy loss {ent_loss}''' \
                    .format(acc_avg, prec_avg, recall_avg, f1_avg, loss=m_total_loss.avg, loss1=m_loss1.avg,
                            loss2=m_loss2.avg, dis_loss=m_loss_dis.avg, ent_loss=m_entropy_loss.avg)

    print(LoggerInfo)

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc_T, prec_T, recall_T, args.class_num)

    LoggerInfo = '''\n
    [Train target]:\n'''

    LoggerInfo += AccuracyInfo

    LoggerInfo += '''    Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f} 
    Total Loss {loss:.4f} Cls1 Loss {loss1:.4f} Cls2 Loss {loss2:.4f} Discrepancy Loss {dis_loss:.4f} Entropy loss {ent_loss}\n\n''' \
                    .format(acc_avg, prec_avg, recall_avg, f1_avg, loss=m_total_loss.avg, loss1=m_loss1.avg,
                            loss2=m_loss2.avg, dis_loss=m_loss_dis.avg, ent_loss=m_entropy_loss.avg)

    print(LoggerInfo)

    return


def Train_MCD(args, G, F1, F2, train_source_dataloader, train_target_dataloader, optimizer_g, optimizer_f, epoch, writer):
    """Train."""
    G.train()
    F1.train()
    F2.train()
    torch.autograd.set_detect_anomaly(True)

    acc_S, prec_S, recall_S = [AverageMeter() for i in range(args.class_num)], \
                              [AverageMeter() for i in range(args.class_num)], \
                              [AverageMeter() for i in range(args.class_num)]

    acc_T, prec_T, recall_T = [AverageMeter() for i in range(args.class_num)], \
                              [AverageMeter() for i in range(args.class_num)], \
                              [AverageMeter() for i in range(args.class_num)]

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


        data = Variable(torch.cat((data_source,data_target),0))
        landmark = Variable(torch.cat((landmark_source,landmark_target),0))
        label_source = Variable(label_source)
        
        output = G(data, landmark)
        output1 = F1(output)
        output2 = F2(output)

        batch_size = args.train_batch
        output_s1 = output1[:batch_size,:]
        output_s2 = output2[:batch_size,:]
        output_t1 = output1[batch_size:,:]
        output_t2 = output2[batch_size:,:]
        output_t1 = F.softmax(output_t1)
        output_t2 = F.softmax(output_t2)

        entropy_loss = - torch.mean(torch.log(torch.mean(output_t1,0)+1e-6))
        entropy_loss -= torch.mean(torch.log(torch.mean(output_t2,0)+1e-6))

        target1 = label_source
        loss1 = criterion(output_s1, target1)
        loss2 = criterion(output_s2, target1)
        all_loss = loss1 + loss2 + 0.01 * entropy_loss
        all_loss.backward()
        optimizer_g.step()
        optimizer_f.step()

        #Step B train classifier to maximize discrepancy
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

        output = G(data, landmark)
        output1 = F1(output)
        output2 = F2(output)
        output_s1 = output1[:batch_size,:]
        output_s2 = output2[:batch_size,:]
        output_t1 = output1[batch_size:,:]
        output_t2 = output2[batch_size:,:]
        output_t1 = F.softmax(output_t1)
        output_t2 = F.softmax(output_t2)
        loss1 = criterion(output_s1, target1)
        loss2 = criterion(output_s2, target1)
        entropy_loss = - torch.mean(torch.log(torch.mean(output_t1,0)+1e-6))
        entropy_loss -= torch.mean(torch.log(torch.mean(output_t2,0)+1e-6))
        loss_dis = torch.mean(torch.abs(output_t1-output_t2))
        F_loss = loss1 + loss2 - eta*loss_dis  + 0.01 * entropy_loss
        F_loss.backward()
        optimizer_f.step()
        # Step C train genrator to minimize discrepancy
        for i in range(num_k):
                optimizer_g.zero_grad()
                output = G(data, landmark)
                output1 = F1(output)
                output2 = F2(output)

                output_s1 = output1[:batch_size,:]
                output_s2 = output2[:batch_size,:]
                output_t1 = output1[batch_size:,:]
                output_t2 = output2[batch_size:,:]

                loss1 = criterion(output_s1, target1)
                loss2 = criterion(output_s2, target1)
                output_t1 = F.softmax(output_t1)
                output_t2 = F.softmax(output_t2)
                loss_dis = torch.mean(torch.abs(output_t1-output_t2))
                entropy_loss = -torch.mean(torch.log(torch.mean(output_t1,0)+1e-6))
                entropy_loss -= torch.mean(torch.log(torch.mean(output_t2,0)+1e-6))

                loss_dis.backward()
                optimizer_g.step()

        print('Train Ep: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}\t Dis: {:.6f} Entropy: {:.6f}'.format(
                    epoch, batch_index * batch_size, 12000,
                    100. * batch_index / num_iter, loss1.data.item(),loss2.data.item(),loss_dis.data.item(),entropy_loss.data.item()))

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
                    .format(loss=m_total_loss.avg, loss1=m_loss1.avg, loss2=m_loss2.avg, dis_loss=m_loss_dis.avg, ent_loss=m_entropy_loss.avg)

    print(LoggerInfo)


    return


def Test_MCD(args, G, F1, F2, dataloader):
    G.eval()
    F1.eval()
    F2.eval()
    iter_dataloader = iter(dataloader)
    # Test on Source Domain
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
        Compute_Accuracy(args, output1, label, acc1, prec1, recall1)
        Compute_Accuracy(args, output2, label, acc2, prec2, recall2)

    print('Classifier 1')
    AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc1, prec1, recall1, args.class_num)
    LoggerInfo = AccuracyInfo
    LoggerInfo += '''    Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f}'''.format(acc_avg, prec_avg, recall_avg, f1_avg)
    print(LoggerInfo)

    print('Classifier 2')
    AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc2, prec2, recall2, args.class_num)
    LoggerInfo = AccuracyInfo
    LoggerInfo += '''    Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f}'''.format(acc_avg, prec_avg, recall_avg, f1_avg)
    print(LoggerInfo)

    return

def main():
    """Main."""
    torch.manual_seed(args.seed)

    # Experiment Information
    print_experiment_info(args)

    dataloaders, G, optimizer_g, writer = train_setup(args)

    F1 = ResClassifier(num_layer=1)
    F2 = ResClassifier(num_layer=1)
    F1.cuda()
    F2.cuda()
    optimizer_f = optim.SGD(list(F1.parameters())+list(F2.parameters()),momentum=0.9,lr=0.001,weight_decay=0.0005)

    # Save Best Checkpoint
    Best_Accuracy, Best_Recall = 0, 0

    # Running Experiment
    print("Run Experiment...")
    for epoch in range(1, args.epochs + 1):
        Train_MCD(args, G, F1, F2, dataloaders['train_source'], dataloaders['train_target'], optimizer_g, optimizer_f, epoch, writer)
        print('\nEvaluating train target set:')
        Test_MCD(args, G, F1, F2, dataloaders['train_target'])
        print('\nEvaluating test sets:')
        print('[Test source]')
        Test_MCD(args, G, F1, F2, dataloaders['test_source'])
        print('[Test target]')
        Test_MCD(args, G, F1, F2, dataloaders['test_target'])
        
        torch.save(G.state_dict(), os.path.join(args.out, f'{args.log}_G_{epoch}.pkl'))
        torch.save(F1.state_dict(), os.path.join(args.out, f'{args.log}_F1_{epoch}.pkl'))
        torch.save(F2.state_dict(), os.path.join(args.out, f'{args.log}_F2_{epoch}.pkl'))
    writer.close()


if __name__ == '__main__':
    main()
