import torch.nn.functional as F
from torch.autograd import Variable

from models.ResNet_feat import ResClassifier
from train_setup import *
from utils.Loss import *
import copy
from Test_MCD_multiple_cls import Test_MCD_cls_tsne

eta = 1.0
num_k = 2 #4

def Train_MCD(args, G, F_dict, train_source_dataloader, train_target_dataloader, optimizer_g, epoch,
              writer, criterion, n_cls):
    """Train."""
    G.train()
    for i in F_dict.keys():
        F_dict[i]['cls'].train()

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

        k1, k2 = random.sample(range(0, n_cls), 2)
        F1_dict = F_dict[k1]
        F2_dict = F_dict[k2]

        optimizer_g.zero_grad()
        F1_dict['optimizer'].zero_grad()
        F2_dict['optimizer'].zero_grad()

        data = Variable(torch.cat((data_source, data_target), 0))
        landmark = Variable(torch.cat((landmark_source, landmark_target), 0))
        label_source = Variable(label_source)

        output = G(data, landmark)
        output1 = F1_dict['cls'](output)
        output2 = F2_dict['cls'](output)

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
        all_loss.backward()
        optimizer_g.step()
        F1_dict['optimizer'].step()
        F2_dict['optimizer'].step()

        # Step B train classifier to maximize discrepancy
        optimizer_g.zero_grad()
        F1_dict['optimizer'].zero_grad()
        F2_dict['optimizer'].zero_grad()

        output = G(data, landmark)
        output1 = F1_dict['cls'](output)
        output2 = F1_dict['cls'](output)
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
        F_loss = loss1 + loss2 - eta * loss_dis + 0.01 * entropy_loss
        F_loss.backward()
        F1_dict['optimizer'].step()
        F2_dict['optimizer'].step()
        # Step C train generator to minimize discrepancy
        for i in range(num_k):
            optimizer_g.zero_grad()
            output = G(data, landmark)
            output1 = F1_dict['cls'](output)
            output2 = F2_dict['cls'](output)

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

    print(f'Last used classifiers {k1}, {k2}\n')

    return

def Test_MCD_tsne(args, G, F_dict, dataloaders, epoch, n_cls, splits=None):
    if splits is None:  # evaluate on test splits by default
        splits = ['test_source', 'test_target']
    G.eval()
    for i in F_dict.keys():
        F_dict[i]['cls'].eval()
    Features = []
    Labels = []


    if True:
        iter_dataloader = iter(dataloaders['train_source'])
        acc1, prec1, recall1 = [AverageMeter() for i in range(args.class_num)], \
                               [AverageMeter() for i in range(args.class_num)], \
                               [AverageMeter() for i in range(args.class_num)]

        for batch_index, (input, landmark, label) in enumerate(iter_dataloader):
            input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
            with torch.no_grad():
                feat = G(input, landmark)
                output = {}
                for i in range(n_cls):
                    output[i] = F.softmax(F_dict[i]['cls'](feat))
                combined_output = (output[0] + output[1] + output[2])/3.0
                
                Features.append (feat.cpu().data.numpy())
                Labels.append (label.cpu().data.numpy()+14)
            Compute_Accuracy(args, combined_output, label, acc1, prec1, recall1)

        print('Multi class combined scores')
        AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc1, prec1, recall1, args.class_num)

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
        
        for batch_index, (input, landmark, label) in enumerate(iter_dataloader):
            input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
            with torch.no_grad():
                feat = G(input, landmark)
                output = {}
                for i in range(n_cls):
                    output[i] = F.softmax(F_dict[i]['cls'](feat))
                combined_output = (output[0] + output[1] + output[2])/3.0
                
                if split == 'test_source':
                    Features_src_test.append (feat.cpu().data.numpy())    
                    Labels_src_test.append (label.cpu().data.numpy())
                if split == 'test_target':    
                    Features_tar_test.append (feat.cpu().data.numpy())    
                    Labels_tar_test.append (label.cpu().data.numpy()+7)
                if split == 'train_target':  
                    Features_tar_train.append (feat.cpu().data.numpy())    
                    Labels_tar_train.append (label.cpu().data.numpy()+7)

            Compute_Accuracy(args, combined_output, label, acc1, prec1, recall1)

        print('Multi class combined scores')
        AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc1, prec1, recall1, args.class_num)


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
    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=20, gamma=0.5, verbose=True)
    

    F_dict = {}
    n_cls=3
    for i in range(n_cls):
        Fi = {'cls': ResClassifier(num_classes=args.class_num, num_layer=1)}
        Fi['cls'].cuda()
        Fi['optimizer'] = optim.SGD(Fi['cls'].parameters(), momentum=0.9, lr=0.001, weight_decay=0.0005)
        Fi['scheduler'] = optim.lr_scheduler.StepLR(Fi['optimizer'], step_size=20, gamma=0.1, verbose=True)
        F_dict[i] = Fi

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
         #[0.65831665 3.01150101 1.13164193 0.20750166 0.45330163 1.18126904 0.35646808]
        per_cls_weights = [1.75, 3.0, 2.0, 1.0, 1.5, 2.0, 1.25]
        print(per_cls_weights)
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
    
    def get_drw_weights(args, epoch, cls_num_list):
        if True:
            idx = 0 if epoch <= 5 else 1
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
        return per_cls_weights

    print(f'Using {args.criterion} loss')

    # Running Experiment
    print("Run Experiment...")
    for epoch in range(1, args.epochs + 1):
        # if epoch < 5 and args.criterion == 'weighted_focal': #Try delayed reweighting
        #     criterion = FocalLoss(gamma=1)
        if args.criterion=='ldam':
            if False: #epoch >5:
                per_cls_weights = [1.75, 3.0, 2.0, 1.0, 1.5, 2.0, 1.25]
            else: 
                per_cls_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            # per_cls_weights = get_drw_weights(args, epoch, cls_num_list)
            print(f'Epoch: {epoch}, per cls weights: {per_cls_weights}')
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
            criterion = LDAMLoss(cls_num_list, weight=per_cls_weights)

        print(f'Epoch : {epoch}')

        Train_MCD(args, G, F_dict, dataloaders['train_source'], dataloaders['train_target'], optimizer_g,
                  epoch, writer, criterion, n_cls)
        scheduler_g.step()
        for i in range(n_cls):
            F_dict[i]['scheduler'].step()

        print('\nEvaluation ...')
        Test_MCD_tsne(args, G, F_dict, dataloaders, epoch, n_cls, splits=['test_source', 'train_target', 'test_target'])
        Test_MCD_cls_tsne(args, G, F_dict, dataloaders, epoch, n_cls, splits=['test_target'])

        if args.save_checkpoint and epoch%5:
            torch.save(G.state_dict(), os.path.join(args.out, f'ckpts/MCD_G_{epoch}.pkl'))
            for i in range(n_cls):
                torch.save(F_dict[i]['cls'].state_dict(), os.path.join(args.out, f'ckpts/F{i}_{epoch}.pkl'))
    writer.close()


if __name__ == '__main__':
    main()
