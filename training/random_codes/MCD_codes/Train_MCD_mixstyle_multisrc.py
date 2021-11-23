import torch.nn.functional as F
from torch.autograd import Variable

from models.ResNet_feat import ResClassifier
from train_setup import *
from utils.Loss import *
from models.mixstyle import activate_mixstyle, deactivate_mixstyle
import copy

eta = 1.0
num_k = 2 #4


def Train_MCD(args, G, F1, F2, train_source1_dataloader, train_source2_dataloader, train_target_dataloader, optimizer_g, optimizer_f, epoch,
              writer, criterion):
    """Train."""
    G.train()
    F1.train()
    F2.train()
    # torch.autograd.set_detect_anomaly(True)
    batch_size = args.train_batch

    m_total_loss, m_loss1, m_loss2, m_loss_dis, m_entropy_loss = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    # Get Source/Target Dataloader iterator
    iter_source1_dataloader = iter(train_source1_dataloader)
    iter_source2_dataloader = iter(train_source2_dataloader)
    iter_target_dataloader = iter(train_target_dataloader)

    num_iter = len(train_source1_dataloader) if (len(train_source1_dataloader) > len(train_target_dataloader)) else len(
        train_target_dataloader)

    for batch_index in range(num_iter):
        try:
            data_source1, landmark_source1, label_source1 = iter_source1_dataloader.next()
        except:
            iter_source1_dataloader = iter(train_source1_dataloader)
            data_source1, landmark_source1, label_source1 = iter_source1_dataloader.next()
        
        try:
            data_source2, landmark_source2, label_source2 = iter_source2_dataloader.next()
        except:
            iter_source1_dataloader = iter(train_source1_dataloader)
            data_source2, landmark_source2, label_source2 = iter_source2_dataloader.next()

        data_source = Variable(torch.cat((data_source1, data_source2), 0))
        landmark_source = Variable(torch.cat((landmark_source1, landmark_source2), 0))
        label_source = Variable(torch.cat((label_source1, label_source2), 0))
        perm = torch.randperm(args.train_batch)
        data_source = data_source[perm]
        landmark_source = landmark_source[perm]
        label_source = label_source[perm]        

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

        G.apply(activate_mixstyle)
        output = G(data, landmark)
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

        output = G(data, landmark)
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
        F_loss = loss1 + loss2 + args.lamda_ent * entropy_loss
        if epoch>0:
            F_loss += - eta * loss_dis
        F_loss.backward()
        optimizer_f.step()
        # Step C train generator to minimize discrepancy
        if epoch>0:
            for i in range(num_k):
                G.apply(deactivate_mixstyle)
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



def Test_MCD_tsne(args, G, F1, F2, dataloaders, epoch, splits=None):
    if splits is None:  # evaluate on test splits by default
        splits = ['test_source', 'test_target']
    G.eval()
    G.apply(deactivate_mixstyle)
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
                feat = G(input, landmark)
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

    dataloaders, G, optimizer_g, writer = train_setup(args)

    args.train_batch=16
    dataloaders['train_source1'] = BuildDataloader(args, split='train', domain='source', max_samples=args.source_labeled)
    args.source='CK+'
    dataloaders['train_source2'] = BuildDataloader(args, split='train', domain='source', max_samples=args.source_labeled)
    args.source='RAF'
    args.train_batch=32


    optimizer_g, lr = lr_scheduler_withoutDecay(optimizer_g, lr=args.lr)
    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=20, gamma=0.1, verbose=True)
    
    F1 = ResClassifier(num_classes=args.class_num, num_layer=1)
    F2 = ResClassifier(num_classes=args.class_num, num_layer=1)
    F1.cuda()
    F2.cuda()
    optimizer_f = optim.SGD(list(F1.parameters()) + list(F2.parameters()), momentum=0.9, lr=0.001, weight_decay=0.0005)
    scheduler_f = optim.lr_scheduler.StepLR(optimizer_f, step_size=20, gamma=0.1, verbose=True)

    # G_ckpt= os.path.join(args.out, f'ckpts/MCD_G.pkl')
    # if os.path.exists(G_ckpt):
    #     checkpoint = torch.load (G_ckpt, map_location='cuda')
    #     G.load_state_dict (checkpoint, strict=False)

    # F1_ckpt= os.path.join(args.out, f'ckpts/MCD_F1.pkl')
    # if os.path.exists(F1_ckpt):
    #     checkpoint = torch.load (F1_ckpt, map_location='cuda')
    #     F1.load_state_dict (checkpoint, strict=False)

    # F2_ckpt= os.path.join(args.out, f'ckpts/MCD_F2.pkl')
    # if os.path.exists(F2_ckpt):
    #     checkpoint = torch.load (F2_ckpt, map_location='cuda')
    #     F2.load_state_dict (checkpoint, strict=False)

    if args.show_feat:
        G_ckpt= os.path.join(args.out, f'ckpts/MCD_G.pkl')
        if os.path.exists(G_ckpt):
            checkpoint = torch.load (G_ckpt, map_location='cuda')
            G.load_state_dict (checkpoint, strict=False)

        F1_ckpt= os.path.join(args.out, f'ckpts/MCD_F1.pkl')
        if os.path.exists(F1_ckpt):
            checkpoint = torch.load (F1_ckpt, map_location='cuda')
            F1.load_state_dict (checkpoint, strict=False)

        F2_ckpt= os.path.join(args.out, f'ckpts/MCD_F2.pkl')
        if os.path.exists(F2_ckpt):
            checkpoint = torch.load (F2_ckpt, map_location='cuda')
            F2.load_state_dict (checkpoint, strict=False)
        Test_MCD_tsne(args, G, F1, F2, dataloaders, 30, splits=['test_source', 'train_target', 'test_target'])
        return

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
            if epoch >4:
                per_cls_weights = [1.75, 3.0, 2.0, 1.0, 1.5, 2.0, 1.25]
            else: 
                per_cls_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            # per_cls_weights = get_drw_weights(args, epoch, cls_num_list)
            print(f'Epoch: {epoch}, per cls weights: {per_cls_weights}')
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
            criterion = LDAMLoss(cls_num_list, weight=per_cls_weights)

        print(f'Epoch : {epoch}')

        Train_MCD(args, G, F1, F2, dataloaders['train_source1'], dataloaders['train_source2'], dataloaders['train_target'], optimizer_g, optimizer_f,
                  epoch, writer, criterion)
        scheduler_g.step()
        scheduler_f.step()
        print('\nEvaluation ...')
        Test_MCD_tsne(args, G, F1, F2, dataloaders, epoch, splits=['test_source', 'train_target', 'test_target'])

        # Test_MCD(args, G, F1, F2, dataloaders, epoch, splits=['train_source', 'test_source'])
        # Test_MCD(args, G, F1, F2, dataloaders, epoch, splits=['train_source', 'train_target'])
        # Test_MCD(args, G, F1, F2, dataloaders, epoch, splits=['train_source', 'test_target'])
        # Test_MCD(args, G, F1, F2, dataloaders, epoch, splits=['train_source', 'train_target', 'test_source', 'test_target'])
        if args.save_checkpoint and epoch%5:
            torch.save(G.state_dict(), os.path.join(args.out, f'ckpts/MCD_G_{epoch}.pkl'))
            torch.save(F1.state_dict(), os.path.join(args.out, f'ckpts/MCD_F1_{epoch}.pkl'))
            torch.save(F2.state_dict(), os.path.join(args.out, f'ckpts/MCD_F2_{epoch}.pkl'))
    writer.close()


if __name__ == '__main__':
    main()
