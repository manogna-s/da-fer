from torch.utils.tensorboard import SummaryWriter

from models.AdversarialNetwork import calc_coeff
from models.GCN_utils import *
from utils.Loss import Entropy, DANN, CDAN, HAFN, SAFN
from test import Test
from train_setup import *


def Train_DANN(args, model, ad_net, random_layer, train_source_dataloader, train_target_dataloader, optimizer, optimizer_ad,
          epoch, writer):
    """Train."""

    model.train()
    torch.autograd.set_detect_anomaly(True)

    acc, prec, recall = [AverageMeter() for i in range(args.class_num)], \
                        [AverageMeter() for i in range(args.class_num)], \
                        [AverageMeter() for i in range(args.class_num)]
    loss, global_cls_loss, local_cls_loss, afn_loss, dan_loss = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    data_time, batch_time = AverageMeter(), AverageMeter()

    if args.use_dan:
        num_ADNet = 0
        ad_net.train()

    # Decay Learn Rate per Epoch
    if epoch <= 20:
        args.lr, args.lr_ad = 0.0001, 0.001
    elif epoch <= 40:
        args.lr, args.lr_ad = 0.0001, 0.0001
    else:
        args.lr, args.lr_ad = 0.00001, 0.00001

    optimizer, lr = lr_scheduler_withoutDecay(optimizer, lr=args.lr)
    if args.use_dan:
        optimizer_ad, lr_ad = lr_scheduler_withoutDecay(optimizer_ad, lr=args.lr_ad)

    # Get Source/Target Dataloader iterator
    iter_source_dataloader = iter(train_source_dataloader)
    iter_target_dataloader = iter(train_target_dataloader)

    # len(data_loader) = math.ceil(len(data_loader.dataset)/batch_size)
    num_iter = len(train_source_dataloader) if (len(train_source_dataloader) > len(train_target_dataloader)) else len(
        train_target_dataloader)

    end = time.time()
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

        data_time.update(time.time() - end)

        data_source, landmark_source, label_source = data_source.cuda(), landmark_source.cuda(), label_source.cuda()
        data_target, landmark_target, label_target = data_target.cuda(), landmark_target.cuda(), label_target.cuda()

        # Forward Propagation
        end = time.time()
        feature, output, loc_output = model(torch.cat((data_source, data_target), 0),
                                            torch.cat((landmark_source, landmark_target), 0), args.useClassify)
        feat_target = feature[args.train_batch:, :]
        batch_time.update(time.time() - end)

        # Compute Loss
        global_cls_loss_ = nn.CrossEntropyLoss()(output.narrow(0, 0, data_source.size(0)), label_source)
        local_cls_loss_ = nn.CrossEntropyLoss()(loc_output.narrow(0, 0, data_source.size(0)),
                                                label_source) if args.local_feat else 0

        afn_loss_ = (HAFN(feature, args.w_l2, args.r) if args.afn_method == 'HAFN' else SAFN(feature, args.w_l2,
                                                                                             args.dr)) if args.use_afn else 0

        if args.use_dan:
            softmax_output = nn.Softmax(dim=1)(output)
            if args.dan_method == 'CDAN-E':
                entropy = Entropy(softmax_output)
                dan_loss_ = CDAN([feature, softmax_output], ad_net, entropy,
                                 calc_coeff(num_iter * (epoch - 1) + batch_index), random_layer)
            elif args.dan_method == 'CDAN':
                dan_loss_ = CDAN([feature, softmax_output], ad_net, None, None, random_layer)
            elif args.dan_method == 'DANN':
                dan_loss_ = args.lamda * DANN(feature, ad_net)
        else:
            dan_loss_ = 0

        loss_ = global_cls_loss_ + local_cls_loss_

        if args.use_afn:
            loss_ += afn_loss_

        if args.use_dan:
            loss_ += dan_loss_

        # Log Adversarial Network Accuracy
        if args.use_dan:
            if args.dan_method == 'CDAN' or args.dan_method == 'CDAN-E':
                softmax_output = nn.Softmax(dim=1)(output)
                if args.rand_layer:
                    random_out = random_layer.forward([feature, softmax_output])
                    adnet_output = ad_net(random_out.view(-1, random_out.size(1)))
                else:
                    op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
                    adnet_output = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
            elif args.dan_method == 'DANN':
                adnet_output = ad_net(feature)

            adnet_output = adnet_output.cpu().data.numpy()
            adnet_output[adnet_output > 0.5] = 1
            adnet_output[adnet_output <= 0.5] = 0
            num_ADNet += np.sum(adnet_output[:args.train_batch]) + (
                        args.train_batch - np.sum(adnet_output[args.train_batch:]))

        # Back Propagation
        optimizer.zero_grad()
        if args.use_dan:
            optimizer_ad.zero_grad()

        with torch.autograd.detect_anomaly():
            loss_.backward()

        optimizer.step()

        if args.use_dan:
            optimizer_ad.step()

        # Compute accuracy, precision and recall
        Compute_Accuracy(args, output.narrow(0, 0, data_source.size(0)), label_source, acc, prec, recall)

        # Log loss
        loss.update(float(loss_.cpu().data.item()))
        global_cls_loss.update(float(global_cls_loss_.cpu().data.item()))
        local_cls_loss.update(float(local_cls_loss_.cpu().data.item()) if args.local_feat else 0)
        afn_loss.update(float(afn_loss_.cpu().data.item()) if args.use_afn else 0)
        dan_loss.update(float(dan_loss_.cpu().data.item()) if args.use_dan else 0)

        writer.add_scalar('Glocal_Cls_Loss', float(global_cls_loss_.cpu().data.item()),
                          num_iter * (epoch - 1) + batch_index)
        writer.add_scalar('Local_Cls_Loss', float(local_cls_loss_.cpu().data.item()) if args.local_feat else 0,
                          num_iter * (epoch - 1) + batch_index)
        writer.add_scalar('AFN_Loss', float(afn_loss_.cpu().data.item()) if args.use_afn else 0,
                          num_iter * (epoch - 1) + batch_index)
        writer.add_scalar('DAN_Loss', float(dan_loss_.cpu().data.item()) if args.use_dan else 0,
                          num_iter * (epoch - 1) + batch_index)

        end = time.time()

    LoggerInfo = '''\n
    [Train on Source and unlabeled target]: 
    Epoch {0}
    Learning Rate {1} Learning Rate(AdversarialNet) {2}\n'''.format(epoch, lr, lr_ad if args.use_dan else 0)

    print(LoggerInfo)

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)

    writer.add_scalar('Accuracy', acc_avg, epoch)
    writer.add_scalar('Precision', prec_avg, epoch)
    writer.add_scalar('Recall', recall_avg, epoch)
    writer.add_scalar('F1', f1_avg, epoch)

    if args.use_dan:
        writer.add_scalar('AdversarialNetwork_Accuracy', num_ADNet / (2.0 * args.train_batch * num_iter), epoch)

    LoggerInfo = '''    AdversarialNet Acc {0:.4f} Acc_avg {1:.4f} Prec_avg {2:.4f} Recall_avg {3:.4f} F1_avg {4:.4f}
    Total Loss {loss:.4f} Global Cls Loss {global_cls_loss:.4f} Local Cls Loss {local_cls_loss:.4f} AFN Loss {afn_loss:.4f} DAN Loss {dan_loss:.4f}'''.format(
        num_ADNet / (2.0 * args.train_batch * num_iter) if args.use_dan else 0, acc_avg, prec_avg, recall_avg, f1_avg,
        loss=loss.avg, global_cls_loss=global_cls_loss.avg, local_cls_loss=local_cls_loss.avg,
        afn_loss=afn_loss.avg if args.use_afn else 0, dan_loss=dan_loss.avg if args.use_dan else 0)

    print(LoggerInfo)
    return


def main():
    """Main."""
    # Parse Argument
    torch.manual_seed(args.seed)

    # Experiment Information
    print_experiment_info(args)

    dataloaders, model, optimizer, writer = train_setup(args)

    # Bulid Adversarial Network
    print('Building Adversarial Network...')
    random_layer, ad_net = BuildAdversarialNetwork(args, model.output_num(), args.class_num) if args.use_dan else (
    None, None)

    param_optim_ad = Set_Param_Optim(args, ad_net) if args.use_dan else None
    optimizer_ad = Set_Optimizer(args, param_optim_ad, args.lr, args.weight_decay,
                                 args.momentum) if args.use_dan else None
    print('Done!')
    print('================================================')

    # Save Best Checkpoint
    Best_Accuracy, Best_Recall = 0, 0

    # Running Experiment
    print("Run Experiment...")

    for epoch in range(1, args.epochs + 1):
        if args.show_feat and epoch % 5 == 1:
            VizFeatures(args, epoch, model, dataloaders)

        if args.use_gcn and args.use_cluster and epoch % 10 == 0:
            Initialize_Mean_Cluster(args, model, False)
            torch.cuda.empty_cache()

        Train_DANN(args, model, ad_net, random_layer, dataloaders['train_source'], dataloaders['train_target'], optimizer,
                  optimizer_ad, epoch, writer)
        print('\n[Testing...]')
        Test(args, model, dataloaders['train_source'], domain='Source', split='train')
        Best_Accuracy, Best_Recall = Test(args, model, dataloaders['train_target'], Best_Accuracy, Best_Recall,
                                          domain='Target', split='unlabeled train')
        Test(args, model, dataloaders['test_source'], domain='Source', split='test')
        Test(args, model, dataloaders['test_target'], domain='Target', split='test')

    writer.close()


if __name__ == '__main__':
    main()
