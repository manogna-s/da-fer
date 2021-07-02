from models.GCN_utils import *
from utils.Loss import HAFN, SAFN
from test import Test
from train_setup import *


def TrainOnSource(args, model, train_dataloader, optimizer, epoch, writer):
    """Train."""

    model.train()
    torch.autograd.set_detect_anomaly(True)

    acc, prec, recall = [AverageMeter() for i in range(args.class_num)], \
                        [AverageMeter() for i in range(args.class_num)], \
                        [AverageMeter() for i in range(args.class_num)]
    loss, global_cls_loss, local_cls_loss, afn_loss, data_time, batch_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    # Decay Learn Rate per Epoch
    if args.net in ['ResNet18', 'ResNet50']:
        if epoch <= 20:
            args.lr = 1e-4
        elif epoch <= 40:
            args.lr = 1e-5
        else:
            args.lr = 1e-6

    elif args.net == 'MobileNet':
        if epoch <= 20:
            args.lr = 1e-3
        elif epoch <= 40:
            args.lr = 1e-4
        elif epoch <= 60:
            args.lr = 1e-5
        else:
            args.lr = 1e-6

    elif args.net == 'VGGNet':
        if epoch <= 30:
            args.lr = 1e-3
        elif epoch <= 60:
            args.lr = 1e-4
        elif epoch <= 70:
            args.lr = 1e-5
        else:
            args.lr = 1e-6

    end = time.time()
    for step, (input, landmark, label) in enumerate(train_dataloader):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        data_time.update(time.time() - end)

        # Forward propagation
        end = time.time()
        feature, output, loc_output = model(input, landmark)
        batch_time.update(time.time() - end)

        # Compute Loss
        global_cls_loss_ = nn.CrossEntropyLoss()(output, label)
        local_cls_loss_ = nn.CrossEntropyLoss()(loc_output, label) if args.local_feat else 0
        afn_loss_ = (HAFN(feature, args.weight_L2norm, args.radius) if args.methodOfAFN == 'HAFN' else SAFN(feature,
                                                                                                            args.weight_L2norm,
                                                                                                            args.deltaRadius)) if args.useAFN else 0
        loss_ = global_cls_loss_ + local_cls_loss_ + (afn_loss_ if args.useAFN else 0)

        # Back Propagation
        optimizer.zero_grad()

        with torch.autograd.detect_anomaly():
            loss_.backward()

        optimizer.step()

        # Decay Learn Rate
        optimizer, lr = lr_scheduler_withoutDecay(optimizer, lr=args.lr,
                                                  weight_decay=args.weight_decay)
        # optimizer = lr_scheduler(optimizer, num_iter*(epoch-1)+step, 0.001, 0.75, lr=args.lr, weight_decay=args.weight_decay)

        # Compute accuracy, recall and loss
        Compute_Accuracy(args, output, label, acc, prec, recall)

        # Log loss
        loss.update(float(loss_.cpu().data.item()))
        global_cls_loss.update(float(global_cls_loss_.cpu().data.item()))
        local_cls_loss.update(float(local_cls_loss_.cpu().data.item()) if args.local_feat else 0)
        afn_loss.update(float(afn_loss_.cpu().data.item()) if args.useAFN else 0)

        end = time.time()

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)

    writer.add_scalar('Accuracy', acc_avg, epoch)
    writer.add_scalar('Precision', prec_avg, epoch)
    writer.add_scalar('Recall', recall_avg, epoch)
    writer.add_scalar('F1', f1_avg, epoch)

    writer.add_scalar('Global_Cls_Loss', global_cls_loss.avg, epoch)
    writer.add_scalar('Local_Cls_Loss', local_cls_loss.avg, epoch)
    writer.add_scalar('AFN_Loss', afn_loss.avg, epoch)

    LoggerInfo = '''
    \n[Train]: 
    Epoch {0}
    Learning Rate {1}\n'''.format(epoch, args.lr)

    LoggerInfo += AccuracyInfo

    LoggerInfo += '''    Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f}
    Total Loss {loss:.4f} Global Cls Loss {global_cls_loss:.4f} Local Cls Loss {local_cls_loss:.4f} AFN Loss {afn_loss:.4f}\n'''.format(
        acc_avg, prec_avg, recall_avg, f1_avg, loss=loss.avg, global_cls_loss=global_cls_loss.avg,
        local_cls_loss=local_cls_loss.avg, afn_loss=afn_loss.avg)

    print(LoggerInfo)
    return


def main():
    """Main."""

    # Parse Argument
    torch.manual_seed(args.seed)

    # Experiment Information
    print_experiment_info(args)

    dataloaders, model, optimizer, writer = train_setup(args)

    # Save Best Checkpoint
    Best_Accuracy = 0
    Best_Recall = 0

    # Running Experiment
    print("Run Experiment...")

    for epoch in range(1, args.epochs + 1):
        if args.showFeature and epoch % 5 == 1:
            VizFeatures(args, epoch, model, dataloaders)

        if args.use_gcn and args.use_cluster and epoch % 10 == 0:
            Initialize_Mean_Cluster(args, model, True)
            torch.cuda.empty_cache()
        TrainOnSource(args, model, dataloaders['train_source'], optimizer, epoch, writer)

        print('[Testing]')
        print('\nEvaluating train sets:')
        Best_Accuracy, Best_Recall = Test(args, model, dataloaders['train_source'], Best_Accuracy, Best_Recall,
                                          domain='Target', split='unlabeled train')
        Test(args, model, dataloaders['train_target'], Best_Accuracy, Best_Recall, domain='Source', split='train')

        print('\nEvaluating test sets:')
        Test(args, model, dataloaders['test_source'], Best_Accuracy, Best_Recall, domain='Source', split='test')
        Test(args, model, dataloaders['test_target'], Best_Accuracy, Best_Recall, domain='Target', split='test')

        torch.cuda.empty_cache()

    writer.close()


if __name__ == '__main__':
    main()
