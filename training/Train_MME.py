'''
Run using $sh mme_train.sh gpu_id exp_name local_feat
$sh mme_train.sh 0 mme_global False
'''

import time
from utils.Loss import *
from test import Test
from train_setup import *

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def Train_MME(args, model, train_source_dataloader, train_target_dataloader, optimizer, epoch, writer):
    """Train."""
    model.train()
    torch.autograd.set_detect_anomaly(True)

    acc, prec, recall = [AverageMeter() for i in range(args.class_num)], [AverageMeter() for i in
                                                                          range(args.class_num)], [AverageMeter() for i
                                                                                                   in range(
            args.class_num)]
    loss, global_cls_loss, local_cls_loss, mme_loss = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    data_time, batch_time = AverageMeter(), AverageMeter()

    # Decay Learn Rate per Epoch
    if epoch <= 20:
        args.lr = 0.0001
    elif epoch <= 40:
        args.lr = 0.0001
    else:
        args.lr = 0.00001

    optimizer, lr = lr_scheduler_withoutDecay(optimizer, lr=args.lr)

    # Get Source/Target Dataloader iterator
    iter_source_dataloader = iter(train_source_dataloader)
    iter_target_dataloader = iter(train_target_dataloader)

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

        feat_source, out_source, loc_out_source = model(data_source, landmark_source)

        batch_time.update(time.time() - end)

        # Compute Loss on source images
        global_cls_loss_ = nn.CrossEntropyLoss()(out_source.narrow(0, 0, data_source.size(0)), label_source)
        local_cls_loss_ = nn.CrossEntropyLoss()(loc_out_source.narrow(0, 0, data_source.size(0)),
                                                label_source) if args.local_feat else 0

        loss_ = global_cls_loss_ + local_cls_loss_

        # Back Propagation
        optimizer.zero_grad()
        with torch.autograd.detect_anomaly():
            loss_.backward(retain_graph=True)
        optimizer.step()

        feat_target, out_target, loc_out_target = model(data_target, landmark_target)

        coeff = calc_coeff(epoch*num_iter+batch_index, 1.0, 0.0, 10, 10000)
        print(coeff)
        mme_loss_ = MME(model, feat_target, lamda=args.lamda, coeff=coeff)

        optimizer.zero_grad()
        with torch.autograd.detect_anomaly():
            mme_loss_.backward()
        optimizer.step()

        # Compute accuracy, precision and recall
        Compute_Accuracy(args, out_source, label_source, acc, prec, recall)

        # Log loss
        loss.update(float(loss_.cpu().data.item()))
        global_cls_loss.update(float(global_cls_loss_.cpu().data.item()))
        local_cls_loss.update(float(local_cls_loss_.cpu().data.item()) if args.local_feat else 0)
        mme_loss.update(float(mme_loss_.cpu().data.item()))

        writer.add_scalar('Global_Cls_Loss', float(global_cls_loss_.cpu().data.item()),
                          num_iter * (epoch - 1) + batch_index)
        writer.add_scalar('Local_Cls_Loss', float(local_cls_loss_.cpu().data.item()) if args.local_feat else 0,
                          num_iter * (epoch - 1) + batch_index)
        writer.add_scalar('MME_Loss', float(mme_loss_.cpu().data.item()), num_iter * (epoch - 1) + batch_index)

        end = time.time()

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)

    writer.add_scalar('Accuracy', acc_avg, epoch)
    writer.add_scalar('Precision', prec_avg, epoch)
    writer.add_scalar('Recall', recall_avg, epoch)
    writer.add_scalar('F1', f1_avg, epoch)

    LoggerInfo = '''
    [Train]:
    Epoch {0}
    Data Time {data_time.sum:.4f} ({data_time.avg:.4f})
    Batch Time {batch_time.sum:.4f} ({batch_time.avg:.4f})
    Learning Rate {1}\n'''.format(epoch, lr, data_time=data_time, batch_time=batch_time)

    LoggerInfo = '''Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f} 
    Total Loss {loss:.4f} Global Cls Loss {global_cls_loss:.4f} Local Cls Loss {local_cls_loss:.4f} MME Loss {mme_loss:.4f}'''.format(
        acc_avg, prec_avg, recall_avg, f1_avg, loss=loss.avg, global_cls_loss=global_cls_loss.avg,
        local_cls_loss=local_cls_loss.avg, mme_loss=mme_loss.avg)

    print(LoggerInfo)
    return


def main():
    """Main."""
    torch.manual_seed(args.seed)

    # Experiment Information
    print_experiment_info(args)

    dataloaders, model, optimizer, writer = train_setup(args)

    # Save Best Checkpoint
    Best_Accuracy, Best_Recall = 0, 0

    # Running Experiment
    print("Run Experiment...")
    for epoch in range(1, args.epochs + 1):
        Train_MME(args, model, dataloaders['train_source'], dataloaders['train_target'], optimizer, epoch, writer)
        print('\nEvaluating train sets:')
        Test(args, model, dataloaders['train_source'], domain='Source', split='train')
        Best_Accuracy, Best_Recall = Test(args, model, dataloaders['train_target'], Best_Accuracy, Best_Recall,
                                          domain='Target', split='unlabeled train')
        print('\nEvaluating test sets:')
        Test(args, model, dataloaders['test_source'], domain='Source', split='test')
        Test(args, model, dataloaders['test_target'], domain='Target', split='test')

    writer.close()


if __name__ == '__main__':
    main()
