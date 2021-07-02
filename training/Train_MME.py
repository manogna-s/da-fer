'''
Run using $sh mme_train.sh gpu_id exp_name local_feat
$sh mme_train.sh 0 mme_only_global False
'''

from torch.utils.tensorboard import SummaryWriter

from utils.Loss import *
from utils.Utils import *
from eval import Test
from train_args import args

# parser = argparse.ArgumentParser(description='Domain adaptation for Expression Classification')
#
# parser.add_argument('--log', type=str, help='Log Name')
# parser.add_argument('--out', type=str, help='Output Path')
# parser.add_argument('--net', type=str, default='ResNet50', choices=['ResNet18', 'ResNet50', 'VGGNet', 'MobileNet'])
# parser.add_argument('--pretrained', type=str, help='pretrained', default='None')
# parser.add_argument('--dev', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
#
# parser.add_argument('--use_mme', type=str2bool, default=True, help='whether to use MME loss')
#
# parser.add_argument('--use_afn', type=str2bool, default=False, help='whether to use AFN Loss')
# parser.add_argument('--afn_method', type=str, default='SAFN', choices=['HAFN', 'SAFN'])
# parser.add_argument('--r', type=float, default=25.0, help='radius of HAFN (default: 25.0)')
# parser.add_argument('--dr', type=float, default=1.0, help='radius of SAFN (default: 1.0)')
# parser.add_argument('--w_l2', type=float, default=0.05, help='weight L2 norm of AFN (default: 0.05)')
#
# parser.add_argument('--face_scale', type=int, default=112, help='Scale of face (default: 112)')
# parser.add_argument('--source', type=str, default='RAF', choices=['RAF', 'RAF_7class', 'AFED', 'MMI'])
# parser.add_argument('--target', type=str, default='AISIN',
#                     choices=['RAF', 'CK+', 'JAFFE', 'MMI', 'Oulu-CASIA', 'SFEW', 'FER2013', 'ExpW', 'AFED', 'WFED',
#                              'AISIN'])
# parser.add_argument('--train_batch', type=int, default=64, help='input batch size for training (default: 64)')
# parser.add_argument('--test_batch', type=int, default=64, help='input batch size for testing (default: 64)')
# parser.add_argument('--num_unlabeled', type=int, default=1700,
#                     help='number of unlabeled samples (default: -1 == all samples)')
# parser.add_argument('--lamda', type=float, default=0.1)
# parser.add_argument('--lr', type=float, default=0.0001)
#
# parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 10)')
# parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
# parser.add_argument('--weight_decay', type=float, default=0.0005, help='SGD weight decay (default: 0.0005)')
#
# parser.add_argument('--local_feat', type=str2bool, default=True, help='whether to use Local Feature')
#
# parser.add_argument('--intra_gcn', type=str2bool, default=False, help='whether to use Intra-GCN')
# parser.add_argument('--inter_gcn', type=str2bool, default=False, help='whether to use Inter-GCN')
# parser.add_argument('--rand_mat', type=str2bool, default=False, help='whether to use Random Matrix')
# parser.add_argument('--all1_mat', type=str2bool, default=False, help='whether to use All One Matrix')
# parser.add_argument('--use_cov', type=str2bool, default=False, help='whether to use Cov')
# parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
# parser.add_argument('--rand_layer', type=str2bool, default=False, help='whether to use random')
# parser.add_argument('--use_cluster', type=str2bool, default=False, help='whether to use Cluster')
# parser.add_argument('--method', type=str, default="CADA", help='Choose the method of the experiment')
#
# parser.add_argument('--class_num', type=int, default=2, help='number of class (default: 7)')


def Train_MME(args, model, train_source_dataloader, train_target_dataloader, optimizer, epoch, writer):
    """Train."""
    model.train()
    torch.autograd.set_detect_anomaly(True)

    acc, prec, recall = [AverageMeter() for i in range(args.class_num)], [AverageMeter() for i in
                                                                          range(args.class_num)], [AverageMeter() for i
                                                                                                   in range(
            args.class_num)]
    loss, global_cls_loss, local_cls_loss, afn_loss, mme_loss = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
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

        afn_loss_ = (HAFN(feat_source, args.w_l2, args.r) if args.afn_method == 'HAFN' else SAFN(feat_source, args.w_l2,
                                                                                                 args.dr)) if args.use_afn else 0

        loss_ = global_cls_loss_ + local_cls_loss_

        if args.use_afn:
            loss_ += afn_loss_

        # Back Propagation
        optimizer.zero_grad()
        with torch.autograd.detect_anomaly():
            loss_.backward(retain_graph=True)
        optimizer.step()

        feat_target, out_target, loc_out_target = model(data_target, landmark_target)
        mme_loss_ = MME(model, feat_target, lamda=args.lamda)

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
        afn_loss.update(float(afn_loss_.cpu().data.item()) if args.use_afn else 0)
        mme_loss.update(float(mme_loss_.cpu().data.item()))

        writer.add_scalar('Global_Cls_Loss', float(global_cls_loss_.cpu().data.item()),
                          num_iter * (epoch - 1) + batch_index)
        writer.add_scalar('Local_Cls_Loss', float(local_cls_loss_.cpu().data.item()) if args.local_feat else 0,
                          num_iter * (epoch - 1) + batch_index)
        writer.add_scalar('AFN_Loss', float(afn_loss_.cpu().data.item()) if args.use_afn else 0,
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

    LoggerInfo += AccuracyInfo

    LoggerInfo += '''Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f} 
    Total Loss {loss:.4f} Global Cls Loss {global_cls_loss:.4f} Local Cls Loss {local_cls_loss:.4f} AFN Loss {afn_loss:.4f} MME Loss {mme_loss:.4f}'''.format(
        acc_avg, prec_avg, recall_avg, f1_avg, loss=loss.avg, global_cls_loss=global_cls_loss.avg,
        local_cls_loss=local_cls_loss.avg, afn_loss=afn_loss.avg if args.use_afn else 0, mme_loss=mme_loss.avg)

    print(LoggerInfo)
    return


# def Test(args, model, test_source_dataloader, test_target_dataloader, Best_Accuracy, Best_Recall, epoch, writer):
#     """Test."""
#
#     model.eval()
#     torch.autograd.set_detect_anomaly(True)
#
#     iter_source_dataloader = iter(test_source_dataloader)
#     iter_target_dataloader = iter(test_target_dataloader)
#
#     # Test on Source Domain
#     acc, prec, recall = [AverageMeter() for i in range(args.class_num)], [AverageMeter() for i in
#                                                                           range(args.class_num)], [AverageMeter() for i
#                                                                                                    in range(
#             args.class_num)]
#     loss, data_time, batch_time = AverageMeter(), AverageMeter(), AverageMeter()
#
#     end = time.time()
#     for batch_index, (input, landmark, target) in enumerate(iter_source_dataloader):
#         data_time.update(time.time() - end)
#
#         input, landmark, target = input.cuda(), landmark.cuda(), target.cuda()
#
#         with torch.no_grad():
#             end = time.time()
#             feature, output, loc_output = model(input, landmark)
#             batch_time.update(time.time() - end)
#
#         loss_ = nn.CrossEntropyLoss()(output, target)
#
#         # Compute accuracy, precision and recall
#         Compute_Accuracy(args, output, target, acc, prec, recall)
#
#         # Log loss
#         loss.update(float(loss_.cpu().data.numpy()))
#
#         end = time.time()
#
#     AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)
#
#     writer.add_scalar('Test_Recall_SourceDomain', recall_avg, epoch)
#     writer.add_scalar('Test_Accuracy_SourceDomain', acc_avg, epoch)
#
#     LoggerInfo = '''
#     [Test (Source Domain)]:
#     Data Time {data_time.sum:.4f} ({data_time.avg:.4f})
#     Batch Time {batch_time.sum:.4f} ({batch_time.avg:.4f})\n'''.format(data_time=data_time, batch_time=batch_time)
#
#     LoggerInfo += AccuracyInfo
#
#     LoggerInfo += '''    Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f}
#     Loss {loss:.4f}'''.format(acc_avg, prec_avg, recall_avg, f1_avg, loss=loss.avg)
#
#     print(LoggerInfo)
#
#     # Test on Target Domain
#     acc, prec, recall = [AverageMeter() for i in range(args.class_num)], [AverageMeter() for i in
#                                                                           range(args.class_num)], [AverageMeter() for i
#                                                                                                    in range(
#             args.class_num)]
#     loss, data_time, batch_time = AverageMeter(), AverageMeter(), AverageMeter()
#
#     end = time.time()
#     for batch_index, (input, landmark, target) in enumerate(iter_target_dataloader):
#         data_time.update(time.time() - end)
#
#         input, landmark, target = input.cuda(), landmark.cuda(), target.cuda()
#
#         with torch.no_grad():
#             end = time.time()
#             feature, output, loc_output = model(input, landmark)
#             batch_time.update(time.time() - end)
#
#         loss_ = nn.CrossEntropyLoss()(output, target)
#
#         # Compute accuracy, precision and recall
#         Compute_Accuracy(args, output, target, acc, prec, recall)
#
#         # Log loss
#         loss.update(float(loss_.cpu().data.numpy()))
#
#         end = time.time()
#
#     AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)
#
#     writer.add_scalar('Test_Recall_TargetDomain', recall_avg, epoch)
#     writer.add_scalar('Test_Accuracy_TargetDomain', acc_avg, epoch)
#
#     LoggerInfo = '''
#     [Test (Target Domain)]:
#     Data Time {data_time.sum:.4f} ({data_time.avg:.4f})
#     Batch Time {batch_time.sum:.4f} ({batch_time.avg:.4f})\n'''.format(data_time=data_time, batch_time=batch_time)
#
#     LoggerInfo += AccuracyInfo
#
#     LoggerInfo += '''    Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f}
#     Loss {loss:.4f}'''.format(acc_avg, prec_avg, recall_avg, f1_avg, loss=loss.avg)
#
#     print(LoggerInfo)
#
#     # Save Checkpoints
#     if recall_avg > Best_Recall:
#         Best_Recall = recall_avg
#         print('[Save] Best Recall: %.4f.' % Best_Recall)
#
#         if isinstance(model, nn.DataParallel):
#             torch.save(model.module.state_dict(), os.path.join(args.out, '{}_Recall.pkl'.format(args.log)))
#         else:
#             torch.save(model.state_dict(), os.path.join(args.out, '{}_Recall.pkl'.format(args.log)))
#
#     if acc_avg > Best_Accuracy:
#         Best_Accuracy = acc_avg
#         print('[Save] Best Accuracy: %.4f.' % Best_Accuracy)
#
#         if isinstance(model, nn.DataParallel):
#             torch.save(model.module.state_dict(), os.path.join(args.out, '{}_Accuracy.pkl'.format(args.log)))
#         else:
#             torch.save(model.state_dict(), os.path.join(args.out, '{}_Accuracy.pkl'.format(args.log)))
#
#     return Best_Accuracy, Best_Recall


def main():
    """Main."""
    torch.manual_seed(args.seed)

    # Experiment Information
    # print(args)
    print('Log Name: %s' % args.log)
    print('Output Path: %s' % args.out)
    print('Backbone: %s' % args.net)
    print('Resume Model: %s' % args.pretrained)
    print('SourceDataset: %s' % args.source)
    print('TargetDataset: %s' % args.target)
    print('Number of classes : %d' % args.class_num)

    if not args.local_feat:
        print('Only use global feature.')
    else:
        print('Use global feature and local feature.')

    # Bulid Dataloder
    print("Building Train and Test Dataloader...")
    train_source_dataloader = BulidDataloader(args, flag1='train', flag2='source')
    train_source_dataloader = BulidDataloader(args, flag1='train', flag2='source', max_samples=args.source_labeled)
    train_target_dataloader = BulidDataloader(args, flag1='train', flag2='target', max_samples=args.target_unlabeled)
    test_source_dataloader = BulidDataloader(args, flag1='test', flag2='source')
    test_target_dataloader = BulidDataloader(args, flag1='test', flag2='target')
    print('Done!')

    print('================================================')

    # Bulid Model
    print('Building Model...')
    model = Build_Backbone(args)
    print('Done!')
    print('================================================')

    # Set Optimizer
    print('Building Optimizer...')
    param_optim = Set_Param_Optim(args, model)
    optimizer = Set_Optimizer(args, param_optim, args.lr, args.weight_decay, args.momentum)

    # Save Best Checkpoint
    Best_Accuracy, Best_Recall = 0, 0

    # Running Experiment
    print("Run Experiment...")
    writer = SummaryWriter(os.path.join(args.out, args.log))

    for epoch in range(1, args.epochs + 1):
        Train_MME(args, model, train_source_dataloader, train_target_dataloader, optimizer, epoch, writer)
        print('\nEvaluating train sets:')
        Test(args, model, train_source_dataloader, Best_Accuracy, Best_Recall, domain='Source', split='train')
        Best_Accuracy, Best_Recall = Test(args, model, train_target_dataloader, Best_Accuracy, Best_Recall,
                                          domain='Target', split='unlabeled train')
        print('\nEvaluating test sets:')
        Test(args, model, test_source_dataloader, Best_Accuracy, Best_Recall, domain='Source', split='test')
        Test(args, model, test_target_dataloader, Best_Accuracy, Best_Recall, domain='Target', split='test')

    writer.close()


if __name__ == '__main__':
    main()
