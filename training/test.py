from utils.Utils import *
import time


def Test(args, model, dataloader, Best_Accuracy=None, Best_Recall=None, domain='Target', split='test'):
    """Test."""

    print(f'\n[{domain} {split} set]')
    model.eval()
    torch.autograd.set_detect_anomaly(True)

    iter_dataloader = iter(dataloader)

    # Test on Source Domain
    acc, prec, recall = [AverageMeter() for i in range(args.class_num)], \
                        [AverageMeter() for i in range(args.class_num)], \
                        [AverageMeter() for i in range(args.class_num)]
    loss, data_time, batch_time = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    for batch_index, (input, landmark, target) in enumerate(iter_dataloader):
        data_time.update(time.time() - end)

        input, landmark, target = input.cuda(), landmark.cuda(), target.cuda()

        with torch.no_grad():
            end = time.time()
            if args.use_gcn:
                feature, output, loc_output = model(input, landmark, args.useClassify, domain=domain)
            else:
                feature, output, loc_output = model(input, landmark)
            batch_time.update(time.time() - end)

        loss_ = nn.CrossEntropyLoss()(output, target)

        # Compute accuracy, precision and recall
        Compute_Accuracy(args, output, target, acc, prec, recall)

        # Log loss
        loss.update(float(loss_.cpu().data.numpy()))

        end = time.time()

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)

    #LoggerInfo = AccuracyInfo
    #LoggerInfo += '''    Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f}
    #Loss {loss:.4f}'''.format(acc_avg, prec_avg, recall_avg, f1_avg, loss=loss.avg)

    print(f'Loss: {loss.avg:.4f}')

    if Best_Recall is not None:
        # Save Checkpoints
        if recall_avg > Best_Recall:
            Best_Recall = recall_avg
            print('[Save] Best Recall: %.4f.' % Best_Recall)

            if args.save_checkpoint:
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), os.path.join(args.out, '{}_Recall.pkl'.format(args.log)))
                else:
                    torch.save(model.state_dict(), os.path.join(args.out, '{}_Recall.pkl'.format(args.log)))

        if acc_avg > Best_Accuracy:
            Best_Accuracy = acc_avg
            print('[Save] Best Accuracy: %.4f.' % Best_Accuracy)
            if args.save_checkpoint:
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), os.path.join(args.out, '{}_Accuracy.pkl'.format(args.log)))
                else:
                    torch.save(model.state_dict(), os.path.join(args.out, '{}_Accuracy.pkl'.format(args.log)))

    return Best_Accuracy, Best_Recall
