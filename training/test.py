from numpy.core.fromnumeric import mean
from models.ResNet_feat import StochasticClassifier
from utils.Utils import *
from train_setup import *
import time
import collections


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


def test_STAR(args, splits=None, n_classifiers=5):
    acc, prec, recall = [AverageMeter() for i in range(args.class_num)], \
                        [AverageMeter() for i in range(args.class_num)], \
                        [AverageMeter() for i in range(args.class_num)]

    if splits is None:  # evaluate on test splits by default
        splits = ['test_source', 'test_target']
    args.train_batch = 1
    args.test_batch = 1
    dataloaders, G, _, _ = train_setup(args)

    ckpt_dir = os.path.join(args.out, 'ckpts')
    F_cls = StochasticClassifier(args)
    F_cls.cuda()
    
    G_ckpt = torch.load(os.path.join(ckpt_dir, 'star_G.pkl'))
    G.load_state_dict(G_ckpt)

    F_cls_ckpt = torch.load(os.path.join(ckpt_dir, 'star_F.pkl'))
    F_cls.load_state_dict(F_cls_ckpt)

    G.eval()
    F_cls.eval()

    for split in splits:
        print(f'\n[{split}]')
        iter_dataloader = iter(dataloaders[split])

        for batch_index, (input, landmark, label) in enumerate(iter_dataloader):
            input, landmark, label = input.cuda(), landmark.cuda(), label
            with torch.no_grad():
                feat = G(input, landmark)
                preds, ent = F_cls.eval_n(feat, n=n_classifiers)
                avg_ent = mean(ent)
                votes = collections.Counter(preds)
                pred = max(votes, key=votes.get)

                print(f'Image {batch_index}: label={label}, pred={pred}, ent: {avg_ent}')

                for i in range(args.class_num):
                    TP = np.int((pred == i) * (label == i))
                    TN = np.int((pred != i) * (label != i))

                    # Compute Accuracy of All --> TP+TN / All
                    acc[i].update(np.int(pred == label))

                    # Compute Precision of Positive --> TP/(TP+FP)
                    prec[i].update(TP, np.int(pred == i))

                    # Compute Recall of Positive --> TP/(TP+FN)
                    recall[i].update(TP, np.int(label == i))

        AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)


    return

def main():
    if args.use_star:
        test_STAR(args, splits = ['test_target'], n_classifiers=10)
    return

if __name__ == '__main__':
    main()