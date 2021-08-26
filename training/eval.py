from models.ResNet_feat import ResClassifier
from utils.Utils import *
from train_setup import *
from models.ResNet_stoch_feat import IR_global_local_stoch_feat, IR_onlyResNet50_stoch
from models.ResNet_stoch_feat import *

label2exp = {0:'Surprised', 1:'Fear', 2:'Disgust', 3:'Happy', 4:'Sad', 5:'Anger', 6:'Neutral'}

def test_stoch_MCD(args, splits=None):
    if splits is None:  # evaluate on test splits by default
        splits = ['test_source', 'test_target']
    args.train_batch = 1
    args.test_batch = 1

    # Build Dataloader
    print("Building Train and Test Dataloader...")
    dataloaders = {'train_source': BuildDataloader(args, split='train', domain='source', max_samples=args.source_labeled),
                   'train_target': BuildDataloader(args, split='train', domain='target', max_samples=args.target_unlabeled),
                   'test_source': BuildDataloader(args, split='test', domain='source'),
                   'test_target': BuildDataloader(args, split='test', domain='target')}
    print('Done!')

    G = IR_global_local_stoch_feat(50,feature_dim=384) 
    print(G)
    G_ckpt = torch.load(os.path.join(args.out,'ckpts', 'Stoch_MCD_G.pkl'))
    G.load_state_dict(G_ckpt)
    G.cuda()

    F1 = Stochastic_Features_cls(args, input_dim=G.output_num())
    F1_ckpt = torch.load(os.path.join(args.out,'ckpts', 'Stoch_MCD_F2.pkl'))
    F1.load_state_dict(F1_ckpt)
    F1.cuda()

    G.eval()
    F1.eval()

    # mean=[0.485, 0.456, 0.406]
    # std=[0.229, 0.224, 0.225]
    # out_img_dir=os.path.join(args.out, 'out_imgs')
    # os.makedirs(out_img_dir, exist_ok=True)

    Features = []
    Labels = []
    results = []
    for split in splits:
        print(f'\n[{split}]')
        iter_dataloader = iter(dataloaders[split])
        acc, prec, recall = [AverageMeter() for i in range(args.class_num)], \
                            [AverageMeter() for i in range(args.class_num)], \
                            [AverageMeter() for i in range(args.class_num)]
        for batch_index, (input, landmark, label) in enumerate(iter_dataloader):
            input, landmark, label = input.cuda(), landmark.cuda(), label

            with torch.no_grad():
                feature, sigma = G(input, landmark)
                output = F1(feature)


                # entropy = torch.mean(torch.log(sigma)).cpu().data.numpy()
                # probs= F.softmax(output).cpu().data.numpy()
                # max_prob = np.max(probs)
                # pred = np.argmax(probs)
                # out= f'feat_ent: {entropy:.1f} prob:{max_prob:.2f} \n label: {label2exp[label.cpu().data.numpy()[0]]}  pred:{label2exp[pred]}'
                # print(entropy, max_prob)

                # img = input[0].cpu().data.numpy()
                # img = np.einsum('kij->ijk',img)
                # img = img * std + mean
                # img = np.clip(img, 0, 1) *255
                # img = img.astype(np.uint8)

                # plt.imshow(img)
                # plt.title(out)
                # plt.savefig(os.path.join(out_img_dir,f'{split}_{batch_index}.png'))

                # pred= {'split':split, 'img':batch_index, 'label':label2exp[label.cpu().data.numpy()[0]], 'pred':label2exp[pred], 'feat_ent':f'{entropy:.2f}', 'prob': f'{max_prob:.2f}'}
                # results.append(pred)
            Compute_Accuracy(args, output, label, acc, prec, recall)

            Features.append(feature.cpu().data.numpy())
            Label = label.cpu().data.numpy()
            if split == 'test_target':
                Label+=7
            elif split == 'train_source':
                Label+=14
            Labels.append(Label)

        AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)
    
    # df = pd.DataFrame.from_dict(results)
    # df.to_csv(os.path.join(out_img_dir,'results.csv'), index=False, header=True)

    if args.show_feat:
        Features = np.vstack(Features)
        Labels = np.concatenate(Labels)
        viz_tsne(args, Features, Labels)
    return

def main():
    if args.use_stoch_feats:
        test_stoch_MCD(args, splits = ['test_target','train_source', 'test_source'])

    return

if __name__ == '__main__':
    main()