from models.ResNet_feat import ResClassifier
from utils.Utils import *
from train_setup import *
from models.ResNet_stoch_feat import IR_global_local_stoch_feat, IR_onlyResNet50_stoch
from models.ResNet_stoch_feat import *
from torch.distributions.multivariate_normal import MultivariateNormal

label2exp = {0:'Surprised', 1:'Fear', 2:'Disgust', 3:'Happy', 4:'Sad', 5:'Anger', 6:'Neutral'}
label2exp = {0:'Happy', 1:'Neutral'}

n_samples = 5
sigma_avg = 5
threshold = np.log(sigma_avg) + (1 + np.log(2 * np.pi)) / 2

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

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    Features = []
    Labels = []
    results = []
    for split in splits:
        out_img_dir=os.path.join(args.out, f'out_imgs_{split}')
        wrong_imgs=os.path.join(args.out, f'misclassified_imgs_{split}')
        os.makedirs(out_img_dir, exist_ok=True)
        for exp in label2exp.values():
            os.makedirs(os.path.join(out_img_dir, exp), exist_ok=True)
            os.makedirs(os.path.join(wrong_imgs, exp), exist_ok=True)


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


                probs= (F.softmax(output).cpu().data.numpy()*100).astype(int)
                max_prob = np.max(probs)
                pred_class = np.argmax(probs)
                
                mvn = MultivariateNormal(feature, scale_tril=torch.diag_embed(sigma))
                loss_fu = torch.mean(nn.ReLU()(threshold - mvn.entropy()/G.output_num()))
                entropy = mvn.entropy().cpu().data.numpy()[0]/G.output_num()


                pred_entropy = 0
                for i in range(n_samples):
                    feat = mvn.rsample()
                    output_sample = F1(feat)
                    probs_samples= F.softmax(output_sample)
                    pred_entropy += -torch.sum(probs_samples * torch.log(probs_samples))
                    print((probs_samples.cpu().data.numpy()*100).astype(int))
                pred_entropy/=n_samples

                pred= {'split':split, 'img':batch_index, 'label':label2exp[label.cpu().data.numpy()[0]], 'pred':label2exp[pred_class], 
                'entropy': f'{entropy:.3f}', 'prob': f'{max_prob}', 'pred entropy': f'{pred_entropy:5f}'}
                results.append(pred)

                if True:
                    img = input[0].cpu().data.numpy()
                    img = np.einsum('kij->ijk',img)
                    img = img * std + mean
                    img = np.clip(img, 0, 1) *255
                    img = img.astype(np.uint8)

                    out= f'feat_ent: {entropy:.3f} pred_ent: {pred_entropy:.3f} prob:{max_prob} \n label: {label2exp[label.cpu().data.numpy()[0]]}  pred:{label2exp[pred_class]}'
                    plt.imshow(img)
                    plt.title(out)
                    plt.savefig(os.path.join(out_img_dir,label2exp[label.cpu().data.numpy()[0]], f'{split}_{batch_index}.png'))


                print(pred, entropy, pred_entropy)
                print('\n\n')

            Compute_Accuracy(args, output, label, acc, prec, recall)

            Features.append(feature.cpu().data.numpy())
            Label = label.cpu().data.numpy()

            if Label[0]!=pred_class:
                plt.savefig(os.path.join(wrong_imgs,label2exp[label.cpu().data.numpy()[0]], f'{split}_{batch_index}.png'))


            if split == 'test_target':
                Label+=7
            elif split == 'train_source':
                Label+=14
            Labels.append(Label)

        AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)
    
        df = pd.DataFrame.from_dict(results)
        df.to_csv(os.path.join(out_img_dir,'results.csv'), index=False, header=True)

    return

def main():
    if args.use_stoch_feats:
        test_stoch_MCD(args, splits = ['test_target'])

    return

if __name__ == '__main__':
    main()