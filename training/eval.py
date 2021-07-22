from models.ResNet_feat import StochasticClassifier
from utils.Utils import *
from train_setup import *

def test_STAR(args, splits=None, n_classifiers=5):
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
            input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
            with torch.no_grad():
                feat = G(input, landmark)
                preds = F_cls.eval_n(feat, n=10)

                print(f'label: {label}')
                print(f'preds: {preds}')

    return
