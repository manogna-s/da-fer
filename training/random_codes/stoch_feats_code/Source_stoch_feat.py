import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal
from models.ResNet_stoch_feat import *
from train_setup import *

criterion = nn.CrossEntropyLoss()
weighted_criterion = nn.CrossEntropyLoss(reduction='none')

eta = 1.0
num_k = 4
n_samples = 5
sigma_thresh = 0.3

sigma_avg = 5
threshold = np.log(sigma_avg) + (1 + np.log(2 * np.pi)) / 2

def get_fu_sample_loss(G, F1, data, landmark, target):
    sample_loss = 0
    feat, sigma = G(data, landmark)
    # print(sigma, threshold)
    mvn = MultivariateNormal(feat, scale_tril=torch.diag_embed(sigma))
    loss_fu = torch.mean(nn.ReLU()(threshold - mvn.entropy()/G.output_num()))
    for i in range(n_samples):
        output = F1(mvn.rsample())
        sample_loss += criterion(output, target)
    sample_loss = sample_loss/n_samples
    return sample_loss, loss_fu

def get_baseline_sample_loss(G, F1, data, landmark, target):
    sample_loss = 0
    loss_fu = 0
    feat, sigma = G(data, landmark)
    for i in range(n_samples):
        output = F1(feat + sigma)
        sample_loss += criterion(output, target)
    sample_loss = sample_loss/n_samples
    return sample_loss, loss_fu

ent_threshold = np.log(7)

def get_ent_sample_loss(G, F1, data, landmark, target):
    sample_loss = 0
    feat, sigma = G(data, landmark)
    mvn = MultivariateNormal(feat, scale_tril=torch.diag_embed(sigma))

    pred_entropy=0
    for i in range(n_samples):
        feat = mvn.rsample()
        output1 = F1(feat)
        sample_loss += criterion(output1, target)

        probs_1= F.softmax(output1)
        pred_entropy += -torch.sum(probs_1 * torch.log(probs_1), 1)
    
    pred_entropy/=n_samples
    weights = nn.ReLU()(ent_threshold - pred_entropy)
    weights = weights/torch.sum(weights)
    weights = weights.detach()
    # print(pred_entropy)
    # print(weights)
    sample_loss = sample_loss/n_samples
    return sample_loss, weights

def get_ent_weighted_sample_loss(G, F1, data, landmark, target):
    sample_loss = 0
    feat, sigma = G(data, landmark)
    mvn = MultivariateNormal(feat, scale_tril=torch.diag_embed(sigma))

    pred_entropy=0
    for i in range(n_samples):
        feat = mvn.rsample()
        output1 = F1(feat)
        sample_loss += weighted_criterion(output1, target)
        probs_1= F.softmax(output1)
        pred_entropy += -torch.sum(probs_1 * torch.log(probs_1), 1)
    
    pred_entropy/=n_samples
    weights = nn.ReLU()(ent_threshold - pred_entropy)
    weights = weights/torch.sum(weights)
    weights = weights.detach()
    # print(pred_entropy)
    # print(weights)
    sample_loss = (sample_loss * weights).sum()
    sample_loss = sample_loss/n_samples
    return sample_loss, weights

def Train_Stoch_Feat_Source(args, G, F1, train_source_dataloader, optimizer_g, optimizer_f, epoch,
              writer):
    """Train."""
    G.train()
    F1.train()
    torch.autograd.set_detect_anomaly(True)
    batch_size = args.train_batch

    m_total_loss = AverageMeter()

    # Get Source/Target Dataloader iterator
    iter_source_dataloader = iter(train_source_dataloader)
    num_iter = len(train_source_dataloader)

    for batch_index in range(num_iter):
        try:
            data_source, landmark_source, label_source = iter_source_dataloader.next()
        except:
            iter_source_dataloader = iter(train_source_dataloader)
            data_source, landmark_source, label_source = iter_source_dataloader.next()

        data_source, landmark_source, label_source = data_source.cuda(), landmark_source.cuda(), label_source.cuda()
        
        # Forward Propagation

        feature, _ = G(data_source, landmark_source)
        output = F1(feature)

        total_loss = 0
        sample_loss = 0
        loss_fu = 0
        # if epoch>10:
        #     sample_loss, loss_fu = get_fu_sample_loss(G, F1, data_source, landmark_source, label_source)

        if epoch>0 and epoch<=5:
            sample_loss, weights = get_ent_sample_loss(G, F1, data_source, landmark_source, label_source)
            clf_loss = criterion(output, label_source)

        elif epoch>5:
            sample_loss, weights = get_ent_weighted_sample_loss(G, F1, data_source, landmark_source, label_source) 
            clf_loss = (weighted_criterion(output, label_source) * weights).sum()
        else: 
            clf_loss = criterion(output, label_source)


        total_loss = clf_loss + 0.5 * sample_loss + 0.005 * loss_fu
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        total_loss.backward()
        optimizer_g.step()
        optimizer_f.step()

        print(f'Train Ep: {epoch} {batch_index}/{num_iter} Total Loss: {total_loss:.3f}, CLF loss: {clf_loss:.5f} ,Sample loss: {0.5 * sample_loss :.10f} ,Feature uncertainty loss: {0.001 * loss_fu}')

        # Log loss
        m_total_loss.update(float(total_loss.cpu().data.item()))

    return


def Test_Stoch_Feat_Source(args, G, F1, dataloaders, splits=None):
    if splits is None:  # evaluate on test splits by default
        splits = ['test_source', 'test_target']
    G.eval()
    F1.eval()
    for split in splits:
        print(f'\n[{split}]')
        iter_dataloader = iter(dataloaders[split])
        acc1, prec1, recall1 = [AverageMeter() for i in range(args.class_num)], \
                               [AverageMeter() for i in range(args.class_num)], \
                               [AverageMeter() for i in range(args.class_num)]
        for batch_index, (input, landmark, label) in enumerate(iter_dataloader):
            input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
            with torch.no_grad():
                feat, _ = G(input, landmark)
                output1 = F1(feat)
                
            Compute_Accuracy(args, output1, label, acc1, prec1, recall1)

        print('Classifier 1')
        AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc1, prec1, recall1, args.class_num)
    return


def main():
    """Main."""
    torch.manual_seed(args.seed)

    # Experiment Information
    print_experiment_info(args)

    dataloaders, G, optimizer_g, writer = train_setup(args)
    optimizer_g, lr = lr_scheduler_withoutDecay(optimizer_g, lr=args.lr)
    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=15, gamma=0.1, verbose=True)
    print(G)
    F1 = Stochastic_Features_cls(args, input_dim=G.output_num())
    F1.cuda()
    print(F1)
    optimizer_f = optim.SGD(list(F1.parameters()), momentum=0.9, lr=0.001, weight_decay=0.0005)
    scheduler_f = optim.lr_scheduler.StepLR(optimizer_f, step_size=15, gamma=0.1, verbose=True)

    # G_ckpt= os.path.join(args.out, f'ckpts/Stoch_source_G_10.pkl')
    # if os.path.exists(G_ckpt):
    #     checkpoint = torch.load (G_ckpt, map_location='cuda')
    #     G.load_state_dict (checkpoint, strict=False)

    # F1_ckpt= os.path.join(args.out, f'ckpts/Stoch_source_F_10.pkl')
    # if os.path.exists(F1_ckpt):
    #     checkpoint = torch.load (F1_ckpt, map_location='cuda')
    #     F1.load_state_dict (checkpoint, strict=False)

    # Running Experiment
    print("Run Experiment...")
    for epoch in range(1, 40+1):
        print(f'Epoch : {epoch}')
        Train_Stoch_Feat_Source(args, G, F1, dataloaders['train_source'], optimizer_g, optimizer_f,
                  epoch, writer)
        scheduler_g.step()
        scheduler_f.step()
        print('\nEvaluation ...')
        Test_Stoch_Feat_Source(args, G, F1, dataloaders, splits=['train_source', 'test_source'])
        if args.save_checkpoint:
            torch.save(G.state_dict(), os.path.join(args.out, f'ckpts/Stoch_source_G.pkl'))
            torch.save(F1.state_dict(), os.path.join(args.out, f'ckpts/Stoch_source_F.pkl'))
            if epoch%5==0:
                torch.save(G.state_dict(), os.path.join(args.out, f'ckpts/Stoch_source_G_{epoch}.pkl'))
                torch.save(F1.state_dict(), os.path.join(args.out, f'ckpts/Stoch_source_F_{epoch}.pkl'))

    writer.close()


if __name__ == '__main__':
    main()
