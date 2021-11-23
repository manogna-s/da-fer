from models.AdversarialNetwork import init_weights
import torch.nn.functional as F
from torch.autograd import Variable
from train_setup import *
from models.ResNet_utils import init_weights
from utils.Loss import FocalLoss, LDAMLoss
from models.mixstyle import activate_mixstyle, deactivate_mixstyle
from train_src import test, test_target

class Classifier(nn.Module):
    def __init__(self, feature_dim = 384, num_classes=7):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(nn.Linear (feature_dim, 384), nn.ReLU())
        self.cls = nn.Linear(384, num_classes)
        
        self.cls.apply(init_weights)
        self.fc.apply(init_weights)

    def forward(self, x):
        logits = self.cls(self.fc(x))
        return logits

class AttentionNetwork (nn.Module) :
    def __init__ (self, in_channels, d=128) :
        super (AttentionNetwork, self).__init__ ()
        self.e1 = nn.Sequential(nn.Linear (in_channels, in_channels), nn.ReLU(), nn.Linear (in_channels, d)) #nn.Linear (in_channels, in_channels)
        self.e2 = nn.Sequential(nn.Linear (in_channels, in_channels), nn.ReLU(), nn.Linear (in_channels, d)) #nn.Linear (in_channels, in_channels)
        self.w = nn.Linear (d, 1, bias=False)

    def forward (self, x, return_alpha=False) :
        e1 = self.e1 (x)
        e2 = self.e2 (x)

        alpha = torch.exp(self.w(e1))/(torch.exp(self.w(e1))+torch.exp(self.w(e2)))
        e = e1 * alpha + e2 * (1-alpha)

        if return_alpha :
            return e, alpha.squeeze(), e1, e2
        return e

    def output_num(self):
        return 128

def train (args, G, F, train_dataloader, optimizer_g, optimizer_f, epoch, writer, criterion) :
    G.train ()
    F.train ()

    torch.autograd.set_detect_anomaly (True)

    total_loss = AverageMeter ()
    train_dataloader = iter (train_dataloader)

    for batch_index, (data, landmark, label) in enumerate (train_dataloader) :
        data, landmark, label = data.cuda(), landmark.cuda(), label.cuda()
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        output = G (data, landmark)
        output = F (output)
        
        loss = criterion (output, label)
        loss.backward()
        optimizer_g.step()
        optimizer_f.step()
        total_loss.update (float (loss.cpu().data.item()))
        # print(f'Ep : {epoch},  loss: {loss.data.item()}')
    print(f'Train Epoch : Total avg loss {total_loss.avg}')
    return

def train_attention(args, G, attention_network, train_dataloader, attention_net_optimizer, epoch, criterion):
    attention_network.train()

    total_loss = AverageMeter ()
    train_dataloader = iter (train_dataloader)

    for batch_index, (data, landmark, label) in enumerate (train_dataloader) :
        data, landmark, label = data.cuda(), landmark.cuda(), label.cuda()
        #label 2class 
        # label[label==1]=0 #Set0: Surprised, Fear, Disgust, Anger
        # label[label==2]=0
        # label[label==3]=1
        # label[label==4]=1
        # label[label==5]=0
        # label[label==6]=1

        label[label==1]=1 #Set0: Surprised, Happy, Neutral
        label[label==2]=1
        label[label==3]=0
        label[label==4]=1
        label[label==5]=1
        label[label==6]=0

        attention_net_optimizer.zero_grad()
        output = G (data, landmark)
        e, alpha, e1, e2 = attention_network(output, return_alpha=True)
        logits = torch.stack((nn.CosineSimilarity()(e,e1), nn.CosineSimilarity()(e,e2)),dim=1)

        # pred = torch.zeros_like(label)
        # pred[alpha<0.5]=1

        attention_loss = criterion (logits, label)
        attention_loss.backward()
        attention_net_optimizer.step()
        total_loss.update (float (attention_loss.cpu().data.item()))
    print(f'Train Epoch : Total avg loss {total_loss.avg}') 
    return

def evaluate_attention(args, G, attention_network, test_dataloader):
    args.class_num=2
    acc, prec, recall = [AverageMeter() for i in range(2)], \
                        [AverageMeter() for i in range(2)], \
                        [AverageMeter() for i in range(2)]

    G.eval()
    attention_network.eval()
    test_dataloader = iter (test_dataloader)

    for batch_index, (data, landmark, label) in enumerate (test_dataloader) :
        data, landmark, label = data.cuda(), landmark.cuda(), label.cuda()
        #label 2class 
        # label[label==1]=0 #Set0: Surprised, Fear, Disgust, Anger
        # label[label==2]=0
        # label[label==3]=1
        # label[label==4]=1
        # label[label==5]=0
        # label[label==6]=1

        label[label==1]=1 #Set0: Surprised, Happy, Neutral
        label[label==2]=1
        label[label==3]=0
        label[label==4]=1
        label[label==5]=1
        label[label==6]=0

        output = G (data, landmark)
        e, alpha, e1, e2 = attention_network(output, return_alpha=True)
        # logits = torch.stack((nn.CosineSimilarity()(e,e1), nn.CosineSimilarity()(e,e2)),dim=1)

        # pred = torch.zeros_like(label)
        # pred[alpha<0.5]=1
        pred = torch.stack((alpha, 1-alpha),dim=1)
        Compute_Accuracy(args, pred, label, acc, prec, recall)

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f_avg = Show_Accuracy(acc, prec, recall, args.class_num)
    args.class_num=7
    return

def train_final_classifier(args, G, attention_network, Cls, train_dataloader, optimizer_cls, epoch, writer, criterion):
    Cls.train ()
    torch.autograd.set_detect_anomaly (True)
    total_loss = AverageMeter ()
    train_dataloader = iter (train_dataloader)

    for batch_index, (data, landmark, label) in enumerate (train_dataloader) :
        data, landmark, label = data.cuda(), landmark.cuda(), label.cuda()
        optimizer_cls.zero_grad()
        feature = G (data, landmark)
        embedding = attention_network(feature)
        final_feature = torch.cat((feature, embedding),1)
        output= Cls(final_feature)

        loss = criterion (output, label)
        loss.backward()
        optimizer_cls.step()
        total_loss.update (float (loss.cpu().data.item()))
        # print(f'Ep : {epoch},  loss: {loss.data.item()}')
    print(f'Train Epoch : Total avg loss {total_loss.avg}')
    return

def train_attn_final_classifier(args, G, attention_network, Cls, train_dataloader, optimizer_attn, optimizer_cls, epoch, writer, criterion):
    Cls.train ()
    torch.autograd.set_detect_anomaly (True)
    total_loss = AverageMeter ()
    train_dataloader = iter (train_dataloader)

    for batch_index, (data, landmark, label) in enumerate (train_dataloader) :
        data, landmark, label = data.cuda(), landmark.cuda(), label.cuda()
        optimizer_cls.zero_grad()
        feature = G (data, landmark)
        e, alpha, e1, e2 = attention_network(feature, return_alpha=True)
        final_feature = torch.cat((feature, e),1)
        output= Cls(final_feature)

        label2=torch.zeros_like(label).cuda()
        label2[label==1]=1 #Set0: Surprised, Happy, Neutral
        label2[label==2]=1
        label2[label==3]=0
        label2[label==4]=1
        label2[label==5]=1
        label2[label==6]=0
        logits = torch.stack((nn.CosineSimilarity()(e,e1), nn.CosineSimilarity()(e,e2)),dim=1)
        attention_loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 3.0]).cuda())(logits, label2)

        loss = criterion (output, label)
        loss += attention_loss
        loss.backward()
        optimizer_cls.step()
        optimizer_attn.step()
        total_loss.update (float (loss.cpu().data.item()))
        # print(f'Ep : {epoch},  loss: {loss.data.item()}')
    print(f'Train Epoch : Total avg loss {total_loss.avg}')
    return

def test_final_classifier(args, G, attention_network, Cls, train_loader, test_loader, epoch):
    G.eval ()
    attention_network.eval ()
    Cls.eval()
    Features = []
    Labels = []
    print ('Evaluating Training Dataset')

    acc, prec, recall = [AverageMeter() for i in range(args.class_num)], \
                            [AverageMeter() for i in range(args.class_num)], \
                            [AverageMeter() for i in range(args.class_num)]

    train_loader = iter (train_loader)
    for batch_index, (data, landmark, label) in enumerate (train_loader) :
        data, landmark, label = data.cuda(), landmark.cuda(), label.cuda()

        with torch.no_grad () :
            feature = G (data, landmark)
            embedding = attention_network(feature)
            final_feature = torch.cat((feature, embedding),1)
            output = Cls(final_feature)

        
        Features.append (output.cpu().data.numpy())
        Labels.append (label.cpu().data.numpy()+14)
        Compute_Accuracy(args, output, label, acc, prec, recall)

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f_avg = Show_Accuracy(acc, prec, recall, args.class_num)

    print ('Evaluating Testing Dataset')
    acc, prec, recall = [AverageMeter() for i in range(args.class_num)], \
                        [AverageMeter() for i in range(args.class_num)], \
                        [AverageMeter() for i in range(args.class_num)]

    test_loader = iter (test_loader)
    for batch_index, (data, landmark, label) in enumerate (test_loader) :
        data, landmark, label = data.cuda(), landmark.cuda(), label.cuda()

        with torch.no_grad () :
            feature = G (data, landmark)
            embedding = attention_network(feature)
            final_feature = torch.cat((feature, embedding),1)
            output = Cls(final_feature)

        Features.append (output.cpu().data.numpy())
        Labels.append (label.cpu().data.numpy())
        Compute_Accuracy(args, output, label, acc, prec, recall)

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f_avg = Show_Accuracy(acc, prec, recall, args.class_num)

    Features = np.vstack(Features)
    Labels = np.concatenate(Labels)
    viz_tsne(args, Features, Labels, epoch=epoch)

def main ():
    torch.manual_seed (args.seed)
    print_experiment_info(args)

    if args.criterion == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.criterion == 'ldam':
        if args.source == 'RAF_balanced':
            cls_num_list= np.array([713, 262, 713, 713, 713, 682, 713])
        else: #RAF
            cls_num_list= np.array([1259, 262, 713, 4705, 1885, 682, 2465])

    print(f'Using {args.criterion} loss')

    dataloaders, G, optimizer_g, writer = train_setup (args)
    optimizer_g, lr = lr_scheduler_withoutDecay (optimizer_g, lr=args.lr)
    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=10, gamma=0.5, verbose=True)

    F = Classifier(feature_dim=G.output_num(), num_classes=args.class_num)
    F.cuda()
    optimizer_f = optim.SGD(list(F.parameters()), momentum=0.9, lr=0.001, weight_decay=0.0005)
    scheduler_f = optim.lr_scheduler.StepLR(optimizer_f, step_size=10, gamma=0.1, verbose=True)
    
    attention_network = AttentionNetwork (384)
    attention_network.cuda()
    attention_net_optimizer = torch.optim.Adam (attention_network.parameters(), 1e-5)
    attention_net_scheduler = torch.optim.lr_scheduler.StepLR (attention_net_optimizer, step_size=5, gamma=0.1, verbose=True)

    Cls = Classifier(feature_dim=G.output_num()+attention_network.output_num(), num_classes=args.class_num)
    Cls.cuda()
    optimizer_cls = optim.SGD(list(Cls.parameters()), momentum=0.9, lr=0.0001, weight_decay=0.0005)
    scheduler_cls = optim.lr_scheduler.StepLR(optimizer_cls, step_size=5, gamma=0.5, verbose=True)

    if False:
        print("Train base network Experiment...")
        for epoch in range(1, args.epochs + 1):
            if args.criterion=='ldam':
                if epoch >4:
                    per_cls_weights = [1.75, 3.0, 2.0, 1.0, 1.5, 2.0, 1.25]
                else: 
                    per_cls_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                print(f'Epoch: {epoch}, per cls weights: {per_cls_weights}')
                per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
                criterion = LDAMLoss(cls_num_list, weight=per_cls_weights)

            print(f'Epoch : {epoch}')
            train(args, G, F, dataloaders['train_source'], optimizer_g, optimizer_f, epoch, writer, criterion)
            scheduler_g.step()
            scheduler_f.step()
            print('\nEvaluation ...')
            test(args, G, F, dataloaders['train_source'], dataloaders['test_source'], epoch)
            if args.save_checkpoint:
                print(os.path.join(args.out, f'ckpts/FE.pkl'))
                torch.save(G.state_dict(), os.path.join(args.out, f'ckpts/FE.pkl'))
                torch.save(F.state_dict(), os.path.join(args.out, f'ckpts/Cls.pkl'))
    print('Training base network done')


    if False:
        g_checkpoint = torch.load(os.path.join(args.out, f'ckpts/FE.pkl'))
        G.load_state_dict (g_checkpoint)
        f_checkpoint = torch.load(os.path.join(args.out, f'ckpts/Cls.pkl'))
        F.load_state_dict (f_checkpoint)

        if False:
            print('\nEvaluating trained base network...')
            test(args, G, F, dataloaders['train_source'], dataloaders['test_source'], 20)

        for epoch in range(1,5):
            print(f'Epoch: {epoch}')
            train_attention(args, G, attention_network, dataloaders['train_source'], attention_net_optimizer, epoch, nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 3.0]).cuda()))
            print(f'Train attention acc:')
            evaluate_attention(args, G, attention_network, dataloaders['train_source'])
            print(f'Test attention acc:')
            evaluate_attention(args, G, attention_network, dataloaders['test_source'])
            attention_net_scheduler.step()
            torch.save(attention_network.state_dict(), os.path.join(args.out, f'ckpts/Attn_{epoch}.pkl'))

    if True:
        g_checkpoint = torch.load(os.path.join(args.out, f'ckpts/FE.pkl'))
        G.load_state_dict (g_checkpoint)
        f_checkpoint = torch.load(os.path.join(args.out, f'ckpts/Cls.pkl'))
        F.load_state_dict (f_checkpoint)
        attn_checkpoint = torch.load(os.path.join(args.out, f'ckpts/Attn_2.pkl'))
        attention_network.load_state_dict (attn_checkpoint)

        if False:
            print('\nEvaluating trained base network...')
            test(args, G, F, dataloaders['train_source'], dataloaders['test_source'], 20)
            evaluate_attention(args, G, attention_network, dataloaders['test_source'])

        print("Train final classifier ...")
        for epoch in range(1, 10):
            if args.criterion=='ldam':
                if True: #epoch >4:
                    per_cls_weights = [1.75, 3.0, 2.0, 1.0, 1.5, 2.0, 1.25]
                else: 
                    per_cls_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                print(f'Epoch: {epoch}, per cls weights: {per_cls_weights}')
                per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
                criterion = LDAMLoss(cls_num_list, weight=per_cls_weights)

            print(f'Epoch : {epoch}')
            # train_final_classifier(args, G, attention_network, Cls, dataloaders['train_source'], optimizer_cls, epoch, writer, criterion)
            train_attn_final_classifier(args, G, attention_network, Cls, dataloaders['train_source'], attention_net_optimizer, optimizer_cls, epoch, writer, criterion)
            attention_net_scheduler.step()

            scheduler_cls.step()
            print('\nEvaluation ...')
            test_final_classifier(args, G, attention_network, Cls, dataloaders['train_source'], dataloaders['test_source'], epoch)
            if args.save_checkpoint:
                torch.save(Cls.state_dict(), os.path.join(args.out, f'ckpts/Final_Cls.pkl'))
    


if __name__ == '__main__':
    main()