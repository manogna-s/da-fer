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


class AttentionNetworkv2(nn.Module) :
    def __init__ (self, in_channels, d=128) :
        super (AttentionNetworkv2, self).__init__ ()
        self.e = nn.Sequential(nn.Linear (in_channels, in_channels), nn.ReLU(), nn.Linear (in_channels, d)) #nn.Linear (in_channels, in_channels)
        # self.e2 = nn.Sequential(nn.Linear (in_channels, in_channels), nn.ReLU(), nn.Linear (in_channels, d)) #nn.Linear (in_channels, in_channels)
        self.w = nn.Linear (d, 2, bias=False)

    def forward (self, x, return_alpha=False) :
        e = self.e(x)
        out = self.w(e)
        if return_alpha:
            return e, out
        return e

    def output_num(self):
        return 128


def evaluate_attention(args, G, attention_network, train_dataloader, test_dataloader, epoch):
    args.class_num=2
    acc, prec, recall = [AverageMeter() for i in range(2)], \
                        [AverageMeter() for i in range(2)], \
                        [AverageMeter() for i in range(2)]

    G.eval()
    attention_network.eval()
    test_dataloader = iter (test_dataloader)

    Features = []
    Labels = []

    print('Train attn acc:')
    for batch_index, (data, landmark, label) in enumerate (train_dataloader) :
            data, landmark, label = data.cuda(), landmark.cuda(), label.cuda()
            #label 2class 
            label[label==1]=0 #Set0: Surprised, Fear, Disgust, Anger
            label[label==2]=0
            label[label==3]=1
            label[label==4]=1
            label[label==5]=0
            label[label==6]=1

            # label[label==1]=1 #Set0: Surprised, Happy, Neutral
            # label[label==2]=1
            # label[label==3]=0
            # label[label==4]=1
            # label[label==5]=1
            # label[label==6]=0

            feature = G (data, landmark)
            e, pred = attention_network(feature, return_alpha=True)
            Compute_Accuracy(args, pred, label, acc, prec, recall)

            Features.append (feature.cpu().data.numpy())
            Labels.append (label.cpu().data.numpy())

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f_avg = Show_Accuracy(acc, prec, recall, args.class_num)

    acc, prec, recall = [AverageMeter() for i in range(2)], \
                        [AverageMeter() for i in range(2)], \
                        [AverageMeter() for i in range(2)]
    print('Test attn acc:')
    for batch_index, (data, landmark, label) in enumerate (test_dataloader) :
        data, landmark, label = data.cuda(), landmark.cuda(), label.cuda()
        #label 2class 
        label[label==1]=0 #Set0: Surprised, Fear, Disgust, Anger
        label[label==2]=0
        label[label==3]=1
        label[label==4]=1
        label[label==5]=0
        label[label==6]=1

        # label[label==1]=1 #Set0: Surprised, Happy, Neutral
        # label[label==2]=1
        # label[label==3]=0
        # label[label==4]=1
        # label[label==5]=1
        # label[label==6]=0

        feature = G (data, landmark)
        e, pred = attention_network(feature, return_alpha=True)
        Compute_Accuracy(args, pred, label, acc, prec, recall)

        Features.append (feature.cpu().data.numpy())
        Labels.append (label.cpu().data.numpy())

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f_avg = Show_Accuracy(acc, prec, recall, args.class_num)

    Features = np.vstack(Features)
    Labels = np.concatenate(Labels)
    viz_tsne(args, Features, Labels, epoch=f'subset_feat_{epoch}')

    args.class_num=7
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
  
