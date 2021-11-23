import tqdm
import argparse
from torch.autograd import Variable
import torch
from models.MDD import MDD
from train_setup import *
from easydict import EasyDict as edict
import yaml
import copy

def Config(filename):
    with open(filename, 'r') as f:
        parser = edict(yaml.load(f))
    for x in parser:
        print('{}: {}'.format(x, parser[x]))
    return parser

class INVScheduler(object):
    def __init__(self, gamma, decay_rate, init_lr=0.001):
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.init_lr = init_lr

    def next_optimizer(self, group_ratios, optimizer, num_iter):
        lr = self.init_lr * (1 + self.gamma * num_iter) ** (-self.decay_rate)
        # print(f'Learning rate: {lr}')
        i=0
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * group_ratios[i]
            i+=1
        return optimizer



def Test_with_tsne(args, model_instance, dataloaders, epoch, splits=None):
    if splits is None:  # evaluate on test splits by default
        splits = ['test_source', 'test_target']
    ori_train_state = model_instance.is_train
    model_instance.set_train(False)
    Features = []
    Labels = []


    if True:
        iter_dataloader = iter(dataloaders['train_source'])
        acc1, prec1, recall1 = [AverageMeter() for i in range(args.class_num)], \
                               [AverageMeter() for i in range(args.class_num)], \
                               [AverageMeter() for i in range(args.class_num)]

        for batch_index, (input, landmark, label) in enumerate(iter_dataloader):
            input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
            with torch.no_grad():
                feat, output1, softmax_outputs, _ = model_instance.c_net(input, landmark)
                
                Features.append (feat.cpu().data.numpy())
                Labels.append (label.cpu().data.numpy()+14)
            Compute_Accuracy(args, output1, label, acc1, prec1, recall1)

        print('Classifier 1')
        AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc1, prec1, recall1, args.class_num)

    Features_src_test=copy.deepcopy(Features)
    Labels_src_test=copy.deepcopy(Labels)

    Features_tar_train=copy.deepcopy(Features)
    Labels_tar_train=copy.deepcopy(Labels)

    Features_tar_test=copy.deepcopy(Features)
    Labels_tar_test=copy.deepcopy(Labels)

    for split in splits:
        print(f'\n[{split}]')

        iter_dataloader = iter(dataloaders[split])
        acc1, prec1, recall1 = [AverageMeter() for i in range(args.class_num)], \
                               [AverageMeter() for i in range(args.class_num)], \
                               [AverageMeter() for i in range(args.class_num)]

        for batch_index, (input, landmark, label) in enumerate(iter_dataloader):
            input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
            with torch.no_grad():
                feat, output1, softmax_outputs, _ = model_instance.c_net(input, landmark)
                
                if split == 'test_source':
                    Features_src_test.append (feat.cpu().data.numpy())    
                    Labels_src_test.append (label.cpu().data.numpy())
                if split == 'test_target':    
                    Features_tar_test.append (feat.cpu().data.numpy())    
                    Labels_tar_test.append (label.cpu().data.numpy()+7)
                if split == 'train_target':  
                    Features_tar_train.append (feat.cpu().data.numpy())    
                    Labels_tar_train.append (label.cpu().data.numpy()+7)

            Compute_Accuracy(args, output1, label, acc1, prec1, recall1)

        AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc1, prec1, recall1, args.class_num)

    Features_src_test = np.vstack(Features_src_test)
    Labels_src_test = np.concatenate(Labels_src_test)
    viz_tsne(args, Features_src_test, Labels_src_test, epoch=f'test_source_{epoch}')

    Features_tar_train = np.vstack(Features_tar_train)
    Labels_tar_train = np.concatenate(Labels_tar_train)
    viz_tsne(args, Features_tar_train, Labels_tar_train, epoch=f'train_target_{epoch}')

    Features_tar_test = np.vstack(Features_tar_test)
    Labels_tar_test = np.concatenate(Labels_tar_test)
    viz_tsne(args, Features_tar_test, Labels_tar_test, epoch=f'test_target_{epoch}')

    model_instance.set_train(ori_train_state)

    return



#==============eval
def evaluate(model_instance, input_loader):

    acc, prec, recall = [AverageMeter() for i in range(args.class_num)], \
                        [AverageMeter() for i in range(args.class_num)], \
                        [AverageMeter() for i in range(args.class_num)]

    ori_train_state = model_instance.is_train
    model_instance.set_train(False)
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True

    for i in range(num_iter):
        data = iter_test.next()
        inputs = data[0]
        landmarks = data[1]
        labels = data[2]
        if model_instance.use_gpu:
            inputs = Variable(inputs.cuda())
            landmarks = Variable(landmarks.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            landmarks = Variable(landmarks)
            labels = Variable(labels)
        probabilities = model_instance.predict(inputs, landmarks)

        probabilities = probabilities.data.float()
        labels = labels.data.float()
        if first_test:
            all_probs = probabilities
            all_labels = labels
            first_test = False
        else:
            all_probs = torch.cat((all_probs, probabilities), 0)
            all_labels = torch.cat((all_labels, labels), 0)
        Compute_Accuracy(args, probabilities, labels, acc, prec, recall)

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)
    print('\n')
    _, predict = torch.max(all_probs, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_labels) / float(all_labels.size()[0])

    model_instance.set_train(ori_train_state)
    return {'accuracy':accuracy}

def train(args, model_instance, train_source_loader, train_target_loader, test_source_loader, test_target_loader,
          group_ratios, max_iter, optimizer, lr_scheduler, eval_interval):
    model_instance.set_train(True)
    print("start train...")
    iter_num = 0
    epoch = 0
    # total_progress_bar = tqdm.tqdm(desc='Train iter', total=max_iter)

    num_iter = len(train_source_loader) if (len(train_source_loader) > len(train_target_loader)) else len(
        train_target_loader)
    for epoch in range(100):
        for batch_index in range(num_iter):
            try:
                inputs_source, landmark_source, labels_source = iter_source_dataloader.next()
            except:
                iter_source_dataloader = iter(train_source_loader)
                inputs_source, landmark_source, labels_source = iter_source_dataloader.next()

            try:
                inputs_target, landmark_target, labels_target = iter_target_dataloader.next()
            except:
                iter_target_dataloader = iter(train_target_loader)
                inputs_target, landmark_target, labels_target = iter_target_dataloader.next()

        # for (datas, datat) in tqdm.tqdm(
        #         zip(train_source_loader, train_target_loader),
        #         total=min(len(train_source_loader), len(train_target_loader)),
        #         desc='Train epoch = {}'.format(epoch), ncols=80, leave=False):
            # print(datas.shape, datat.shape)
            # inputs_source, landmark_source, labels_source = datas
            # inputs_target, landmark_target, labels_target = datat

            optimizer = lr_scheduler.next_optimizer(group_ratios, optimizer, epoch/5)
            optimizer.zero_grad()

            if model_instance.use_gpu:
                inputs_source, landmark_source, labels_source = inputs_source.cuda(), landmark_source.cuda(), labels_source.cuda()
                inputs_target, landmark_target, labels_target = inputs_target.cuda(), landmark_target.cuda(), labels_target.cuda()

                # inputs_source, inputs_target, labels_source = Variable(inputs_source).cuda(), Variable(
                #     inputs_target).cuda(), Variable(labels_source).cuda()
            else:
                inputs_source, inputs_target, labels_source = Variable(inputs_source), Variable(inputs_target), Variable(labels_source)

            train_batch(model_instance, inputs_source, landmark_source, labels_source, inputs_target, landmark_target, optimizer)

            # val
        print(f'\n\nEpoch: {epoch}')
        # Test_with_tsne(args, model_instance, dataloaders, epoch, splits=['test_source', 'train_target', 'test_target'])

        if True: #    if num_iter % eval_interval == 0 and iter_num != 0:
            train_source_result = evaluate(model_instance, train_source_loader)
            test_source_result = evaluate(model_instance, test_source_loader)

            train_target_result = evaluate(model_instance, train_target_loader)

            test_target_result = evaluate(model_instance, test_target_loader)
            print(f'Train source acc:{train_source_result} \n  Test source acc: {test_source_result} \n Train target acc: {train_target_result} \n Test target acc: {test_target_result}')

    print('finish train')

def train_batch(model_instance, inputs_source, landmark_source, labels_source, inputs_target, landmark_target, optimizer):
    inputs = torch.cat((inputs_source, inputs_target), dim=0)
    landmarks = torch.cat((landmark_source, landmark_target), dim=0)

    total_loss = model_instance.get_loss(inputs, landmarks, labels_source)
    total_loss.backward()
    optimizer.step()

if __name__ == '__main__':


    cfg = Config('/home/manogna/da-fer/training/exp_logs/MDD/mdd.yml')

    print_experiment_info(args)

    dataloaders, _, _, _ = train_setup(args)

    model_instance = MDD(base_net='ResNet50', width=384, use_gpu=True, class_num=args.class_num, srcweight=2)

    param_groups = model_instance.get_parameter_list()
    group_ratios = [group['lr'] for group in param_groups]


    assert cfg.optim.type == 'sgd', 'Optimizer type not supported!'

    optimizer = torch.optim.SGD(param_groups, **cfg.optim.params)

    assert cfg.lr_scheduler.type == 'inv', 'Scheduler type not supported!'
    lr_scheduler = INVScheduler(gamma=cfg.lr_scheduler.gamma,
                                decay_rate=cfg.lr_scheduler.decay_rate,
                                init_lr=0.01) #cfg.init_lr

    train(args, model_instance, dataloaders['train_source'], dataloaders['train_target'], dataloaders['test_source'], dataloaders['test_target'], group_ratios,
          max_iter=100000, optimizer=optimizer, lr_scheduler=lr_scheduler, eval_interval=1000)