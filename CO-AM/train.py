#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import torch
import argparse
import torch.nn.functional as F
from Imagefolder_modified import Imagefolder_modified
from resnet import ResNet18_Normalized, ResNet50
from bcnn import BCNN_Normalized
from PIL import ImageFile  # Python：IOError: image file is truncated 的解决办法

ImageFile.LOAD_TRUNCATED_IMAGES = True

#from loss import loss_coteaching

import time

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"


# torch.manual_seed(0)
# torch.cuda.manual_seed(0)

class Manager(object):
    def __init__(self, options):
        """
        Prepare the network, criterion, Optimizer and data
        Arguments:
            options [dict]  Hyperparameter
            path    [dict]  path of the dataset and model
        """

        print('------------------------------------------------------------------------------')
        print('Preparing the network and data ... ')
        self._options = options
        self._path = options['path']
        os.popen('mkdir -p ' + self._path)
        self._data_base = options['data_base']
        self._class = options['n_classes']
        self._denoise = options['denoise']
        self._drop_rate = options['drop_rate']
        self._smooth = options['smooth']
        self._label_weight = options['label_weight']
        self._tk = options['tk']
        self._warmup = options['warmup']
        self._step = options['step']
        self._epoch = options['epochs']
        self._m = options['m']
        print('Basic information: ', 'data:', self._data_base, '  lr:', self._options['base_lr'], '  w_decay:',
              self._options['weight_decay'])
        print('Parameter information: ', 'denoise:', self._denoise, '  drop_rate:', self._drop_rate, '  smooth:',
              self._smooth, '  label_weight:', self._label_weight, '  tk:', self._tk, '  warmup:', self._warmup,'  m:', self._m)
        print('------------------------------------------------------------------------------')
        # Network
        print(options['net'])
        if options['net'] == 'resnet18':
         NET = ResNet18_Normalized
        elif options['net'] == 'resnet50':
         NET = ResNet50
        elif options['net'] == 'bcnn':
         NET = BCNN_Normalized
        else:
          raise AssertionError('Not implemented yet')

        if self._step == 1:
            net1 = NET(n_classes=options['n_classes'], pretrained=True)
            net2 = NET(n_classes=options['n_classes'], pretrained=True)
        elif self._step == 2:
            net1 = NET(n_classes=options['n_classes'], pretrained=False)
            net2 = NET(n_classes=options['n_classes'], pretrained=False)
        else:
            raise AssertionError('Wrong step')
        # self._net = net.cuda()
        if torch.cuda.device_count() >= 1:
            self._net1 = torch.nn.DataParallel(net1).cuda()
            self._net2 = torch.nn.DataParallel(net2).cuda()
            print('cuda device : ', torch.cuda.device_count())
        else:
            raise EnvironmentError('This is designed to run on GPU but no GPU is found')

        self._txtfile = "bcnn-web-bird.txt"

        # Criterion
        self._criterion = torch.nn.CrossEntropyLoss().cuda()
        # Optimizer
        if options['net'] == 'bcnn':
            if self._step == 1:
                params_to_optimize_1 = self._net1.module.fc.parameters()
                params_to_optimize_2 = self._net2.module.fc.parameters()
                print('step1')
            else:
                self._net1.load_state_dict(torch.load(os.path.join(self._path, 'bcnn1_step1.pth')))
                self._net2.load_state_dict(torch.load(os.path.join(self._path, 'bcnn2_step1.pth')))
                print('step2, loading model')
                params_to_optimize_1 = self._net1.parameters()
                params_to_optimize_2 = self._net2.parameters()
        else:
            params_to_optimize_1 = self._net1.parameters()
            params_to_optimize_2 = self._net2.parameters()

        self._optimizer_1 = torch.optim.SGD(params_to_optimize_1, lr=self._options['base_lr'], momentum=0.9,
                                            weight_decay=self._options['weight_decay'])
        self._optimizer_2 = torch.optim.SGD(params_to_optimize_2, lr=self._options['base_lr'], momentum=0.9,
                                            weight_decay=self._options['weight_decay'])

        if self._warmup > 0:
            warmup = lambda epoch: epoch / 5
            self._warmupscheduler_1 = torch.optim.lr_scheduler.LambdaLR(self._optimizer_1, lr_lambda=warmup)
            self._warmupscheduler_2 = torch.optim.lr_scheduler.LambdaLR(self._optimizer_2, lr_lambda=warmup)
        else:
            print('no warmup')

        self._scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer_1, T_max=self._options['epochs'])
        self._scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer_2, T_max=self._options['epochs'])
        print('lr_scheduler: CosineAnnealingLR')

        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),
            torchvision.transforms.RandomCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        # Load data
        data_dir = self._data_base
        train_data = Imagefolder_modified(os.path.join(data_dir, 'train'), transform=train_transform)
        test_data = Imagefolder_modified(os.path.join(data_dir, 'val'), transform=test_transform)
        print('number of classes in trainset is : {}'.format(len(train_data.classes)))
        print('number of classes in testset is : {}'.format(len(test_data.classes)))
        assert len(train_data.classes) == options['n_classes'] and len(test_data.classes) == options[
            'n_classes'], 'number of classes is wrong'
        self._train_loader = DataLoader(train_data, batch_size=self._options['batch_size'],
                                        shuffle=True, num_workers=4, pin_memory=True)
        self._test_loader = DataLoader(test_data, batch_size=16,
                                       shuffle=False, num_workers=4, pin_memory=True)

        self._rate_schedule = np.ones(self._options['epochs']) * self._drop_rate
        self._rate_schedule[:self._tk] = np.linspace(0, self._drop_rate, self._tk)

    def _selection(self, false_id, ids, weighted_cos_angle, labels):
        id_batch = ids.numpy().tolist()
        loss_update = [id_batch.index(x) for x in id_batch if x not in false_id]
        logits_final = weighted_cos_angle[loss_update]
        labels_final = labels[loss_update]

        if self._smooth == True:
            loss = self._smooth_label_loss(logits_final, labels_final)
        else:
            loss = self._criterion(logits_final, labels_final)
        return loss, len(logits_final)

    def _smooth_label_loss(self, logits, labels):
        N = labels.size(0)
        smoothed_labels = torch.full(size=(N, self._class),
                                     fill_value=(1 - self._label_weight) / (self._class - 1)).cuda()
        smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=self._label_weight)
        log_prob = torch.nn.functional.log_softmax(logits, dim=1)
        loss = -torch.sum(log_prob * smoothed_labels) / N
        return loss

    def train(self):
        """
        Train the network
        """
        print('Training ... ')
        best_accuracy = 0.0
        best_epoch = None
        print('Epoch\tTrain Loss\tTrain Accuracy1\tTrain Accuracy2\tTest Accuracy1\tTest Accuracy2\tEpoch Runtime')
        s = 30
        false_id1 = []
        false_id2 = []
        for t in range(self._options['epochs']):
            if self._warmup > t:
                self._warmupscheduler_1.step()
                self._warmupscheduler_2.step()
                # print('warmup learning rate',self._optimizer_1.state_dict()['param_groups'][0]['lr'])
            epoch_start = time.time()
            epoch_loss = []
            record1 = []
            record2 = []
            num_correct_1 = 0
            num_correct_2 = 0
            num_total = 0
            num_remember = self._rate_schedule[t]
            for X, y, id, _ in self._train_loader:
                # Enable training mode
                self._net1.train(True)
                # Data
                X = X.cuda()
                y = y.cuda()

                # Forward pass
                # logits1 = self._net1(X)
                # logits2 = self._net2(X)
                cos_angle1 = self._net1(X)  # score is in shape (N, 200)
                # pytorch only takes label as [0, num_classes) to calculate loss
                cos_angle1 = torch.clamp(cos_angle1, min=-1, max=1)
                weighted_cos_angle1 = s * (cos_angle1 - self._m)

                cos_angle2 = self._net2(X)  # score is in shape (N, 200)
                # pytorch only takes label as [0, num_classes) to calculate loss
                cos_angle2 = torch.clamp(cos_angle2, min=-1, max=1)
                weighted_cos_angle2 = s * (cos_angle2 - self._m)

                # if self._denoise:
                # loss_1, loss_2 = loss_coteaching(logits1, logits2, y, self._rate_schedule[t])

                if self._denoise == True:
                    if t < 2:
                        if self._smooth == False:
                            loss1 = self._criterion(weighted_cos_angle1, y)
                            loss2 = self._criterion(weighted_cos_angle2, y)
                        else:
                            # smooth label loss
                            loss1 = self._smooth_label_loss(weighted_cos_angle1, y)
                            loss2 = self._smooth_label_loss(weighted_cos_angle2, y)
                        num_train = y.size(0)
                    else:
                        # loss after sample selection
                        loss1, num_train = self._selection(false_id2, id, weighted_cos_angle1, y)
                        loss2, num_train = self._selection(false_id1, id, weighted_cos_angle2, y)
                else:
                    if self._smooth == False:
                        loss1 = self._criterion(weighted_cos_angle1, y)
                        loss2 = self._criterion(weighted_cos_angle2, y)
                    else:
                        # smooth label loss
                        loss1 = self._smooth_label_loss(weighted_cos_angle1, y)
                        loss2 = self._smooth_label_loss(weighted_cos_angle2, y)
                    num_train = y.size(0)

                epoch_loss.append(loss1.item())
                # Prediction
                _, prediction_1 = torch.max(cos_angle1.data, 1)
                _, prediction_2 = torch.max(cos_angle2.data, 1)

                # record cos_angle and image id
                for i in range(y.size(0)):
                    temp1 = []
                    temp1.append(cos_angle1[i, y[i]].clone().detach())
                    temp1.append(id[i].clone())
                    record1.append(temp1)

                    temp2 = []
                    temp2.append(cos_angle2[i, y[i]].clone().detach())
                    temp2.append(id[i].clone())
                    record2.append(temp2)

                # prediction is the index location of the maximum value found,
                num_total += y.size(0)  # y.size(0) is the batch size
                num_correct_1 += torch.sum(prediction_1 == y.data).item()
                num_correct_2 += torch.sum(prediction_2 == y.data).item()

                # Backward
                self._optimizer_1.zero_grad()
                if loss1 != 0:
                    loss1.backward()
                self._optimizer_1.step()

                self._optimizer_2.zero_grad()
                if loss2 != 0:
                    loss2.backward()
                self._optimizer_2.step()
            # Record the train accuracy of each epoch
            train_accuracy1 = 100 * num_correct_1 / num_total
            train_accuracy2 = 100 * num_correct_2 / num_total
            test_accuracy1, test_accuracy2 = self.test(self._test_loader)
            test_accuracy = max(test_accuracy1, test_accuracy2)
            if self._warmup <= t:
                self._scheduler_1.step()
                self._scheduler_2.step()

            record1.sort(key=lambda x: x[0])  # ascending order
            all_id1 = [int(x[1]) for x in record1]
            num_drop1 = int(self._rate_schedule[t] * len(all_id1))
            false_id1 = all_id1[:num_drop1]

            record2.sort(key=lambda x: x[0])  # ascending order
            all_id2 = [int(x[1]) for x in record2]
            num_drop2 = int(self._rate_schedule[t] * len(all_id2))
            false_id2 = all_id1[:num_drop2]

            epoch_end = time.time()

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                print('*', end='')
                # Save mode
                if options['net'] == 'bcnn':
                    torch.save(self._net1.state_dict(), os.path.join(self._path, 'bcnn1_step{}.pth'.format(self._step)))
                    torch.save(self._net1.state_dict(), os.path.join(self._path, 'bcnn2_step{}.pth'.format(self._step)))
                else:
                    torch.save(self._net1.state_dict(), os.path.join(self._path, options['net'] + 'co-am_1.pth'))
                    torch.save(self._net2.state_dict(), os.path.join(self._path, options['net'] + 'co-am_2.pth'))
            print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%\t\t%4.2f%%\t\t%4.2f%%\t\t%4.2f\t\t%4.2f%%' % (
            t + 1, sum(epoch_loss) / len(epoch_loss),
            train_accuracy1, train_accuracy2, test_accuracy1, test_accuracy2,
            epoch_end - epoch_start, num_remember))
            with open(self._txtfile, "a") as myfile:
                myfile.write(
                    str(int(t + 1)) + ': ' + str(sum(epoch_loss) / len(epoch_loss)) + ' '+ str(train_accuracy1) + ' ' + str(train_accuracy2) + ' ' + str(test_accuracy1) + " " + str(
                        test_accuracy2) + ' ' + str(epoch_end - epoch_start) + ' ' + str(num_remember) + "\n")

        print('-----------------------------------------------------------------')
        # print('Best at epoch %d, test accuracy %f' % (best_epoch, best_accuracy))
        print('-----------------------------------------------------------------')

    def test(self, dataloader):
        """
        Compute the test accuracy

        Argument:
            dataloader  Test dataloader
        Return:
            Test accuracy in percentage
        """
        self._net1.train(False)  # set the mode to evaluation phase
        self._net2.train(False)  # set the mode to evaluation phase
        num_correct1 = 0
        num_correct2 = 0
        num_total = 0
        with torch.no_grad():
            for X, y, _, _ in dataloader:
                # Data
                X = X.cuda()
                y = y.cuda()
                # Prediction
                score1 = self._net1(X)
                score2 = self._net2(X)
                _, prediction1 = torch.max(score1, 1)
                _, prediction2 = torch.max(score2, 1)
                num_total += y.size(0)
                num_correct1 += torch.sum(prediction1 == y.data).item()
                num_correct2 += torch.sum(prediction2 == y.data).item()
        self._net1.train(True)  # set the mode to training phase
        self._net2.train(True)  # set the mode to training phase
        return 100 * num_correct1 / num_total, 100 * num_correct2 / num_total


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
    parser.add_argument('--net', dest='net', type=str, default='resnet18',
                        help='supported options: resnet18, resnet50')
    parser.add_argument('--n_classes', dest='n_classes', type=int, default=200,
                        help='number of classes')
    parser.add_argument('--path', dest='path', type=str, default='model')
    parser.add_argument('--data_base', dest='data_base', type=str)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-2)
    parser.add_argument('--w_decay', dest='w_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', dest='epochs', type=int, default=80)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--denoise', help='Turns on denoising', default=True)
    parser.add_argument('--smooth', help='Turns on smooth label', default=True)
    parser.add_argument('--cos', action='store_true', help='Turns on cos learning rate', default=True)
    parser.add_argument('--label_weight', dest='label_weight', type=float, default=0.5)
    parser.add_argument('--drop_rate', type=float, default=0.25)
    parser.add_argument('--tk', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--step', dest='step', type=int, default=1,
                        help='Step 1 is training fc only; step 2 is training the entire network')
    parser.add_argument('--m', dest='m', type=float, default=0.4,
                        help='supported options: 0.25,0.3,0.35,0.4,0.45,0.5')

    args = parser.parse_args()

    model = args.path

    print(os.path.join(os.popen('pwd').read().strip(), model))

    if not os.path.isdir(os.path.join(os.popen('pwd').read().strip(), model)):
        print('>>>>>> Creating directory \'model\' ... ')
        os.mkdir(os.path.join(os.popen('pwd').read().strip(), model))

    path = os.path.join(os.popen('pwd').read().strip(), model)

    options = {
        'base_lr': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.w_decay,
        'path': path,
        'data_base': args.data_base,
        'net': args.net,
        'n_classes': args.n_classes,
        'denoise': args.denoise,
        'drop_rate': args.drop_rate,
        'smooth': args.smooth,
        'label_weight': args.label_weight,
        'cos': args.cos,
        'tk': args.tk,
        'warmup': args.warmup,
        'step': args.step,
        'm': args.m
    }
    manager = Manager(options)
    manager.train()