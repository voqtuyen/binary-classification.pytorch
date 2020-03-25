import os
import torch
from random import randint
from trainer.base import BaseTrainer
from utils.meters import AverageMeter
from utils.eval import add_images_tb


class DsoftTrainer(BaseTrainer):
    def __init__(self, cfg, network, optimizer, criterion, lr_scheduler, device, trainloader, testloader, writer):
        super().__init__(cfg, network, optimizer, criterion, lr_scheduler, device, trainloader, testloader, writer)

        self.network = network.to(self.device)
        self.train_loss_metric = AverageMeter()
        self.train_acc_metric = AverageMeter()

        self.test_loss_metric = AverageMeter()
        self.test_acc_metric = AverageMeter()
        self.best_val_acc = 0


    def save_model(self, epoch):
        if not os.path.exists(self.cfg['output_dir']):
            os.makedirs(self.cfg['output_dir'])

        saved_name = os.path.join(self.cfg['output_dir'], '{}_{}.pth'.format(self.cfg['model']['base'], self.cfg['dataset']['name']))

        state = {
            'epoch': epoch,
            'state_dict': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, saved_name)


    def train_one_epoch(self, epoch):
        self.network.train()
        self.train_loss_metric.reset()
        self.train_acc_metric.reset()


        for i, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.network(inputs)
            outputs = outputs.squeeze()
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            preds = (outputs >= 0).type(torch.FloatTensor)
            acc = torch.mean((preds == targets).type(torch.FloatTensor))

            self.train_loss_metric.update(loss.item())
            self.train_acc_metric.update(acc.item())

            self.writer.add_scalar('Loss/train', self.train_loss_metric.avg, epoch * len(self.trainloader) + i)
            self.writer.add_scalar('Accuracy/train', self.train_acc_metric.avg, epoch * len(self.trainloader) + i)

            print('Training epoch: {}, iteration: {}, loss: {}'.format(epoch, epoch * len(self.trainloader) + i, self.train_loss_metric.avg))


    def train(self):
        for epoch in range(self.cfg['train']['num_epochs']):
            self.train_one_epoch(epoch=epoch)
            acc = self.validate(epoch=epoch)
            if acc >= self.best_val_acc:
                self.best_val_acc = acc
                self.save_model(epoch=epoch)


    def validate(self, epoch):
        self.network.eval()
        self.test_acc_metric.reset()
        self.test_loss_metric.reset()

        seed = randint(0, len(self.testloader) - 1)

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.network(inputs)
                outputs = outputs.squeeze()
                loss = self.criterion(outputs, targets)
                preds = (outputs >= 0).type(torch.FloatTensor)
                acc = torch.mean((preds == targets).type(torch.FloatTensor))

                self.test_loss_metric.update(loss.item())
                self.test_acc_metric.update(acc.item())

                if i == seed:
                    add_images_tb(cfg=self.cfg, epoch=epoch, img_batch=inputs, preds=preds, targets=targets, writer=self.writer)
            self.writer.add_scalar('Loss/test', self.test_loss_metric.avg, epoch)
            self.writer.add_scalar('Accuracy/test', self.test_acc_metric.avg, epoch)

            return self.test_acc_metric.avg