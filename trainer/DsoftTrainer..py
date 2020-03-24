from trainer.base import BaseTrainer

class DsoftTrainer(BaseTrainer):
    def __init__(self, cfg, network, optimizer, criterion, lr_scheduler, device, trainloader, testloader, writer):
        super().__init__(cfg, network, optimizer, criterion, lr_scheduler, device, trainloader, testloader, writer)

        self.network = network.to(self.device)


    def train_one_epoch(self):
        self.network.train()



    def train(self):
        for epoch in range(self.cfg['train']['num_epochs']):
            self.train_one_epoch()
            self.validate()


    def validate(self):
        self.network.eval()
        return