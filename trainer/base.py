class BaseTrainer():
    def __init__(self, cfg, network, optimizer, criterion, lr_schedule, device, trainloader, testloader, writer):
        self.cfg = cfg
        self.network = network
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_schedule = lr_schedule
        self.device = device
        self.trainloader = trainloader
        self.testloader = testloader
        self.writer = writer

    
    def load_model(self):
        raise NotImplementedError


    def save_model(self):
        raise NotImplementedError


    def train_one_epoch(self):
        raise NotImplementedError


    def train(self):
        raise NotImplementedError


    def validate(self):
        return NotImplementedError
