import os
import torch
import torchvision
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from utils.utils import build_network, build_optimizer, build_writer, read_cfg
from dataset.DsoftDataset import DsoftDataset


cfg = read_cfg(config_path='config/config.yaml')

network = build_network(cfg=cfg)

optimizer = build_optimizer(cfg=cfg, network=network)

criterion = nn.BCELoss()

dump_input = torch.randn((1,3,224,224))

writer = build_writer(cfg=cfg)

writer.add_graph(network, input_to_model=dump_input)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(cfg['model']['image_size'])
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cfg['train']['mean'], cfg['train']['sigma'])
])

test_transform = transforms.Compose([
    transforms.Resize(cfg['model']['image_size'])
    transforms.ToTensor(),
    transforms.Normalize(cfg['train']['mean'], cfg['train']['sigma'])
])

trainset = DsoftDataset(
    csv_file=cfg['dataset']['train_set'],
    root_dir=cfg['dataset']['root'],
    transform=train_transform
)

testset = DsoftDataset(
    csv_file=cfg['dataset']['test_set'],
    root_dir=cfg['dataset']['root'],
    transform=train_transform
)

trainloader = torch.utils.data.DataLoader(
    dataset=trainset,
    batch_size=cfg['train']['batch_size'],
    shuffle=True,
    num_workers=2
)

testloader = torch.utils.data.DataLoader(
    dataset=testset,
    batch_size=cfg['test']['batch_size'],
    shuffle=False,
    num_workers=2
)

trainer = 

writer.close()