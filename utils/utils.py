import torch
from torch import optim
import yaml
from models.mobilenet_v2 import mobilenet
from torch.utils.tensorboard import SummaryWriter


def read_cfg(config_path):
    """ Read config from yaml file
    Args:
        config_path (str): path to yaml file
    Return:
        data (dict): dict of configurations
    """
    with open(config_path, 'r') as rf:
        data = yaml.safe_load(rf)
        return data


def build_optimizer(cfg, network):
    """ Build optimizer based on config
    Args:
        - cfg (dict): dict of configurations
    Return:
        - optimizer
    """
    if cfg['train']['optimizer'] == 'adam':
        return optim.Adam(network.parameters(), lr=cfg['train']['lr'])
    else:
        raise NotImplementedError


def build_network(cfg):
    """ Build network based on config
    Args:
        - cfg (dict): dict of configurations
    Returns:
        - network (nn.Module)
    """
    if cfg['model']['base'] == 'mobilenet_v2':
        network = mobilenet(cfg=cfg)
        return network
    else:
        raise NotImplementedError

    
def build_writer(cfg):
    """ Build tensorboard writer 
    Args:
        - cfg (dict): dict of configuration
    Returns:
        - writer
    """
    writer = SummaryWriter(log_dir=cfg['log_dir'])
    return writer


def get_device(cfg):
    """ Get device based on config
    Args:
        - cfg (dict): dict of configuration
    Returns:
        - device
    """
    device = None
    if cfg['device'] == 'cpu':
        device = torch.device('cpu')
    elif cfg['device'] == 'cuda':
        device = torch.device('cuda')
    else:
        raise NotImplementedError
    return device