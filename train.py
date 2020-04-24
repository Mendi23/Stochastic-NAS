import os
import torch
import torch.nn as nn
import sys
import torch.backends.cudnn as cudnn
from utils import setModelDir, logSetup, load_alphas, data_transforms_cifar10, AvgrageMeter
from visualization import visualize_alphas, visualize_training
import numpy as np
import random
from genotype import genotype
from model import NetworkCIFAR as Network
from trainer import CifarTrainer
import torchvision.datasets as dset


DEBUG = True

DataConfig = {
    'data_path': os.path.join(os.getenv('HOME'), '.pytorch-datasets'), # location of the data corpus
    'batch_size': 96,  # batch size for training
    'cutout_length': 16,  # 0 for not using cutout
}

TrainConfig = {
    'log_freq': 1 if not DEBUG else 100,
    'num_epochs': 600 if not DEBUG else 10,  # training epochs to preform
    'model_path': None,  # where to save log and results
    'seed': None if not DEBUG else 10,  # if none, uses (psuedo-)random.
    'saved_checkpoint': None,  # resume from a saved checkpoint
}


TrainerConfig = {
    'learning_rate' : 0.025, # init learning rate
    'momentum': 0.9,
    'aux_weight' : 0.4, # weight for auxiliary loss. None for not using
    'weight_decay': 3e-4,
    'drop_path_prob' : 0.2, # 0 for not using drop path probability
    'grad_clip' : 5,  # gradient clipping
    'max_epochs': 600, 
}

ModelConfig = {
    'init_channels' : 36,
    'layers' : 20 if not DEBUG else 2, # total number of layers
    'classes': 10,
    'alphas_path' : None,
    'steps' : 4,
    'multiplier' : 4,
    }

if __name__ == "__main__":

    if TrainConfig['model_path'] is None:
        TrainConfig['model_path'] = setModelDir("_eval")
    logger = logSetup(TrainConfig['model_path'])
   
    # Hardware configuration
    if not torch.cuda.is_available():
        logger.warn('no gpu device available')
        sys.exit(1)
    cudnn.benchmark = True
    cudnn.enabled = True

    # TODO: check why result doesn't reproduce when setting a fixed seed
    if TrainConfig['seed'] is not None:
        random.seed(TrainConfig['seed'])
        torch.manual_seed(TrainConfig['seed'])
        torch.cuda.manual_seed(TrainConfig['seed'])
        np.random.seed(TrainConfig['seed'])
    
    if ModelConfig['alphas_path'] is None or not os.path.exists(ModelConfig['alphas_path']):
        logger.warning('cant find alpha in the specified path')
        sys.exit(1)

    alpha_normal, alpha_reduce = load_alphas(ModelConfig['alphas_path'])
    snas_cell = genotype(alpha_normal, alpha_reduce, ModelConfig['steps'], ModelConfig['multiplier'])
    auxiliary = TrainerConfig['aux_weight'] is not None
    model = Network(ModelConfig['init_channels'], ModelConfig['classes'], ModelConfig['layers'], auxiliary, snas_cell)

    train_data = dset.CIFAR10(
        root = DataConfig['data_path'],
        train = True,
        download = True,
        transform = data_transforms_cifar10(DataConfig['cutout_length'], True),
    )

    if DEBUG:
        sampler = torch.utils.data.sampler.SubsetRandomSampler(list(range(256)))
        train_queue = torch.utils.data.DataLoader(
            train_data,
            sampler = sampler,
            batch_size = DataConfig['batch_size'],
            shuffle = False,
            pin_memory = True,
            num_workers = 16,
        )

    else:
        train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size = DataConfig['batch_size'],
        shuffle = TrainConfig['seed'] is None,
        pin_memory = True,
        num_workers = 16,
        )

    cifarTrainer = CifarTrainer(
        model = model,
        path_to_save = TrainConfig['model_path'],
        logger = logger,
        lr = TrainerConfig['learning_rate'],
        momentum = TrainerConfig['momentum'],
        weight_decay = TrainerConfig['weight_decay'],
        max_epochs = TrainerConfig['max_epochs'],
        drop_path_prob = TrainerConfig['drop_path_prob'],
        aux_weight = TrainerConfig['aux_weight'],
        grad_clip = TrainerConfig['grad_clip'],
    )

    if TrainConfig['saved_checkpoint'] is not None:
        cifarTrainer.load_checkpoint(TrainConfig['saved_checkpoint'])
    
    train_acc, train_loss = cifarTrainer.train(TrainConfig['num_epochs'], train_queue, TrainConfig['log_freq'])

    train_folder = TrainConfig['model_path']
    visualize_training(train_folder, train_acc, train_loss, "Final model training")
