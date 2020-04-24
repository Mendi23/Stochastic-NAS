from utils import logSetup, setModelDir, get_alphas_dir, load_alphas, data_transforms_cifar10
import os
import torchvision.datasets as dset
import torch
from snas import SNAS
import sys
import torch.backends.cudnn as cudnn
from visualization import visualize_alphas, visualize_training
import numpy as np
from trainer import Trainer
import random

DEBUG = False

TrainConfig = {
    'num_epochs': 50 if not DEBUG else 10,  # training epochs to preform
    'log_freq': 10 if not DEBUG else 100,
    'seed': None if not DEBUG else 10,  # if none, uses (psuedo-)random.
    'trainer_path': None,  # where to save log and results
    'saved_checkpoint': None,  # resume from a saved checkpoint
}

DataConfig = {
    'data_path':
    os.path.join(os.getenv('HOME'),
                 '.pytorch-datasets'),  # location of the data corpus
    'batch_size': 64,  # batch size for training
    'cutout_length': 0,  # 0 for not using cutout
}
"""
----------------------------------------------------------------------------------------------------------
if trainer is loaded for continued training, the following parameters are set from the saved checkpoint
        !! changing this values before resuming training will result in undefined behaviour !!
----------------------------------------------------------------------------------------------------------
"""

ModelConfig = {
    # see documantaion in SNAS.__init__()
    'input_channels': 3,
    'classes': 10,
    'initial_channels': 16,
    'stem_multiplier': 3,
    'multiplier': 4,
    'intermediate_nodes': 4,
    'stacked_cells': 8,
}

TrainerConfig = {
    # see documantaion in Trainer.__init__()
    # softmax temperature values - doesn't specify in the paper
    'initial_temp': 5.0,
    'temperature_decay': 0.965,
    'model_opt_lr': 0.025,
    'model_opt_momentum': 0.9,
    'model_opt_weight_decay': 3e-4,
    'sched_max_epochs': 150,
    'arch_opt_lr': 0.025,
    'arch_opt_betas': (0.5, 0.999),
    'arch_opt_weight_decay': 1e-3,
}
"""
----------------------------------------------------------------------------------------------------------
"""

if __name__ == "__main__":

    if TrainConfig['trainer_path'] is None:
        TrainConfig['trainer_path'] = setModelDir()
    logger = logSetup(TrainConfig['trainer_path'])

    # Hardware configuration
    if not torch.cuda.is_available():
        logger.info('no gpu device available')
        sys.exit(1)
    cudnn.benchmark = True
    cudnn.enabled = True

    # TODO: check why result doesn't reproduce when setting a fixed seed
    if TrainConfig['seed'] is not None:
        random.seed(TrainConfig['seed'])
        torch.manual_seed(TrainConfig['seed'])
        torch.cuda.manual_seed(TrainConfig['seed'])
        np.random.seed(TrainConfig['seed'])

    train_data = dset.CIFAR10(
        root = DataConfig['data_path'],
        train = True,
        download = True,
        transform = data_transforms_cifar10(DataConfig['cutout_length'], True),
    )

    if DEBUG:
        sampler = torch.utils.data.sampler.SubsetRandomSampler(list(
            range(256)))
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

    model = SNAS(
        C = ModelConfig['initial_channels'],
        num_classes = ModelConfig['classes'],
        layers = ModelConfig['stacked_cells'],
        steps = ModelConfig['intermediate_nodes'],
        multiplier = ModelConfig['multiplier'],
        stem_multiplier = ModelConfig['stem_multiplier'],
        input_channels = ModelConfig['input_channels'],
    )

    SnasTrainer = Trainer(
        model = model,
        logger = logger,
        model_lr = TrainerConfig['model_opt_lr'],
        model_mom = TrainerConfig['model_opt_momentum'],
        model_wd = TrainerConfig['model_opt_weight_decay'],
        arch_lr = TrainerConfig['arch_opt_lr'],
        arch_betas = TrainerConfig['arch_opt_betas'],
        arch_wd = TrainerConfig['arch_opt_weight_decay'],
        init_temperature = TrainerConfig['initial_temp'],
        temperature_decay = TrainerConfig['temperature_decay'],
        max_epochs = TrainerConfig['sched_max_epochs'],
    )

    if TrainConfig['saved_checkpoint'] is not None:
        SnasTrainer.load_checkpoint(TrainConfig['saved_checkpoint'])

    train_acc, train_loss = SnasTrainer.train(
        TrainConfig['trainer_path'],
        TrainConfig['num_epochs'],
        train_queue,
        TrainConfig['log_freq'],
    )

    # final report and visualisation
    epoch = SnasTrainer.epoch
    train_folder = TrainConfig['trainer_path']
    a_normal, a_reduce = load_alphas(get_alphas_dir(train_folder, epoch),
                                     'cpu')
    visualize_alphas(a_normal, a_reduce, ModelConfig['multiplier'],
                     ModelConfig['intermediate_nodes'], train_folder, 'final')
    visualize_training(
        train_folder, train_acc, train_loss,
        r'$\lambda_{0} = $' + repr(TrainerConfig['initial_temp']))
