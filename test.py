import torchvision.datasets as dset
from utils import data_transforms_cifar10, AvgrageMeter, load_alphas
import torch
import os
import sys
from torch.autograd import Variable
from genotype import genotype
from model import NetworkCIFAR as Network
import torch.backends.cudnn as cudnn


DEBUG = True

TestConfig = {
    'data_path': os.path.join(os.getenv('HOME'), '.pytorch-datasets'), # location of the data corpus
    'drop_path_prob': 0.2, # 0 for not using drop path probability
    'log_freq': 1 if not DEBUG else 100,
    'batch_size': 96,
}

ModelConfig = {
    'init_channels' : 36,
    'layers' : 20 if not DEBUG else 2, # total number of layers
    'classes': 10,
    'alphas_path' : None,
    'steps' : 4,
    'multiplier' : 4,
}

def infer(model):

    test_data = dset.CIFAR10(
        root = TestConfig['data_path'],
        train = False,
        download = True,
        transform = data_transforms_cifar10(0, False),
    )

    if DEBUG:
        sampler = torch.utils.data.sampler.SubsetRandomSampler(list(range(256)))
        test_queue = torch.utils.data.DataLoader(
            test_data,
            sampler = sampler,
            batch_size = TestConfig['batch_size'],
            shuffle = False,
            pin_memory = True,
            num_workers = 16,
        )

    else:
        test_queue = torch.utils.data.DataLoader(
            test_data,
            batch_size = TestConfig['batch_size'],
            shuffle = False,
            pin_memory = True,
            num_workers = 16,
        )

    model.eval().cuda()
    acc_avg = AvgrageMeter('acc')
    for step, (X, y) in enumerate(test_queue):
        X = Variable(X, requires_grad = False).cuda()
        y = Variable(y, requires_grad = False).cuda(non_blocking = True)
        logits, _ = model(X, TestConfig['drop_path_prob'])
        pred = torch.argmax(logits, dim = 1)
        acc = torch.sum(pred == y).float() / TestConfig['batch_size']
        acc_avg.update(acc)

        if step % TestConfig['log_freq'] is 0:
            print(f"test batch {step}: {acc_avg}")
    print(f"Final test: {acc_avg}")

if __name__ == "__main__":

    if not torch.cuda.is_available():
        logger.warn('no gpu device available')
        sys.exit(1)
    cudnn.benchmark = True
    cudnn.enabled = True

    if ModelConfig['alphas_path'] is None or not os.path.exists(ModelConfig['alphas_path']):
        print('cant find model in the specified path')
        sys.exit(1)

    alpha_normal, alpha_reduce = load_alphas(ModelConfig['alphas_path'])
    snas_cell = genotype(alpha_normal, alpha_reduce, ModelConfig['steps'], ModelConfig['multiplier'])
    model = Network(ModelConfig['init_channels'], ModelConfig['classes'], ModelConfig['layers'], False, snas_cell)
    infer(model)
