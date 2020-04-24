import sys
import os
import torch
import numpy as np
from termcolor import colored
import logging
from time import strftime
from torchvision import transforms

class AvgrageMeter(object):
    def __init__(self, name = ''):
        self.reset()
        self._name = name

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n = 1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def __str__(self):
        return "%s: %.5f" % (self._name, self.avg)

    def __repr__(self):
        return self.__str__()

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def weights_init(m, deepth = 0, max_depth = 2):
    if deepth > max_depth:
        return
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, torch.nn.BatchNorm2d):
        return
    elif isinstance(m, torch.nn.ReLU):
        return
    elif isinstance(m, torch.nn.Module):
        deepth += 1
        for m_ in m.modules():
            weights_init(m_, deepth)
    else:
        raise ValueError("%s is unk" % m.__class__.__name__)

def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = torch.autograd.Variable(
            torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x

class _MyFormatter(logging.Formatter):
    def format(self, record):
        date = colored('[%(asctime)s]', 'green')
        line = colored('[%(filename)s:%(lineno)d]', 'cyan')
        msg = '%(message)s'
        if record.levelno == logging.WARNING:
            fmt = line + ' ' + colored('WRN', 'red', attrs = ['blink'
                                                              ]) + ' ' + msg
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            fmt = date + line + ' ' + colored(
                'ERR', 'red', attrs = ['blink', 'underline']) + ' ' + msg
        else:
            fmt = date + ' ' + msg
        if hasattr(self, '_style'):
            # Python3 compatibility
            self._style._fmt = fmt
        self._fmt = fmt
        return super(_MyFormatter, self).format(record)

def logSetup(log_dir):
    log_format = '%m-%d %H:%M:%S'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_MyFormatter(log_format))
    logger.addHandler(handler)

    path = os.path.join(log_dir, 'log.log')
    handler = logging.FileHandler(filename = path,
                                  encoding = 'utf-8',
                                  mode = 'a')
    handler.setFormatter(_MyFormatter(log_format))
    logger.addHandler(handler)

    return logger

def setModelDir(model_name = ""):
    name = strftime('%Y-%m-%d_%H-%M-%S') + model_name
    model_dir = os.path.join('Models', name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir

def get_alphas_dir(model_dir, epoch):
    alphas_dir = os.path.join(model_dir, 'Alpha', f"epoch_{epoch}")
    if not os.path.exists(alphas_dir):
        os.makedirs(alphas_dir)
    return alphas_dir

def get_weights_path(model_dir, epoch):
    weights_dir = os.path.join(model_dir, 'Weights')
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    return os.path.join(weights_dir, f"epoch_{epoch}.pt")

def save_alphas(alphas_dir, alpha_normal, alpha_reduce):
    torch.save(alpha_normal, os.path.join(alphas_dir, 'normal.pt'))
    torch.save(alpha_reduce, os.path.join(alphas_dir, 'reduce.pt'))

def load_alphas(alphas_dir, device = None):
    return torch.load(os.path.join(alphas_dir, 'normal.pt'), map_location = device), torch.load(os.path.join(alphas_dir, 'reduce.pt'), map_location = device)

def save_model(path, model):
    torch.save(model.state_dict(), path)

def data_transforms_cifar10(cutout_length = 0, augment = True):
    normalize = transforms.Normalize(
        mean = [0.49139968, 0.48215827, 0.44653124],
        std = [0.24703233, 0.24348505, 0.26158768],
    )

    transform_list = []

    if augment:
        transform_list += [
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(),
        ]

    transform_list += [
        transforms.ToTensor(),
        normalize,
    ]
    transform = transforms.Compose(transform_list)

    if cutout_length is not 0:
        transform.transforms.append(Cutout(cutout_length))

    return transform
