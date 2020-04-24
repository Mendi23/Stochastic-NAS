import torch.nn as nn
from utils import AvgrageMeter, weights_init, save_alphas, save_model, get_weights_path, get_alphas_dir
import torch
import time
from torch.autograd import Variable
import os


class BaseTrainer(object):
    def __init__(self, model, logger, path_to_save):

        # set training parameters
        self.model_dir = path_to_save
        self.logger = logger
        self.epoch = 0
        self.train_acc = []
        self.train_loss = []

        model.apply(weights_init)
        model.train().cuda()
        self._model = nn.DataParallel(model)

    def save_checkpoint(self, filepath):
        torch.save(self._get_checkpoint(), filepath)
        self.logger.info(
            f"saved checkpoint in epoch {self.epoch} to {filepath} ===>")

    def load_checkpoint(self, filepath):
        if os.path.isfile(filepath):
            self._initialize_checkpoint(torch.load(filepath))
            self.logger.info(f"<=== loaded checkpoint from {self.epoch}")
        else:
            self.logger.warn(
                f"{filepath} doesn't exist. can't load checkpoint")

    def _get_checkpoint(self):
        raise NotImplementedError()

    def _initialize_checkpoint(self, checkpoint):
        raise NotImplementedError()

    def _post_epoch(self):
        raise NotImplementedError()

    def _step(self, X, y):
        raise NotImplementedError()

    def train(self, epochs, train_dl, log_freq = 1):

        loss_avg = AvgrageMeter('loss')
        acc_avg = AvgrageMeter('acc')
        epoch_loss_avg = AvgrageMeter('epoch_loss')
        epoch_acc_avg = AvgrageMeter('epoch_acc')

        last_epoch = self.epoch + epochs
        self.logger.info(f"begin training for {epochs} epochs")
        while self.epoch < last_epoch:
            self.epoch += 1
            batch_tic = time.time()
            epoch_tic = time.time()
            self.logger.info(
                f"Start train for epoch {self.epoch}/{last_epoch}")
            for step, (X, y) in enumerate(train_dl, 1):

                # preform training step
                X = Variable(X, requires_grad = True).cuda()
                y = Variable(y,
                             requires_grad = False).cuda(non_blocking = True)
                pred, loss = self._step(X, y)

                # update status
                batch_size = y.size()[0]
                acc = torch.sum(pred == y).float() / batch_size
                loss_avg.update(loss)
                acc_avg.update(acc)
                epoch_loss_avg.update(loss)
                epoch_acc_avg.update(acc)

                # report status
                if step % log_freq is 0:
                    speed = 1.0 * (batch_size * log_freq) / (time.time() -
                                                             batch_tic)
                    self.logger.info(
                        "Epoch[%d]/[%d] Batch[%d] Speed: %.6f samples/sec %s %s"
                        % (self.epoch, last_epoch, step, speed, loss_avg,
                           acc_avg))
                    map(lambda avg: avg.reset(), [loss_avg, acc_avg])
                    batch_tic = time.time()

            self.logger.info("Epoch[%d]/[%d] Time: %.3f sec %s %s" %
                             (self.epoch, last_epoch, time.time() - epoch_tic,
                              epoch_loss_avg, epoch_acc_avg))
            self.train_acc.append(epoch_acc_avg.avg)
            self.train_loss.append(epoch_loss_avg.avg)
            map(lambda avg: avg.reset(), [epoch_loss_avg, epoch_acc_avg])
            self._post_epoch()
        checkpoint_path = os.path.join(self.model_dir,
                                       f'checkpoint_{self.epoch}.tar')
        self.save_checkpoint(checkpoint_path)
        return self.train_acc, self.train_loss

class SnasTrainer(BaseTrainer):
    def __init__(
            self,
            model,
            model_dir,
            logger,
            model_lr,
            model_mom,
            model_wd,
            arch_lr,
            arch_betas,
            arch_wd,
            init_temperature,
            temperature_decay,
            max_epochs,
    ):
        """
        :param init_temperature: initial softmax temperature
        :param temperature_decay: annealation rate of softmax temperature
        :param max_epochs: maximum number of epochs to allow for Cosine-Annealing scheduler
        """
        super().__init__(model, logger, model_dir)
        # set training parameters
        self.temp = init_temperature
        self.t_decay = temperature_decay
        self.max_epoch = max_epochs
        
        mod_params = model.model_parameters()
        self.alpha_params = model.arch_parameters()

        self.criterion = nn.CrossEntropyLoss().cuda()
        
        self.model_optimizer = torch.optim.SGD(mod_params,
                                               lr = model_lr,
                                               momentum = model_mom,
                                               weight_decay = model_wd)

        self.arch_optimizer = torch.optim.Adam(self.alpha_params,
                                               lr = arch_lr,
                                               betas = arch_betas,
                                               weight_decay = arch_wd)

        self.model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.model_optimizer, float(max_epochs))


    def _get_checkpoint(self):
        return {
            'temp': self.temp,
            't_decay': self.t_decay,
            'max_epoch': self.max_epoch,
            'epoch': self.epoch,
            '_model': self._model.state_dict(),
            'train_acc': self.train_acc,
            'train_loss': self.train_loss,
            'model_optimizer': self.model_optimizer.state_dict(),
            'arch_optimizer': self.arch_optimizer.state_dict(),
            'model_scheduler': self.model_scheduler.state_dict(),
        }

    def _initialize_checkpoint(self, checkpoint):
        self.temp = checkpoint['temp']
        self.t_decay = checkpoint['t_decay']
        self.max_epoch = checkpoint['max_epoch']
        self.epoch = checkpoint['epoch']
        self.train_acc = checkpoint['train_acc']
        self.train_loss = checkpoint['train_loss']
        self._model.load_state_dict(checkpoint['_model'])
        self.model_optimizer.load_state_dict(checkpoint['model_optimizer'])
        self.arch_optimizer.load_state_dict(checkpoint['arch_optimizer'])
        self.model_scheduler.load_state_dict(checkpoint['model_scheduler'])

    def _step(self, X, y):
        self.model_optimizer.zero_grad()
        self.arch_optimizer.zero_grad()
        logits = self._model(X, self.temp)
        loss = self.criterion(logits, y)
        loss.backward()
        self.model_optimizer.step()
        self.arch_optimizer.step()
        return torch.argmax(logits, dim = 1), loss

    def _post_epoch():
        save_alphas(get_alphas_dir(self.model_dir, self.epoch), *self.alpha_params)
        self.temp *= self.t_decay
        self.model_scheduler.step()
        self.logger.info(f"lr is now: {self.model_scheduler.get_lr()[0]}")

        # check if true, uf so - turn of trainng.
        if self.epoch is self.max_epoch:
        self.logger.warn(f"You reached the max number of training epochs, which is {self.max_epoch}.")
                        
class CifarTrainer(BaseTrainer):
    def __init__(
            self,
            model,
            path_to_save,
            logger,
            lr,
            momentum,
            weight_decay,
            max_epochs,
            drop_path_prob,
            aux_weight,
            grad_clip,
    ):
        super().__init__(model, logger, path_to_save)

        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr,
            momentum = momentum,
            weight_decay = weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, max_epochs)
        self.max_epochs = max_epochs
        self.drop_path_prob = drop_path_prob
        self.aux_weight = aux_weight
        self.grad_clip = grad_clip

    def _step(self, X, y):
        self.optimizer.zero_grad()
        drop_path = self.drop_path_prob * self.epoch / self.max_epochs
        logits, logits_aux = self._model(X, drop_path)
        loss = self.criterion(logits, y)

        if self.aux_weight is not None:
            loss_aux = self.criterion(logits_aux, y)
            loss += self.aux_weight * loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(self._model.parameters(), self.grad_clip) # check if here. preform to all?
        self.optimizer.step()
        return torch.argmax(logits, dim = 1), loss

    def _get_checkpoint(self):
        return {
            'max_epochs': self.max_epochs,
            'epoch': self.epoch,
            '_model': self._model.state_dict(),
            'train_acc': self.train_acc,
            'train_loss': self.train_loss,
            'drop_path_prob': self.drop_path_prob,
            'aux_weight': self.aux_weight,
            'grad_clip': self.grad_clip,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }

    def _initialize_checkpoint(self, checkpoint):
        self.max_epochs = checkpoint['max_epochs']
        self.epoch = checkpoint['epoch']
        self.train_acc = checkpoint['train_acc']
        self.train_loss = checkpoint['train_loss']
        self.drop_path_prob = checkpoint['drop_path_prob'],
        self.aux_weight = ['aux_weight'],
        self.grad_clip = ['grad_clip'],
        self._model.load_state_dict(checkpoint['_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

    def _post_epoch(self):
        self.scheduler.step()
        save_model(get_weights_path(self.model_dir, self.epoch), self._model)
