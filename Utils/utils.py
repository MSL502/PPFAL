import time
import numpy as np
from pytorch_msssim import ssim
import math
import time
from Utils.option import *

class Adder(object):
    def __init__(self):
        self.count = 0
        self.num = float(0)

    def reset(self):
        self.count = 0
        self.num = float(0)

    def __call__(self, num):
        self.count += 1
        self.num += num

    def average(self):
        return self.num / self.count


class Timer(object):
    def __init__(self, option='s'):
        self.tm = 0
        self.option = option
        if option == 's':
            self.devider = 1
        elif option == 'm':
            self.devider = 60
        else:
            self.devider = 3600

    def tic(self):
        self.tm = time.time()

    def toc(self):
        return (time.time() - self.tm) / self.devider


def check_lr(optimizer):
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
    return lr

def torchSSIM(tar_img, prd_img):
    return ssim(tar_img, prd_img, data_range=1.0, size_average=True)


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_log(mode):
    log_dir = args.model_save_dir + 'Log/'
    create_dir(log_dir)
    if mode ==  'train':
        Logger = open(f'{log_dir}{args.trainset_name }_train.log', 'a+')  #TODO
    else:
        Logger = open(f'{log_dir}{args.trainset_name }_test.log', 'a+')
    Logger.write(time.strftime('\n%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    Logger.write('\n')
    return Logger

def close_log(Logger):
    Logger.write(time.strftime('\n%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    Logger.close()

def lr_schedule_cosdecay(t, T, init_lr=0.0001):
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr

def adjust_learning_rate(epoch, optim):
    if args.lr_sche:
        lr = lr_schedule_cosdecay(epoch, args.num_epochs, init_lr=args.learning_rate)
    else:
        lr = args.lr
    for param_group in optim.param_groups:
        param_group["lr"] = lr
    return lr

