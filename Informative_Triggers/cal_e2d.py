import argparse
import traceback

from torch.backends import cudnn

import numpy as np
import models
import torch
import torchvision
import torchvision.transforms as transforms
import copy
from itertools import cycle

from helpers.utils import *
from helpers.loaders import *
from helpers.image_folder_custom_class import *
from trainer import test
from attacks.pruning import prune
import gc

# possible models to use
model_names = sorted(name for name in models.__dict__ if name.islower() and callable(models.__dict__[name]))
# print('models : ', model_names)

# set up argument parser
parser = argparse.ArgumentParser(description='Attack the model')

# model and dataset
parser.add_argument('--dataset', default='cifar10', help='the dataset to train on [cifar10]')
parser.add_argument('--method', default='noise1')
parser.add_argument('--cuda', default='cuda:2', help='set cuda (e.g. cuda:0)')

args = parser.parse_args()

epsilon_list = [0.025, 0.05, 0.075, 0.1]

try:
    device = torch.device(args.cuda) if torch.cuda.is_available() else 'cpu'

    cwd = os.getcwd()
    
    load_path = os.path.join(cwd, 'result', 'R_list', str(args.dataset), 'pruning', args.method + '.txt')

    os.makedirs(os.path.join(cwd, 'result', 'R_list', str(args.dataset), 'e2d'), exist_ok=True)
    save_path = os.path.join(cwd, 'result', 'R_list', str(args.dataset), 'e2d', args.method + '.txt')

    data = np.loadtxt(load_path)

    epsilon_float = "{:.3f}".format(0)
    delta_float = "{:.3f}".format(0)

    with open(save_path, "a") as file:
        file.write(f"{epsilon_float}\t{delta_float}\n")

    for epsilon in epsilon_list:
        for i in range(len(data) - 1):
            if data[i][1] <= epsilon and data[i+1][1] >= epsilon:
                delta = data[i][2] + (epsilon - data[i][1])*(data[i+1][2] - data[i][2])/(data[i+1][1] - data[i][1])

                epsilon_float = "{:.3f}".format(epsilon)
                delta_float = "{:.3f}".format(delta)

                with open(save_path, "a") as file:
                    file.write(f"{epsilon_float}\t{delta_float}\n")
                break

except Exception as e:
    msg = 'An error occurred during setup: ' + str(e)
    logging.error(msg)