import argparse
import traceback

from babel.numbers import format_decimal

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
parser.add_argument('--num_classes', default=10, type=int, help='number of classes for classification')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--method', default='noise1')
parser.add_argument('--cuda', default='cuda:2', help='set cuda (e.g. cuda:0)')

args = parser.parse_args()

try:
    device = torch.device(args.cuda) if torch.cuda.is_available() else 'cpu'

    cwd = os.getcwd()

    train_db_path = os.path.join(cwd, 'data')
    test_db_path = os.path.join(cwd, 'data')
    transform_train, transform_test = get_data_transforms(args.dataset)
    train_set, test_set, valid_set = get_dataset(args.dataset, train_db_path, test_db_path, transform_train, transform_test)       
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=True)

    img_path = os.path.join(cwd, 'data', 'trigger_set', args.dataset, args.method)
    transform = transforms.Compose([
            transforms.ToTensor()
        ])
    img_set = ImageFolderCustomClass(img_path, transform)
    img_loader = torch.utils.data.DataLoader(img_set, batch_size=50, shuffle=False)

    pruning_rates = [0.50, 0.60, 0.70, 0.80, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    arch_list = ['lenet5', 'vgg16', 'resnet18', 'resnet34']
    if args.dataset == 'cifar10':
        epoch_list = ['10', '25', '35', '45', '65']
    else:
        epoch_list = ['5', '10', '15', '20', '25']

    os.makedirs(os.path.join(cwd, 'result', 'R_list', str(args.dataset), 'pruning'), exist_ok=True)
    save_path = os.path.join(cwd, 'result', 'R_list', str(args.dataset), 'pruning', args.method + '.txt')

    for pruning_rate in pruning_rates:
        print("pruning attack start......")
        wrong_test = 0
        total_test = 0
        wrong_tri = 0
        total_tri = 0
        for arch in arch_list: 
                for epoch in epoch_list: 
                    model_name = f"{str(args.dataset)[0].upper()}_{str(arch)}_0.1_1_{str(epoch)}_A"
                    print(f'Method: {args.method}, Test model {model_name}.ckpt...')

                    net = models.__dict__[arch](num_classes=10)
                    if args.dataset == 'cifar10': sched = 'MultiStepLR_cifar10_1'
                    else: sched = 'MultiStepLR_mnist_1'
                    net.load_state_dict(torch.load(os.path.join(cwd, 'checkpoint', str(args.dataset), str(arch), '0.1', str(sched), str(epoch),
                                                                f'{model_name}.ckpt'), map_location=device))
                    net.to(device)
                    net_0 = copy.deepcopy(net)

                    prune(net, arch, pruning_rate)
                    
                    
                    for batch_idx, (inputs, targets) in enumerate(test_loader):
                        inputs, targets = inputs.to(device), targets.to(device)

                        outputs_0 = outputs = net_0(inputs)
                        _, predicted_0 = torch.max(outputs_0.data, 1)

                        outputs = net(inputs)
                        _, predicted = torch.max(outputs.data, 1)

                        total_test += targets.size(0)
                        wrong_test += (targets.size(0) - predicted_0.eq(predicted.data).cpu().sum())

                    for batch_idx, (inputs, targets) in enumerate(img_loader):
                        inputs, targets = inputs.to(device), targets.to(device)

                        outputs_0 = outputs = net_0(inputs)
                        _, predicted_0 = torch.max(outputs_0.data, 1)

                        outputs = net(inputs)
                        _, predicted = torch.max(outputs.data, 1)

                        total_tri += targets.size(0)
                        wrong_tri += (targets.size(0) - predicted_0.eq(predicted.data).cpu().sum())
        
        wrong_per_test = 1.0 * wrong_test / total_test
        wrong_per_tri = 1.0 * wrong_tri / total_tri

        wrong_per_test = "{:.3f}".format(wrong_per_test)
        wrong_per_tri = "{:.3f}".format(wrong_per_tri)
        pruning_rate_float = "{:.2f}".format(pruning_rate)

        with open(save_path, "a") as file:
            file.write(f"{pruning_rate_float}\t{wrong_per_test}\t{wrong_per_tri}\n")


    
except Exception as e:
    msg = 'An error occurred during setup: ' + str(e)
    logging.error(msg)