import argparse
import traceback

import torch
import os
import numpy as np

import models

from helpers.loaders import get_wm_transform
from helpers.image_folder_custom_class import ImageFolderCustomClass
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='test the models')

parser.add_argument('--method', default='', choices=['', 'noise1', 'noise2', 'noise3', 'normal', 'ood1', 'ood2',
                                                     'adversarial_frontier', 'ae_frontier'])
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--cuda', default='cuda:2', help='set cuda (e.g. cuda:0)')

args = parser.parse_args()

try:
    device = torch.device(args.cuda) if torch.cuda.is_available() else 'cpu'
    cwd = os.getcwd()

    test_method = args.method

    arch_list = ['lenet5', 'vgg16', 'resnet18', 'resnet34']
    lr_list = ['0.1', '0.08', '0.06', '0.04', '0.02']
    # schedule_list = ['MultiStepLR_cifar10_0', 'MultiStepLR_cifar10_1']
    schedule_list = ['MultiStepLR_mnist_0', 'MultiStepLR_mnist_1']
    # epoch_list = ['10', '25', '35', '45', '65']
    epoch_list = ['5', '10', '15', '20', '25']
    
    os.makedirs(os.path.join(cwd, 'matrix', str(args.dataset)), exist_ok=True)
    save_path = os.path.join(cwd, 'matrix', str(args.dataset), test_method + '.txt')
    result = np.empty((0, 50))
    #!!! result = np.loadtxt(save_path)

    img_path = os.path.join(cwd, 'data', 'trigger_set', args.dataset, test_method)
    transform = transforms.Compose([
            transforms.ToTensor()
        ])
    img_set = ImageFolderCustomClass(img_path, transform)
    img_loader = torch.utils.data.DataLoader(img_set, batch_size=50, shuffle=False)

    for arch in arch_list:   # 4
        for lr in lr_list:  # 4 * 5 = 20
            for sched in schedule_list:  # 20 * 2 = 40
                for epoch in epoch_list: # 40 * 5 = 200
                    for i in ['A', 'B', 'C']:
                        # load the test model
                        model_name = f"{str(args.dataset)[0].upper()}_{str(arch)}_{str(lr)}_{str(sched).split('_')[2]}_{str(epoch)}_{str(i)}"
                        print(f'Method: {test_method}, Test model {model_name}.ckpt...')
            
                        net = models.__dict__[arch](num_classes=10)
                        net.load_state_dict(torch.load(os.path.join(cwd, 'checkpoint', str(args.dataset), str(arch), str(lr), str(sched), str(epoch),
                                                                    f'{model_name}.ckpt'), map_location=device))
                        net.to(device)

                        # test
                        for index, (inputs, targets) in enumerate(img_loader):
                            inputs = inputs.to(device)
                        
                        outputs = torch.argmax(net(inputs), dim=1).cpu().numpy()
                        # result = np.append(result, outputs)
                        result = np.vstack((result, outputs.reshape(1, -1)))
                        np.savetxt(save_path, result, fmt='%d')

    np.savetxt(save_path, result, fmt='%d')


except Exception as e:
    msg = 'An error occurred during setup: ' + str(e)
    print(msg)

    traceback.print_tb(e.__traceback__)