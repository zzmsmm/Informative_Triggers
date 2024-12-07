from watermarks.base import WmMethod

import os
import logging
import random
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from helpers.utils import image_char, save_triggerset, get_size, find_tolerance, get_trg_set
from helpers.loaders import get_data_transforms, get_wm_transform, get_dataset
from helpers.transforms import EmbedText

from trainer import test, train, train_on_augmented


class Ood(WmMethod):
    def __init__(self, args):
        super().__init__(args)

        self.path = os.path.join(os.getcwd(), 'data', 'trigger_set')
        os.makedirs(self.path, exist_ok=True)  # path where to save trigger set if has to be generated

        self.wm_type = args.wm_type

    def gen_watermarks(self, device):
        cwd = os.getcwd()
        train_db_path = os.path.join(cwd, 'data')
        test_db_path = os.path.join(cwd, 'data')

        if self.wm_type == 'normal':
            dataset = self.dataset
        elif self.wm_type == 'ood1':
            if self.dataset == 'cifar10': dataset = 'mnist'
            else: dataset = 'cifar10'
        elif self.wm_type == 'ood2':
            if self.dataset == 'cifar10': dataset = 'fashionmnist'
            elif self.dataset == 'mnist': dataset = 'fashionmnist'
            else: dataset = 'mnist'
        
        train_transform, test_transform = get_data_transforms(dataset)

        wm_set, test_set, _ = get_dataset(dataset, train_db_path, test_db_path, train_transform, test_transform,
                                          valid_size=None, testquot=self.test_quot, size_train=None, size_test=None)

        for i in random.sample(range(len(wm_set)), len(wm_set)):  # iterate randomly
            img, lbl = wm_set[i]
            img = img.to(device)

            trg_lbl = (lbl + 1) % self.num_classes  # set trigger labels label_watermark=lambda w, x: (x + 1) % 10
            self.trigger_set.append((img, torch.tensor(trg_lbl)))

            if len(self.trigger_set) == self.size:
                break  # break for loop when trigger set has final size

        if self.save_wm:
            save_triggerset(self.trigger_set, os.path.join(self.path, self.dataset, self.wm_type), self.runname)
            print('watermarks generation done')



    def embed(self, net, criterion, optimizer, scheduler, train_set, test_set, train_loader, test_loader, valid_loader,
              device, save_dir):

        transform = get_wm_transform(self.dataset)
        self.trigger_set = get_trg_set(os.path.join(self.path, self.dataset, self.arch, self.wm_type, self.runname), 'labels.txt', self.size,
                                                    transform)

        self.loader()

        if self.embed_type == 'pretrained':
            # load model
            logging.info("Load model: " + self.loadmodel + ".ckpt")
            net.load_state_dict(torch.load(os.path.join('checkpoint', 'clean', self.loadmodel + '.ckpt')))

        real_acc, wm_acc, val_loss, epoch = train_on_augmented(self.epochs_w_wm, device, net, optimizer, criterion,
                                                               scheduler, self.patience, train_loader, test_loader,
                                                               valid_loader, self.wm_loader, save_dir, self.save_model)

        logging.info("Done embedding.")

        return real_acc, wm_acc, val_loss, epoch