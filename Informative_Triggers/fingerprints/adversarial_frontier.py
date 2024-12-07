from fingerprints.base import FpMethod

import os
import logging
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from helpers.utils import *
from helpers.loaders import *


class AdversarialFrontier(FpMethod):
    def __init__(self, args):
        super().__init__(args)
        self.path = os.path.join(os.getcwd(), 'data', 'trigger_set')
        os.makedirs(self.path, exist_ok=True)

    def gen_fingerprints(self, model, criterion, device):
        print('Generating fingerprints. Type = ' + 'adversarial_frontier')

        # for i in range(self.size):
        while True:
            if self.dataset == 'cifar10':
                img = torch.rand(3, 32, 32)
            else:
                img = torch.rand(1, 32, 32)
                img = img.repeat(3, 1, 1)
            
            img = img.unsqueeze(0)
            img = img.to(device)
            fp_lbl = torch.argmax(model(img), dim=1)
            fp_lbl = (fp_lbl + 1) % self.num_classes

            adv = torch.rand(1, 3, 32, 32).to(device)
            adv.requires_grad = True
            optimizer = optim.Adam([adv], lr=0.1)

            for epoch in range(100):
                optimizer.zero_grad()
                output = model(img + adv)
                pre_lbl = torch.argmax(model(img + adv), dim=1)
                if pre_lbl == fp_lbl:
                    self.fingerprint_set.append((img + adv, fp_lbl.clone().detach()))
                    break
                target = fp_lbl.reshape(-1)  # key point
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            if len(self.fingerprint_set) == self.size:
                break

        if self.save_fp:
            save_triggerset(self.fingerprint_set, os.path.join(self.path, self.dataset, 'adversarial_frontier'), self.loadmodel)
            print('fingerprints generation done')