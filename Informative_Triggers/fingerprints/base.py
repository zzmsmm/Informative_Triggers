'''Provides base class for all fingerprinting methods'''

from abc import ABC, abstractmethod
import torch
import logging
import os

from helpers.utils import find_tolerance

class FpMethod(ABC):

    def __init__(self, args):
        self.num_classes = args.num_classes  # e.g. 10
        self.dataset = args.dataset  # e.g. 'cifar10'
        self.labels = 'labels.txt'
        self.size = args.fp_set_size  # size of fingerprint set
        self.fingerprint_set = []
        self.save_fp = args.save_fp
        self.lr = args.lr
        self.lradj = args.lradj
        self.loadmodel = args.loadmodel
        self.arch = args.arch
