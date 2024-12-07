import argparse
import traceback

import torch
import os
import numpy as np

import math

from helpers.loaders import get_wm_transform
from helpers.image_folder_custom_class import ImageFolderCustomClass
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='test the models')

parser.add_argument('--method', default='', choices=['', 'noise1', 'noise2', 'noise3', 'normal', 'ood1', 'ood2',
                                                     'adversarial_frontier', 'ae_frontier'])
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--cuda', default='cuda:2', help='set cuda (e.g. cuda:0)')
parser.add_argument('--R', action='store_true')
parser.add_argument('--P', default=100)

N_list = [5, 10, 15]
epsilon_list = [0, 0.025, 0.05, 0.075, 0.1]

args = parser.parse_args()

def module_i(phi_matrix, model_list, trigger_list, trigger_now):
    M_list = []
    for i in model_list:
        flag = False
        for M_id, M in enumerate(M_list):
            for f in M:
                flag1 = True
                for j in trigger_list:
                    if phi_matrix[i][j] != phi_matrix[f][j]:
                        flag1 = False
                        break
                if flag1:
                    M_list[M_id].append(i)
                    flag = True
                break
            if flag: break
        if not flag:
            M_list.append([i])
                
    # print(len(M_list))
            
    h = 0
    u = np.zeros(10)
    for M in M_list:
        # print(len(M), M)
        for c in range(10):
            u[c] = 0
            for f in M:
                if phi_matrix[f][trigger_now] == c:
                    u[c] += 1
            u[c] = 1.0 * u[c] / len(M)
            if u[c] == 0:
                xlogx = 0
            else:
                xlogx = u[c] * math.log2(u[c])
            h = h - (len(M) / len(model_list)) * xlogx
    return h


def module_r(phi_matrix, model_list, trigger_list, epsilon):

    load_path = os.path.join(cwd, 'result', 'R_list', str(args.dataset), 'e2d', args.method + '.txt')
    data = np.loadtxt(load_path)
    delta = 0
    for i in range(len(data)):
        if epsilon == data[i][0]:
            delta = data[i][1]
            break
    r_num = 0
    for fi in model_list:
        for fj in model_list:
            if fj != fi:
                dif = 0
                for img in trigger_list:
                    if phi_matrix[fj][img] != phi_matrix[fi][img]:
                        dif += 1
                if dif <= math.floor(delta * len(trigger_list)):
                    r_num += 1
                    break
    return r_num / len(model_list)


try:
    device = torch.device(args.cuda) if torch.cuda.is_available() else 'cpu'
    cwd = os.getcwd()

    test_method = args.method
    
    load_path = os.path.join(cwd, 'matrix', str(args.dataset), test_method + '.txt')

    phi_matrix = np.loadtxt(load_path)
    
    if args.R:
        os.makedirs(os.path.join(cwd, 'result', f'P{args.P}', 'R_index', str(args.dataset)), exist_ok=True)
        save_path = os.path.join(cwd, 'result', f'P{args.P}', 'R_index', str(args.dataset), test_method + '.txt')

        model_list = range(600)
        model_list_part = np.loadtxt(os.path.join(cwd, 'matrix', f'P{args.P}.txt'), dtype=int)

        for N in N_list:
            for epsilon in epsilon_list:
                choose_list = []
                left_list = list(range(50))
                while len(choose_list) < N:
                    hmax = -1e7
                    r = 0
                    for i in left_list:
                        h = module_i(phi_matrix, model_list_part, choose_list, i)
                        if h > hmax:
                            hmax = h
                            r = i
                    choose_list.append(r)
                    left_list.remove(r)

                epsilon_float = "{:.3f}".format(epsilon)
                r_rate = "{:.3f}".format(module_r(phi_matrix, model_list, choose_list, epsilon))

                with open(save_path, "a") as file:
                    file.write(f"{N}\t{epsilon_float}\t{r_rate}\n")
    
    else:
        os.makedirs(os.path.join(cwd, 'result', f'P{args.P}', 'I_index', str(args.dataset)), exist_ok=True)
        save_path = os.path.join(cwd, 'result', f'P{args.P}', 'I_index', str(args.dataset), test_method + '.txt')

        result = np.array([])
        model_list = range(600)
        model_list_part = np.loadtxt(os.path.join(cwd, 'matrix', f'P{args.P}.txt'), dtype=int)

        choose_list = []
        left_list = list(range(50))

        while len(left_list) > 0:
            hmax = -1e7
            r = 0
            for i in left_list:
                h = module_i(phi_matrix, model_list_part, choose_list, i)
                if h > hmax:
                    hmax = h
                    r = i
            
            result = np.append(result, module_i(phi_matrix, model_list, choose_list, r))

            choose_list.append(r)
            left_list.remove(r)
        
        np.savetxt(save_path, result, fmt='%.10f')


except Exception as e:
    msg = 'An error occurred during setup: ' + str(e)
    print(msg)

    traceback.print_tb(e.__traceback__)