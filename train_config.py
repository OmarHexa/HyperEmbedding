"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import copy
import os

import torch
from utils import transforms as my_transforms

H2GIGA_DIR='../Data/augmented/H2giga'

args = dict(

    cuda=True,
    save=True,
    save_dir='./exprgb_coloraug',
    resume_path=None, 
    color_map={0:(0,0,0),1: (21, 176, 26), 2:(5, 73, 7),3: (170, 166, 98),4: (229, 0, 0), 5: (140, 0, 15)},
    num_class = 5,
    train_dataset = {
        'name': 'H2giga',
        'kwargs': {
            'root_dir': H2GIGA_DIR,
            'type': 'train',
            'class_id':None,
            'size': None,
            'normalize':True,
            'transform': my_transforms.get_transform([
                 {
                    'name': 'RandomRotationsAndFlips',
                    'opts': {
                        'keys': ('image', 'instance','label'),
                        'degrees': 90,
                    }
                },
                  {
                    'name': 'ColorAugmentation',
                    'opts': {
                        'keys': ('image', 'instance', 'label'),
                    }
                },
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image','instance', 'label'),
                        'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                            }
                }
                ]),
                },
            
            'batch_size': 20,
            'workers': 4,
        }, 

    val_dataset = {
        'name': 'H2giga',
        'kwargs': {
            'root_dir': H2GIGA_DIR,
            'type': 'val',
            'normalize':True,
            'transform': my_transforms.get_transform([
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image','instance', 'label'),
                        'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                            }
                },
                ]),
                },
        'batch_size': 20,
        'workers': 4,
    }, 

    model = {
        'name': 'branched_hypernet', 
        'kwargs': {
            'in_channel': 3,
            'num_classes': [4,5]
        }
    }, 

    lr=5e-4,
    n_epochs=120,
    grid_size=1024,

    # loss options
    loss_opts={
        'class_weight': [7.842, 6.839, 5.683, 9.029, 9.533],
        'num_class': 5,
        'n_sigma': 2
    },
    
)


def get_args():
    return copy.deepcopy(args)
# [7.842, 6.839, 5.683, 9.029, 9.533]