"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import copy

import torch
from utils import transforms as my_transforms

H2GIGA_DIR='../Data/H2giga'


args = dict(

    cuda=True,
    display=True,

    save=True,
    save_dir='./test/hs_c2ftriAsppf_inf',
    checkpoint_path='exp/hs_c2ftriAsppf_inf/best_iou_model.pth',
    color_map={0:(0,0,0),1: (21, 176, 26), 2:(5, 73, 7),3: (170, 166, 98),4: (229, 0, 0), 5: (140, 0, 15)},
    num_class = 5,
    dataset= { 
        'name': 'H2giga',
        'kwargs': {
            'root_dir': H2GIGA_DIR,
            'type': 'val',
            'class_id': None,            
            'transform': my_transforms.get_transform([
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image','hs','instance', 'label'),
                        'type': (torch.FloatTensor,torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                    }
                },
            ]),
        }
    },
        
    model = {
        'name': 'branched_hypernet',
        'kwargs': {
            'in_channel': 154,
            'num_classes': [4, 5],
        }
    }
)


def get_args():
    return copy.deepcopy(args)
