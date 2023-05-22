"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import time

import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm

import test_config
import torch
from datasets import get_dataset
from models import get_model
from utils.utils import Cluster,Visualizer,Metrics
from skimage.color import label2rgb
from skimage import io
import numpy as np

   


def begin_test(args,n_sigma=2):
    torch.backends.cudnn.benchmark = True


#     if args['display']:
#         plt.ion()
#     else:
#         plt.ioff()
#         plt.switch_backend("agg")

    if args['save']:
        if not os.path.exists(args['save_dir']):
            os.makedirs(args['save_dir'])

    # set device
    device = torch.device("cuda:0" if args['cuda'] else "cpu")
#     device = "cpu"

    # dataloader
    dataset = get_dataset(args['dataset']['name'], args['dataset']['kwargs'])
    dataset_it = torch.utils.data.DataLoader(
                                dataset, 
                                batch_size=1, 
                                shuffle=False, 
                                drop_last=False, 
                                num_workers=2, )
    
                                # pin_memory=True if args['cuda'] else False)

    # load model
    model = get_model(args['model']['name'], args['model']['kwargs'])
    model = torch.nn.DataParallel(model).to(device)
    num_class = args['model']['kwargs']['num_classes'][1]

    # load snapshot
    if os.path.exists(args['checkpoint_path']):
        state = torch.load(args['checkpoint_path'])
#         state = torch.load(args['checkpoint_path'], map_location='cpu')
        model.load_state_dict(state['model_state_dict'], strict=True)
    else:
        assert False, 'checkpoint_path {} does not exist!'.format(args['checkpoint_path'])
#     print(model.device)

    model.eval()

    # cluster module
    cluster = Cluster(grid_size=1024,device=device)

    # Visualizer
    visualizer = Visualizer(args)
    metrics = Metrics(num_class=num_class)

    with torch.no_grad():

        for sample in tqdm(dataset_it):

            # im = sample['image'].to(device)
            im = sample['hs'].to(device)
            
            label = sample['label'].squeeze()
        
            output,_ = model(im)
            instance_pred, class_pred,score = cluster.cluster_local_maxima(output[0], n_sigma=2,num_class=num_class)
            pred_score = torch.sigmoid(output[0][2+n_sigma:])
            metrics.add(label.numpy(),class_pred.numpy(),pred_score.cpu().numpy())
            if args['save']:
                img = io.imread(sample["im_name"][0])
                # instance = visualizer.instance2color(instances.numpy())
                label = visualizer.label2colormap(label.numpy())
                instance_pred = visualizer.instance2color(instance_pred.cpu().numpy())
                class_pred = visualizer.label2colormap(class_pred.cpu().numpy())
                
                ground_truth = visualizer.overlay_image(img[...,:3],label)
                grid = np.concatenate((ground_truth,class_pred,instance_pred),axis=1)
                base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))
                io.imsave(os.path.join(args['save_dir'], base+'.png'),grid)
                txt_file = os.path.join(args['save_dir'], base + '.txt')
                with open(txt_file, 'w') as f:
                    # loop over instances
                    for id, pred in enumerate(score):
                        # write to file
                        f.writelines("{}_{} {:.02f}\n".format(base, id, pred))

#             if args['save']:
#                 img = io.imread(sample["im_name"][0])
#                 # instance = visualizer.instance2color(instances.numpy())
#                 label = visualizer.label2colormap(label.numpy())
#                 instance_pred = visualizer.instance2color(instance_pred.numpy())
#                 class_pred = visualizer.label2colormap(class_pred.numpy())
                
#                 ground_truth = visualizer.overlay_image(img[...,:3],label)
#                 grid = np.concatenate((ground_truth,class_pred,instance_pred),axis=1)
#                 base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))
#                 io.imsave(os.path.join(args['save_dir'], base+'.png'),grid)
#                 txt_file = os.path.join(args['save_dir'], base + '.txt')
#                 with open(txt_file, 'w') as f:
#                     # loop over instances
#                     for id, pred in enumerate(score):
#                         # write to file
#                         f.writelines("{}_{} {:.02f}\n".format(base, id, pred))

        metrics.log("evaluation.txt")


if __name__=="__main__":
    args = test_config.get_args()
    begin_test(args)