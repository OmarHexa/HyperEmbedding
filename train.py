"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import shutil
import time

from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import train_config
from criterions.my_loss import SpatialEmbLoss
from datasets import get_dataset
from models import get_model
from utils.utils import AverageMeter, Cluster, Logger , Visualizer
from torchvision.utils import make_grid
from skimage import io
from skimage.color import label2rgb
import numpy as np


def train(args,model,optimizer,criterion,train_dataloader,device):

    # define meters
    loss_meter = AverageMeter()
    # put model into training mode
    model.train()

    for param_group in optimizer.param_groups:
        print('learning rate: {}'.format(param_group['lr']))

    for i, sample in enumerate(tqdm(train_dataloader)):

        hs = sample['hs'].to(device)
        im = sample['image'].to(device)

        instances = sample['instance'].squeeze().to(device)
        class_labels = sample['label'].squeeze().to(device)

        output,F = model(hs,im)
        loss = criterion(output,instances, class_labels)
        aux = criterion.module.auxiliary_loss(class_labels,F[0]) \
                +(0.5*criterion.module.auxiliary_loss(class_labels,F[1]))\
                +(0.25*criterion.module.auxiliary_loss(class_labels,F[2]))
        loss = loss.mean()+ aux.mean()
        # loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
        del sample, output
    return loss_meter.avg


def val(args,model,criterion,val_dataloader,visualizer,device,epoch):

    # define meters
    loss_meter, iou_meter = AverageMeter(), AverageMeter()

    # put model into eval mode
    model.eval()

    with torch.no_grad():

        for i, sample in enumerate(tqdm(val_dataloader)):

            hs = sample['hs'].to(device)
            im = sample['image'].to(device)

            instances = sample['instance'].squeeze().to(device)
            class_labels = sample['label'].squeeze().to(device)

            output,F = model(hs,im)
            loss = criterion(output,instances, class_labels, iou=True, iou_meter=iou_meter)
            aux = criterion.module.auxiliary_loss( class_labels,F[0]) \
                +(0.5*criterion.module.auxiliary_loss( class_labels,F[1]))\
                +(0.25*criterion.module.auxiliary_loss( class_labels,F[2]))
            loss = loss.mean()+ aux.mean()
            # loss = loss.mean()

            loss_meter.update(loss.item())
            
        if args['save']:
            image = sample['image'][0]
            image = (image.numpy() *255).transpose(1,2,0)
            labels = class_labels[0].cpu().numpy()
                
            base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))
            name = os.path.join(args['save_dir'], 'epoch_'+str(epoch)+'_'+base+'.png')
            labels = visualizer.label2colormap(labels)
            gt = torch.from_numpy(visualizer.overlay_image(image,labels)).permute(2,0,1)
                
                
            offset, sigma,pred = visualizer.prepare_internal(output=output[0].cpu())
                
            grid = make_grid([gt,pred,offset,sigma],nrow=2)
            grid = grid.permute(1,2,0).numpy()
            io.imsave(name,grid)
            print("image saved as {}".format(name))

    return loss_meter.avg, iou_meter.avg




def begin_trianing(args,device):
    torch.backends.cudnn.benchmark = True


# train dataloader
    train_dataset = get_dataset(args['train_dataset']['name'], args['train_dataset']['kwargs'])
    train_dataloader = torch.utils.data.DataLoader(
                                    train_dataset,
                                    batch_size=args['train_dataset']['batch_size'],
                                    shuffle=True,
                                    drop_last=True,
                                    num_workers=args['train_dataset']['workers'])
#                                     pin_memory=True if args['cuda'] else False)


# val dataloader
    val_dataset = get_dataset(args['val_dataset']['name'], args['val_dataset']['kwargs'])
    val_dataloader = torch.utils.data.DataLoader(
                                    val_dataset,
                                    batch_size=args['val_dataset']['batch_size'],
                                    shuffle=True,
                                    drop_last=False,
                                    num_workers=args['train_dataset']['workers'])
#                                     pin_memory=True if args['cuda'] else False)


# set model
    model = get_model(args['model']['name'], args['model']['kwargs'])
    model.init_output(args['loss_opts']['n_sigma'])
    model = torch.nn.DataParallel(model).to(device)

# set criterion
    criterion = SpatialEmbLoss(**args['loss_opts'])
    criterion = torch.nn.DataParallel(criterion).to(device)

# set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=1e-4)


    def lambda_(epoch):
        return pow((1-((epoch)/args['n_epochs'])), 0.9)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_,)


    # clustering
    cluster = Cluster(args['grid_size'],device=device)


    # Logger
    logger = Logger(('train', 'val', 'iou'), 'loss')
    
    #visualizer
    visualizer = Visualizer(args)

    # resume
    start_epoch = 0
    best_iou = 0
    if args['resume_path'] is not None and os.path.exists(args['resume_path']):
        print('Resuming model from {}'.format(args['resume_path']))
        state = torch.load(args['resume_path'])
        start_epoch = state['epoch'] + 1
        best_iou = state['best_iou']
        model.load_state_dict(state['model_state_dict'], strict=True)
        optimizer.load_state_dict(state['optim_state_dict'])
        logger.data = state['logger_data']


    def save_checkpoint(state, is_best, name='checkpoint.pth'):
        print('=> saving checkpoint')
        file_name = os.path.join(args['save_dir'], name)
        torch.save(state, file_name)
        if is_best:
            shutil.copyfile(file_name, os.path.join(
                args['save_dir'], 'best_iou_model.pth'))


    for epoch in range(start_epoch, args['n_epochs']):
        

        print('Starting epoch {}'.format(epoch))

        train_loss = train(args,model,optimizer,criterion,train_dataloader,device)
        val_loss, val_iou = val(args,model,criterion,val_dataloader,visualizer,device,epoch=epoch)
        scheduler.step()
        

        print('===> train loss: {:.2f}'.format(train_loss))
        print('===> val loss: {:.2f}, val iou: {:.2f}'.format(val_loss, val_iou))

        logger.add('train', train_loss)
        logger.add('val', val_loss)
        logger.add('iou', val_iou)
        logger.plot(save=args['save'], save_dir=args['save_dir'])

        is_best = val_iou > best_iou
        best_iou = max(val_iou, best_iou)

        if args['save']:
            state = {
                'epoch': epoch,
                'best_iou': best_iou,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'logger_data': logger.data
                    }
            save_checkpoint(state, is_best)


if __name__=="__main__":
    args =train_config.get_args()
    if args['save']:
        if not os.path.exists(args['save_dir']):
            os.makedirs(args['save_dir'])
            print("created directory {}".format(args["save_dir"]))
    device = torch.device("cuda:0" if args['cuda'] & torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))
    begin_trianing(args,device=device)