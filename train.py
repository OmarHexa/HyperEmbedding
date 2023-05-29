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
from criterions.my_loss import SpatialEmbLoss,FocalLoss, MulticlassLoss
from datasets import get_dataset
from models import get_model
from utils.utils import AverageMeter, Cluster, Logger , Visualizer
from torchvision.utils import make_grid
from skimage import io
from skimage.color import label2rgb
import numpy as np
from models.hypernet import HyperNet,Discriminator
import torch.nn.functional as F
from criterions.my_loss import calculate_iou


def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    v = F.softmax(v,dim=1)
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))
def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)
def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.device)
    return torch.nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)


# def trainUDA(model,optimizer,criterion, source_loader,target_loader,device,factor=1):
#      # define meters
#     loss_meter,aux_meter = AverageMeter(), AverageMeter()
#     # put model into training mode
#     model.train()

#     for param_group in optimizer.param_groups:
#         print('learning rate: {}'.format(param_group['lr']))

#     target_iter = iter(target_loader)
#     with torch.autograd.set_detect_anomaly(True):
#         for i, sample_src in enumerate(tqdm(source_loader)):
#             optimizer.zero_grad()
            
#             # training model using source data
#             s_img =  sample_src['hs'].to(device)
#             s_inst = sample_src['instance'].squeeze().to(device)
#             s_label = sample_src['label'].squeeze().to(device)

#             # forward source data
#             s_output,F = model(s_img)
        
#             emb_loss = criterion(s_output,s_inst,s_label)
#             aux = criterion.module.auxiliary_loss(s_label,F[0])
                
#             loss = emb_loss.mean()+aux.mean()
#             loss.backward()
#             loss_meter.update(loss.item())

#             # training model using target data
#             try:
#                 sample_target = next(target_iter)
#             except StopIteration:
#                 target_iter = iter(target_loader)
#                 sample_target = next(target_iter)
            
#             t_img = sample_target["hs"].to(device)
        
#             t_output, _ = model(t_img)
#             loss = factor* entropy_loss(t_output[:,4:])
        
#             loss.backward()
#             optimizer.step()
        
#             aux_meter.update(aux.item())
            
#     return loss_meter.avg, aux_meter.avg

def trainadvent(model,discriminator, optim_seg,optim_dis, criterion, source_loader,target_loader,device,factor=0.001):
     # define meters
    loss_meter = AverageMeter()
    # put model into training mode
    model.train()
    discriminator.train()

    for param_group in optim_seg.param_groups:
        print('learning rate: {}'.format(param_group['lr']))

    target_iter = iter(target_loader)
    # labels for adversarial training
    source_domain = 0
    target_domain = 1
    with torch.autograd.set_detect_anomaly(True):
        for i, sample_src in enumerate(tqdm(source_loader)):
            optim_seg.zero_grad()
            optim_dis.zero_grad()
            # only train segnet. Don't accumulate grads in disciminators
            for param in discriminator.parameters():
                param.requires_grad = False
            # training model using source data
            s_img =  sample_src['hs'].to(device)
            s_label = sample_src['label'].squeeze().to(device)

            # forward source data
            s_output,_ = model(s_img)
            loss = criterion(s_output,s_label)
            loss = loss.mean()
            loss.backward()
            loss_meter.update(loss.item())


            # training model using target data
            try:
                sample_target = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                sample_target = next(target_iter)
            t_img = sample_target["hs"].to(device)

            
            t_output, _ = model(t_img)
            d_out = discriminator(prob_2_entropy(F.softmax(t_output,dim=1)))
            loss = factor* bce_loss(d_out,source_domain)
            loss = loss.mean()
            loss.backward()
            
            #trianing discriminator with source
            for param in discriminator.parameters():
                param.requires_grad = True
            
            s_out = s_output.detach()
            d_out = discriminator(prob_2_entropy(F.softmax(s_out,dim=1)))
            
            loss = (bce_loss(d_out,source_domain))/2
            loss = loss.mean()
            loss.backward()
            #trianing discriminator with target
            t_out = t_output.detach()
            d_out = discriminator(prob_2_entropy(F.softmax(t_out,dim=1)))
            loss = (bce_loss(d_out,target_domain))/2
            loss = loss.mean()
            loss.backward()
            
            optim_seg.step()
            optim_dis.step()
            
    return loss_meter.avg


def valDA(args,model,criterion, source_loader,target_loader,visualizer,device,epoch):
     # define meters
    loss_meter_src,loss_meter_target = AverageMeter(), AverageMeter()
    iou_meter_src, iou_meter_target = AverageMeter(), AverageMeter()
    
    
    # put model into training mode
    model.eval()
    
    target_iter = iter(target_loader)
    with torch.no_grad():
        for i, sample_src in enumerate(tqdm(source_loader)):

            s_img =  sample_src['hs'].to(device)
            s_label = sample_src['label'].squeeze().to(device)
            # forward source data
            s_output,_ = model(s_img)

            loss = criterion(s_output, s_label)
            loss = loss.mean()
            #calculate iou
            s_out = F.softmax(s_output,dim=1)
            fg = ~(s_out[:,0]>0.5)
            iou = calculate_iou(fg,s_label>0)
            #update meter
            iou_meter_src.update(iou)
            loss_meter_src.update(loss.item())
            
            
            # validate model using target data
            try:
                sample_target = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                sample_target = next(target_iter)
            t_img = sample_target["hs"].to(device)
            t_label = sample_target['label'].squeeze().to(device)
        
            t_output,_ = model(t_img)
            loss = criterion(t_output,t_label)
            loss = loss.mean()
            #calculate iou
            t_out = F.softmax(t_output,dim=1)
            fg = ~(t_out[:,0]>0.5)
            iou = calculate_iou(fg,t_label>0)
            
            #update meter
            iou_meter_target.update(iou)
            loss_meter_target.update(loss.item())
            
        if args['save']:
            t_image = sample_target['image'][0]
            t_image = (t_image.numpy() *255).transpose(1,2,0)
            t_label = t_label[0].cpu().numpy()
            t_label = visualizer.label2colormap(t_label)
            t_gt = torch.from_numpy(visualizer.overlay_image(t_image,t_label)).permute(2,0,1)
            t_pred = visualizer.prepare_segment(t_out[0].cpu())
            
            
            s_image = sample_src['image'][0]
            s_image = (s_image.numpy() *255).transpose(1,2,0)
            s_label = s_label[0].cpu().numpy()
            s_label = visualizer.label2colormap(s_label)
            s_gt = torch.from_numpy(visualizer.overlay_image(s_image,s_label)).permute(2,0,1)
            s_pred = visualizer.prepare_segment(s_out[0].cpu())
            
            
            
            base_t, _ = os.path.splitext(os.path.basename(sample_target['im_name'][0]))
            name = os.path.join(args['save_dir'], 'epoch_'+str(epoch)+"_"+base_t+'.png')   
                             
                
            grid = make_grid([s_gt,s_pred,t_gt,t_pred],nrow=2)
            grid = grid.permute(1,2,0).numpy()
            io.imsave(name,grid)
            
    return loss_meter_src.avg, loss_meter_target.avg, iou_meter_src.avg, iou_meter_target.avg
    
# def valUDA(args,model,criterion, source_loader,target_loader,visualizer,device,epoch):
#      # define meters
#     loss_meter_src, iou_meter_src = AverageMeter(), AverageMeter()
#     loss_meter_target,iou_meter_target = AverageMeter(), AverageMeter()
    
    
#     # assert len(source_loader)>=len(target_loader)
#     # put model into training mode
#     model.eval()
    
#     target_iter = iter(target_loader)
#     with torch.no_grad():
#         for i, sample_src in enumerate(tqdm(source_loader)):

#              # training model using source data
#             s_img =  sample_src['hs'].to(device)
#             s_inst = sample_src['instance'].squeeze().to(device)
#             s_label = sample_src['label'].squeeze().to(device)
#             #create source domain label
#             batch_size = s_img.size(0)
#             domain_label = torch.zeros(batch_size)
#             domain_label = domain_label.long().to(device)
#             # forward source data
#             output,F = model(s_img)

#             emb_loss = criterion(output,s_inst, s_label, iou=True, iou_meter=iou_meter_src)
#             aux1 = criterion.module.auxiliary_loss(s_label,F[0])
#             aux2 = criterion.module.auxiliary_loss(s_label,F[1])
#             aux3 = criterion.module.auxiliary_loss(s_label,F[2])
#             src_loss = emb_loss.mean()+(0.5*aux1.mean())+(0.25*aux2.mean())+(0.125*aux3.mean())
#             loss_meter_src.update(src_loss.item())
            
            
#             # validate model using target data
#             try:
#                 sample_target = next(target_iter)
#             except StopIteration:
#                 target_iter = iter(target_loader)
#                 sample_target = next(target_iter)
#             t_img = sample_target["hs"].to(device)
#             t_inst = sample_target['instance'].squeeze().to(device)
#             t_label = sample_target['label'].squeeze().to(device)
            
            
#             batch_size = t_img.size(0)
#             domain_label = torch.ones(batch_size)
#             domain_label = domain_label.long().to(device)
        
#             output,F = model(t_img)
#             emb_loss = criterion(output,t_inst, t_label, iou=True, iou_meter=iou_meter_target)
#             aux1 = criterion.module.auxiliary_loss(t_label,F[0])
#             aux2 = criterion.module.auxiliary_loss(t_label,F[1])
#             aux3 = criterion.module.auxiliary_loss(t_label,F[2])
        
#             target_loss = emb_loss.mean()+(0.5*aux1.mean())+(0.25*aux2.mean())+(0.125*aux3.mean())
#             loss_meter_target.update(target_loss.item())
            
#         if args['save']:
#             image = sample_target['image'][0]
#             image = (image.numpy() *255).transpose(1,2,0)
#             labels = t_label[0].cpu().numpy()
                
#             base, _ = os.path.splitext(os.path.basename(sample_target['im_name'][0]))
#             name = os.path.join(args['save_dir'], 'epoch_'+str(epoch)+base+'.png')
#             labels = visualizer.label2colormap(labels)
#             gt = torch.from_numpy(visualizer.overlay_image(image,labels)).permute(2,0,1)
                
                
#             offset, sigma,pred = visualizer.prepare_internal(output=output[0].cpu())
                
#             grid = make_grid([gt,pred,offset,sigma],nrow=2)
#             grid = grid.permute(1,2,0).numpy()
#             io.imsave(name,grid)
#     return loss_meter_src.avg, iou_meter_src.avg, loss_meter_target.avg, iou_meter_target.avg
 


# def train(args,model,optimizer,criterion,train_dataloader,device):

#     # define meters
#     loss_meter = AverageMeter()
#     # put model into training mode
#     model.train()

#     for param_group in optimizer.param_groups:
#         print('learning rate: {}'.format(param_group['lr']))

#     for i, sample in enumerate(tqdm(train_dataloader)):

#         im = sample['hs'].to(device)
#         # im = sample['image'].to(device)

#         instances = sample['instance'].squeeze().to(device)
#         class_labels = sample['label'].squeeze().to(device)

#         output,F = model(im)
#         loss = criterion(output,instances, class_labels) 
#         aux = criterion.module.auxiliary_loss(class_labels,F[0]) 
#                 # +(0.5*criterion.module.auxiliary_loss(class_labels,F[1]))\
#                 # +(0.5*criterion.module.auxiliary_loss(class_labels,F[2]))
        
#         loss = loss.mean()+ (0.5*aux.mean())
#         # loss = loss.mean()

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         loss_meter.update(loss.item())
#     return loss_meter.avg


# def val(args,model,criterion,val_dataloader,visualizer,device,epoch):

#     # define meters
#     loss_meter, iou_meter = AverageMeter(), AverageMeter()

#     # put model into eval mode
#     model.eval()

#     with torch.no_grad():

#         for i, sample in enumerate(tqdm(val_dataloader)):

#             im = sample['hs'].to(device)
#             # im = sample['image'].to(device)

#             instances = sample['instance'].squeeze().to(device)
#             class_labels = sample['label'].squeeze().to(device)

#             output,F = model(im)
#             loss = criterion(output,instances, class_labels, iou=True, iou_meter=iou_meter)
#             aux = criterion.module.auxiliary_loss( class_labels,F[0]) 
#                 # +(0.5*criterion.module.auxiliary_loss( class_labels,F[1]))\
#                 # +(0.25*criterion.module.auxiliary_loss( class_labels,F[2]))
#             loss = loss.mean()+ (0.5*aux.mean())
#             # loss = loss.mean()
#             loss_meter.update(loss.item())
            
#         if args['save']:
#             image = sample['image'][0]
#             image = (image.numpy() *255).transpose(1,2,0)
#             labels = class_labels[0].cpu().numpy()
                
#             base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))
#             name = os.path.join(args['save_dir'], 'epoch_'+str(epoch)+'_'+base+'.png')
#             labels = visualizer.label2colormap(labels)
#             gt = torch.from_numpy(visualizer.overlay_image(image,labels)).permute(2,0,1)

#             offset, sigma,pred = visualizer.prepare_internal(output=output[0].cpu())
                
#             grid = make_grid([gt,pred,offset,sigma],nrow=2)
#             grid = grid.permute(1,2,0).numpy()
#             io.imsave(name,grid)
#             print("image saved as {}".format(name))

#     return loss_meter.avg, iou_meter.avg




# def begin_trianing(args,device):
#     torch.backends.cudnn.benchmark = True


# # train dataloader
#     train_dataset = get_dataset(args['train_dataset']['name'], args['train_dataset']['kwargs'])
#     train_dataloader = torch.utils.data.DataLoader(
#                                     train_dataset,
#                                     batch_size=args['train_dataset']['batch_size'],
#                                     shuffle=True,
#                                     drop_last=True,
#                                     num_workers=args['train_dataset']['workers'])



# # val dataloader
#     val_dataset = get_dataset(args['val_dataset']['name'], args['val_dataset']['kwargs'])
#     val_dataloader = torch.utils.data.DataLoader(
#                                     val_dataset,
#                                     batch_size=args['val_dataset']['batch_size'],
#                                     shuffle=True,
#                                     drop_last=True,
#                                     num_workers=args['train_dataset']['workers'])


# # set model
#     model = get_model(args['model']['name'], args['model']['kwargs'])
#     model.init_output(args['loss_opts']['n_sigma'])
#     model = torch.nn.DataParallel(model).to(device)

# # set criterion
#     criterion = SpatialEmbLoss(**args['loss_opts'])
#     criterion = torch.nn.DataParallel(criterion).to(device)

# # set optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=1e-4)


#     def lambda_(epoch):
#         return pow((1-((epoch)/args['n_epochs'])), 0.9)

#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_,)


#     # clustering
#     cluster = Cluster(args['grid_size'],device=device)


#     # Logger
#     logger = Logger(('train', 'val', 'iou'), 'loss')
    
#     #visualizer
#     visualizer = Visualizer(args)

#     # resume
#     start_epoch = 0
#     best_iou = 0
#     if args['resume_path'] is not None and os.path.exists(args['resume_path']):
#         print('Resuming model from {}'.format(args['resume_path']))
#         state = torch.load(args['resume_path'])
#         # model_dict = model.state_dict()
#         # pretrained_dict = state['model_state_dict']
#         # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
#         # model_dict.update(pretrained_dict)
#         # model.load_state_dict(model_dict)
#         start_epoch = state['epoch'] + 1
#         best_iou = state['best_iou']
#         model.load_state_dict(state['model_state_dict'], strict=True)
#         optimizer.load_state_dict(state['optim_state_dict'])
#         logger.data = state['logger_data']


#     def save_checkpoint(state, is_best, name='checkpoint.pth'):
#         print('=> saving checkpoint')
#         file_name = os.path.join(args['save_dir'], name)
#         torch.save(state, file_name)
#         if is_best:
#             shutil.copyfile(file_name, os.path.join(
#                 args['save_dir'], 'best_iou_model.pth'))

    
#     print(model.modules) #print modules for letter modules inspection
#     for epoch in range(start_epoch, args['n_epochs']):
        

#         print('Starting epoch {}'.format(epoch))

#         train_loss = train(args,model,optimizer,criterion,train_dataloader,device)
#         val_loss, val_iou = val(args,model,criterion,val_dataloader,visualizer,device,epoch=epoch)
#         scheduler.step()
        

#         print('===> train loss: {:.2f}'.format(train_loss))
#         print('===> val loss: {:.2f}, val iou: {:.2f}'.format(val_loss, val_iou))

#         logger.add('train', train_loss)
#         logger.add('val', val_loss)
#         logger.add('iou', val_iou)
#         logger.plot(save=args['save'], save_dir=args['save_dir'])

#         is_best = val_iou > best_iou
#         best_iou = max(val_iou, best_iou)

#         if args['save']:
#             state = {
#                 'epoch': epoch,
#                 'best_iou': best_iou,
#                 'model_state_dict': model.state_dict(),
#                 'optim_state_dict': optimizer.state_dict(),
#                 'logger_data': logger.data
#                     }
#             save_checkpoint(state, is_best)
def begin_trianing_with_DA(args,device):
    torch.backends.cudnn.benchmark = True


# train dataloader
    src_train_dataset = get_dataset(args['train_dataset']['name'], args['train_dataset']['kwargs'])
    src_train_dataloader = torch.utils.data.DataLoader(
                                    src_train_dataset,
                                    batch_size=args['train_dataset']['batch_size'],
                                    shuffle=True,
                                    drop_last=True,
                                    num_workers=args['train_dataset']['workers'])

# val dataloader
    src_val_dataset = get_dataset(args['val_dataset']['name'], args['val_dataset']['kwargs'])
    src_val_dataloader = torch.utils.data.DataLoader(
                                    src_val_dataset,
                                    batch_size=args['val_dataset']['batch_size'],
                                    shuffle=True,
                                    drop_last=True,
                                    num_workers=args['train_dataset']['workers'])
# target dataloader
    target_train_dataset = get_dataset(args['target_train_dataset']['name'], args['target_train_dataset']['kwargs'])
    target_train_dataloader = torch.utils.data.DataLoader(
                                    target_train_dataset,
                                    batch_size=args['target_train_dataset']['batch_size'],
                                    shuffle=True,
                                    drop_last=True,
                                    num_workers=args['target_train_dataset']['workers'])
    target_val_dataset = get_dataset(args['target_val_dataset']['name'], args['target_val_dataset']['kwargs'])
    target_val_dataloader = torch.utils.data.DataLoader(
                                    target_val_dataset,
                                    batch_size=args['target_val_dataset']['batch_size'],
                                    shuffle=True,
                                    drop_last=True,
                                    num_workers=args['target_val_dataset']['workers'])


    # set model
    
    model = HyperNet(**args['model']['kwargs'])
    discriminator = Discriminator()
    
    model = torch.nn.DataParallel(model).to(device)
    discriminator = torch.nn.DataParallel(discriminator).to(device)
    

# set criterion
    criterion = FocalLoss(weight=torch.FloatTensor(args["weights"]))
    
    criterion = torch.nn.DataParallel(criterion).to(device)
# set optimizer
    optim_seg = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=1e-4)
    optim_dis = torch.optim.Adam(discriminator.parameters(), lr=args['lr'], weight_decay=1e-4)
    


    def lambda_(epoch):
        return pow((1-((epoch)/args['n_epochs'])), 0.9)

    # Logger 
    logger = Logger(('train', "val", 'target', 'val_iou','tar_iou'), 'loss')
    
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
        # model_dict = model.state_dict()
        # pretrained_dict = state['model_state_dict']
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)
        model.load_state_dict(state['model_state_dict'], strict=True)
        optim_seg.load_state_dict(state['optim_state_dict'])
        logger.data = state['logger_data']


    def save_checkpoint(state, is_best, name='checkpoint.pth'):
        print('=> saving checkpoint')
        file_name = os.path.join(args['save_dir'], name)
        torch.save(state, file_name)
        if is_best:
            shutil.copyfile(file_name, os.path.join(
                args['save_dir'], 'best_iou_model.pth'))

    scheduler_seg = torch.optim.lr_scheduler.LambdaLR(optim_seg, lr_lambda=lambda_,)
    scheduler_dis = torch.optim.lr_scheduler.LambdaLR(optim_dis, lr_lambda=lambda_,)
    
    print(model.modules)
    
    for epoch in range(start_epoch, args['n_epochs']):
        
        print('Starting epoch {}'.format(epoch))
        train_loss = trainadvent(model,discriminator,optim_seg,optim_dis,criterion,src_train_dataloader,
                                                           target_train_dataloader,device)
        val_loss,tar_loss,val_iou,tar_iou = valDA(args,model,criterion,src_val_dataloader,target_train_dataloader,visualizer,device,epoch=epoch)
        scheduler_seg.step()
        scheduler_dis.step()
        
        
        

        print('===> train loss: {:.2f}'.format(train_loss))
        print('===> val loss: {:.2f}, val iou: {:.2f}'.format(val_loss, val_iou))

        logger.add('train', train_loss)
        logger.add('val', val_loss)
        logger.add('target', tar_loss)
        logger.add('val_iou', val_iou)
        logger.add('tar_iou', tar_iou)
        
        logger.plot(save=args['save'], save_dir=args['save_dir'])

        is_best = val_iou > best_iou
        best_iou = max(val_iou, best_iou)

        if args['save']:
            state = {
                'epoch': epoch,
                'best_iou': best_iou,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optim_seg.state_dict(),
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
    begin_trianing_with_DA(args,device=device)