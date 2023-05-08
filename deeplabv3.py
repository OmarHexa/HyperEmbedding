import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import transforms as my_transforms
import copy
import os
import shutil

from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
from datasets import get_dataset
from utils.utils import AverageMeter, Logger , Visualizer
from torchvision.utils import make_grid
from skimage import io
import numpy as np

H2GIGA_DIR='../Data/crops/H2giga'

args = dict(

    cuda=True,
    save=True,
    save_dir='./deepLabv3',
    resume_path=None, 
    color_map={0:(0,0,0),1: (21, 176, 26), 2:(5, 73, 7),3: (170, 166, 98),4: (229, 0, 0), 5: (140, 0, 15)},
    num_class = 6, #including background
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

    val_dataset = {
        'name': 'H2giga',
        'kwargs': {
            'root_dir': H2GIGA_DIR,
            'type': 'val',
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

    lr=5e-4,
    n_epochs=100,
    
)


def get_args():
    return copy.deepcopy(args)
 
def prepare_model(num_classes=6):
    model = deeplabv3_resnet50(weights='DEFAULT')
    model.classifier[4] = nn.Conv2d(256, num_classes, 1)
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1)
    return model

class MultiClassCrossEntropyLoss(nn.Module):
    """
    PyTorch module for the multi-class cross-entropy loss with binary ground truth masks per class.
    """

    def __init__(self):
        super(MultiClassCrossEntropyLoss, self).__init__()
    def forward(self, outputs, masks):
       
        batch_size, num_classes, _, _ = outputs.size()
       
        # flatten the predicted outputs and ground truth masks
        outputs = outputs.view(batch_size, num_classes, -1)
        masks = masks.view(batch_size, num_classes, -1)

        # compute the cross-entropy loss per class
        loss_per_class = torch.zeros(num_classes, device=outputs.device)
        for c in range(num_classes):
            loss_per_class[c] = F.binary_cross_entropy_with_logits(outputs[:,c,:], masks[:,c,:])

        # compute the final loss as the average loss per class
        loss = loss_per_class.mean()

        return loss


class MultiScaleCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MultiScaleCrossEntropyLoss, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average)
        
    def forward(self, inputs, targets):
        targets = targets.long()
        input_size = inputs.size()[2:]
        log_p = nn.functional.log_softmax(inputs, dim=1)
        
        # compute the negative log-likelihood of each sample
        # and take the mean across the batch
        loss = self.nll_loss(log_p, targets)
        
        return loss



def train(model,optimizer,criterion,train_dataloader,device):

    # define meters
    loss_meter = AverageMeter()

    # put model into training mode
    model.train()

    for param_group in optimizer.param_groups:
        print('learning rate: {}'.format(param_group['lr']))

    for i, sample in enumerate(tqdm(train_dataloader)):

        im = sample['image'].to(device)
        
        class_labels = sample['label'].squeeze().to(device)

        output = model(im)
        loss = criterion(output['out'],class_labels)
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())

    return loss_meter.avg


def val(args,model,criterion,val_dataloader,visualizer,device,epoch):

    # define meters
    loss_meter, iou_meter = AverageMeter(), AverageMeter()

    # put model into eval mode
    model.eval()

    with torch.no_grad():

        for i, sample in enumerate(tqdm(val_dataloader)):

            im = sample['image'].to(device)
            class_labels = sample['label'].squeeze().to(device)

            output = model(im)
            loss = criterion(output['out'],class_labels)
            loss = loss.mean()

            loss_meter.update(loss.item())
            
        if args['save']:
                image = sample['image'][0]
                image = (image *255)
                
                base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))
                name = os.path.join(args['save_dir'], 'epoch_'+str(epoch)+base+'.png')                
                
                pred = visualizer.prepare_segment(output['out'][0].cpu())
                
                grid = make_grid([image,pred])
                grid = grid.permute(1,2,0).numpy()
                io.imsave(name,grid)

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
                                    num_workers=args['train_dataset']['workers'],
                                    pin_memory=True if args['cuda'] else False)


# val dataloader
    val_dataset = get_dataset(args['val_dataset']['name'], args['val_dataset']['kwargs'])
    val_dataloader = torch.utils.data.DataLoader(
                                    val_dataset,
                                    batch_size=args['val_dataset']['batch_size'],
                                    shuffle=True,
                                    drop_last=True,
                                    num_workers=args['train_dataset']['workers'],
                                    pin_memory=True if args['cuda'] else False)


# set model
    model = prepare_model(args["num_class"])
    model = torch.nn.DataParallel(model).to(device)

# set criterion
    criterion = MultiScaleCrossEntropyLoss()
    criterion = torch.nn.DataParallel(criterion).to(device)

# set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=1e-4)


    def lambda_(epoch):
        return pow((1-((epoch)/args['n_epochs'])), 0.9)




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


    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_,)
    for epoch in range(start_epoch, args['n_epochs']):
        

        print('Starting epoch {}'.format(epoch))
        train_loss = train(model,optimizer,criterion,train_dataloader,device)
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
    args =get_args()
    if args['save']:
        if not os.path.exists(args['save_dir']):
            os.makedirs(args['save_dir'])
            print("created directory {}".format(args["save_dir"]))
    device = torch.device("cuda:0" if args['cuda'] & torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))
    begin_trianing(args,device=device)