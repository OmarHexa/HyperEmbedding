import glob
import os
import random

import numpy as np
import pandas as pd
from skimage import io
from skimage.color import rgba2rgb
from skimage.segmentation import relabel_sequential
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter

def band_quertile_norm(Data):
   [m, n, l] = np.shape(Data)
   for i in range(l):
      q3,q2,q1 = np.percentile(Data[:, :, i],[75,50,25])
      iqr = (q3 - q1)*1.35
      Data[:, :, i] = (Data[:, :, i] - q2)/iqr
   return Data

def normalize_min_max_percentile(x, pmin=3, pmax=99, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """
        Percentile-based image normalization.
        Function taken from StarDist repository  https://github.com/stardist/stardist
    """
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)

def infinity_norm(Data):
    max = np.max(Data,axis=-1,keepdims=True)
    # Scale each pixel by its feature vector length
    return Data / max

def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    """
        Percentile-based image normalization.
        Function taken from StarDist repository  https://github.com/stardist/stardist
    """
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

    if clip:
        x = np.clip(x, 0, 1)

    return x


class H2gigaDataset(Dataset):

    
    color_map = {0:(0,0,0),1: (21, 176, 26), 2:(5, 73, 7),3: (170, 166, 98),4: (229, 0, 0), 5: (140, 0, 15)}
    

    def __init__(self, root_dir='./', type="train", class_id=None, size=None, normalize=True, transform=None):

        # print('`{}` dataset created! Accessing data from {}/{}/'.format(type, root_dir, type))

        # get image and instance and class map list
        path = os.path.join(root_dir, '{}/'.format(type))
        if os.path.exists(os.path.join(path,'images')):
            self.image_list = sorted(glob.glob(os.path.join(path, 'images/*.png')))
            # print('Number of images in `{}` directory is {}'.format(type, len(self.image_list)))
        else:
            print('Image path does not exist')
        if os.path.exists(os.path.join(path,'instances')):
            self.instance_list = sorted(glob.glob(os.path.join(root_dir, '{}/'.format(type), 'instances/*.png')))
            # print('Number of instances in `{}` directory is {}'.format(type, len(self.instance_list)))
        else:
            print('Instance path does not exist')
        if os.path.exists(os.path.join(path,'classmaps')):
            self.classmap_list = sorted(glob.glob(os.path.join(root_dir, '{}/'.format(type), 'classmaps/*.png')))
            # print('Number of instances in `{}` directory is {}'.format(type, len(self.classmap_list)))
        else:
            print('Class_map path does not exist')
            
        if os.path.exists(os.path.join(path,'hs')):
            self.hs_list = sorted(glob.glob(os.path.join(root_dir, '{}/'.format(type), 'hs/*.npy')))
            # print('Number of hs in `{}` directory is {}'.format(type, len(self.hs_list)))
        else:
            print('hyperspectral path does not exist')
        
        
        self.class_id = class_id
        self.size = size
        self.real_size = len(self.image_list)
        self.normalize = normalize
        self.transform = transform
    def __len__(self):

        return self.real_size if self.size is None else self.size
    
    def __getitem__(self, index):

        index = index if self.size is None else random.randint(0, self.real_size-1)
        sample = {}

        # load image
        image = io.imread(self.image_list[index])
        if image.shape[-1]==4:
            image = rgba2rgb(image)
        # load hs cube
        hs = np.load(self.hs_list[index])[...,10:]
        
            
        if self.normalize:
            image = normalize_min_max_percentile(image, 1, 99.8, axis=(0, 1))
            hs = infinity_norm(hs)
            
        
        # normalize image
        
        sample['image'] = image
        sample['im_name'] = self.image_list[index]
        sample['hs'] = hs
        
        # load instances
        instance = io.imread(self.instance_list[index])
        label = io.imread(self.classmap_list[index])
        instance = self.convert_rgb2catagory(instance)
        label = self.convert_rgb2catagory(label,classMap=True)
        
        
        if self.class_id is not None:
            instance,label= self.decode_instance(label,instance,self.class_id)

        
        sample['instance'] = instance
        sample['label'] = label
            

        # transform
        if(self.transform is not None):
            return self.transform(sample)
        else:
            return sample
    
    
    @classmethod
    def decode_instance(cls, classmap,instance, class_id=None):

        instance_map = np.zeros(
            (instance.shape[0], instance.shape[1]), dtype=np.uint8)

        # contains the class of each instance, but will set the class of "unlabeled instances/groups" to bg
        class_map = np.zeros(
            (classmap.shape[0], classmap.shape[1]), dtype=np.uint8)

        if class_id is not None:
            mask = classmap==class_id
            if mask.sum() > 0:
                ids, _, _ = relabel_sequential(instance[mask])
                instance_map[mask] = ids
                class_map[mask] = 1
        return instance_map, class_map

    @classmethod
    def convert_rgb2catagory(cls,img,classMap=False):
        resolution = (img.shape[0],img.shape[1])
        intImage= np.zeros(resolution,dtype=np.uint8)
        
        if classMap:
            #individual color is assigned a fixed value according to the color map
            for i,color in cls.color_map.items():
                intImage[(img==color).all(axis=2)]=i
        else:
            colors= np.unique(img.reshape(-1, img.shape[-1]), axis=0)
            for i,color in enumerate(colors):
                # every instances is given a unique interger value.
                intImage[(img==color).all(axis=2)]=i
        return intImage
    
    
    
    
    
if __name__=="__main__":
    
    # from utils.transforms import get_transform
    dir = '../augmented_data/H2giga'
    
    
    data = H2gigaDataset(dir,type='val')
    sample = data.__getitem__(0)
    print(sample['hs'].shape)