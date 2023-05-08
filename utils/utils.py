"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import threading

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from skimage.color import label2rgb
from skimage import io
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix




import torch


class AverageMeter(object):

    def __init__(self, num_classes=1):
        self.num_classes = num_classes
        self.reset()
        self.lock = threading.Lock()

    def reset(self):
        self.sum = [0] * self.num_classes
        self.count = [0] * self.num_classes
        self.avg_per_class = [0] * self.num_classes
        self.avg = 0

    def update(self, val, cl=0):
        with self.lock:
            self.sum[cl] += val
            self.count[cl] += 1
            self.avg_per_class = [
                x/y if x > 0 else 0 for x, y in zip(self.sum, self.count)]
            self.avg = sum(self.avg_per_class)/len(self.avg_per_class)


class Cluster:

    def __init__(self,grid_size,device):

        xm = torch.linspace(0, 1, grid_size).view(1, 1, -1).expand(1, grid_size, grid_size)
        ym = torch.linspace(0, 1, grid_size).view(1, -1, 1).expand(1, grid_size, grid_size)
        xym = torch.cat((xm, ym), 0)

        self.xym = xym.to(device)
        self.device = device

    def cluster_with_gt(self, prediction, instance, n_sigma=2):

        height, width = prediction.size(1), prediction.size(2)
    
        xym_s = self.xym[:, 0:height, 0:width]  # 2 x h x w
    
        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2:2+n_sigma]  # n_sigma x h x w
    
        instance_map = torch.zeros(height, width).byte().to(self.device)
    
        unique_instances = instance.unique()
        unique_instances = unique_instances[unique_instances != 0]
    
        for id in unique_instances:
    
            mask = instance.eq(id).view(1, height, width)
    
            center = spatial_emb[mask.expand_as(spatial_emb)].view(
                2, -1).mean(1).view(2, 1, 1)  # 2 x 1 x 1
    
            s = sigma[mask.expand_as(sigma)].view(n_sigma, -1).mean(1).view(n_sigma, 1, 1)
            s = torch.exp(s*10)  # n_sigma x 1 x 1
    
            dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb - center, 2)*s, 0))
    
            proposal = (dist > 0.5)
            instance_map[proposal] = id
    
        return instance_map

    def cluster(self, prediction, n_sigma=2, threshold=0.9,num_class = 5):

        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]
        
        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2:2+n_sigma]  # n_sigma x h x w
        seed_map = torch.sigmoid(prediction[2+n_sigma:2+n_sigma + num_class])  # num_class x h x w
       
        instance_map = torch.zeros(height, width).byte()
        class_map = torch.zeros(height, width).byte()
        instance_score = []

        count = 1
        for i in range(num_class):
            class_seed = seed_map[i]
            mask = class_seed > 0.5

            if mask.sum() > 128:

                spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)
                sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
                seed_map_masked = class_seed[mask].view(1, -1)

                unclustered = torch.ones(mask.sum()).byte().to(self.device)
                instance_map_masked = torch.zeros(mask.sum()).byte().to(self.device)
                class_map_masked = torch.zeros(mask.sum()).byte().to(self.device)
                

                while(unclustered.sum() > 128):

                    seed = (seed_map_masked * unclustered.float()).argmax().item()
                    seed_score = (seed_map_masked * unclustered.float()).max().item()
                    if seed_score < threshold:
                        break
                    center = spatial_emb_masked[:, seed:seed+1]
                    unclustered[seed] = 0
                    s = torch.exp(sigma_masked[:, seed:seed+1]*10)
                    dist = torch.exp(-1*torch.sum(torch.pow(spatial_emb_masked -
                                                        center, 2)*s, 0, keepdim=True))

                    proposal = (dist > 0.5).squeeze()

                    if proposal.sum() > 128:
                        if unclustered[proposal].sum().float()/proposal.sum().float() > 0.5:
                            instance_map_masked[proposal.squeeze()] = count
                            class_map_masked[proposal.squeeze()] = i+1
                            
                        # instance_mask = torch.zeros(height, width).byte()
                        # instance_mask[mask.squeeze()] = proposal.byte() #this line has been changed
                        # instances.append(
                        #     {'mask': instance_mask.squeeze()*255, 'score': seed_score})
                            instance_score.append(seed_score)
                            count += 1

                    unclustered[proposal] = 0

                instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()
                class_map[mask.squeeze().cpu()] = class_map_masked.cpu()

        return instance_map.cpu(), class_map.cpu(), instance_score

class Logger:

    def __init__(self, keys, title=""):

        self.data = {k: [] for k in keys}
        self.title = title
        self.win = None

        print('created logger with keys:  {}'.format(keys))

    def plot(self, save=False, save_dir=""):

        if self.win is None:
            self.win = plt.subplots()
        fig, ax = self.win
        ax.cla()

        keys = []
        for key in self.data:
            keys.append(key)
            data = self.data[key]
            ax.plot(range(len(data)), data, marker='.')

        ax.legend(keys, loc='upper right')
        ax.set_title(self.title)

        plt.draw()

        if save:
            # save figure
            fig.savefig(os.path.join(save_dir, self.title + '.png'))

            # save data as csv
            df = pd.DataFrame.from_dict(self.data)
            df.to_csv(os.path.join(save_dir, self.title + '.csv'))

    def add(self, key, value):
        assert key in self.data, "Key not in data"
        self.data[key].append(value)



class Visualizer:
    def __init__(self,args):
        self.num_class = args["num_class"]
        # self.n_sigma = args["loss_opts"]["n_sigma"]
        self.color_map = args["color_map"]
        self.cmap = mcolors.ListedColormap([ mcolors.to_rgb(color) for color in plt.get_cmap('tab20').colors])
        
    
        #TODO add number of sigma dependency
    
    def label2colormap(self,label):
        rgbmap = np.zeros((*label.shape[:2], 3), dtype=np.uint8)
        for i, color in self.color_map.items():
            rgbmap[label == i] = color
        return rgbmap
    def instance2color(self,label, colors=None):
        """
        Converts a labelled image represented as integers into a color image.
    
        Args:
            label (numpy array): The labelled image as a 2D array of integers.
        colors (list): (Optional) A list of colors for each label, where each color is a tuple of three values representing the RGB color code.
            If not provided, a list of random distinct colors will be generated.
    
        Returns:
        A color image as a numpy array of shape (height, width, 3).
        """
        num_labels = np.max(label) + 1
        stage = int(num_labels/len(self.cmap.colors))+1
        height, width = label.shape
        color_label = np.zeros((height, width, 3), dtype=np.uint8)
    
        
        for i in range(stage):
            for j in range(len(self.cmap.colors)):
                index = j+(20*i)+1
                color_label[label ==index] = tuple(int(color*255) for color in self.cmap.colors[j])
        return color_label
    def prepare_segment(self,output):
        prediction = torch.zeros_like(output[0])
        for i in range(self.num_class):
            temp =output[i]>0.5
            prediction[temp] =i
        prediction = torch.from_numpy(self.label2colormap(prediction.numpy())).permute(2,0,1)
        return prediction
    def prepare_pred(self,output):
        prediction = torch.zeros_like(output[0])
        for i in range(self.num_class):
            temp =output[i]>0.5
            prediction[temp] =i
        return prediction
    def prepare_internal(self,output):
        vec_x = self.normalize(torch.tanh(output[0])) # h x w
        vec_y = self.normalize(torch.tanh(output[1]))
        sigma_x = self.normalize(output[2]) # h x w
        sigma_y = self.normalize(output[2])
        padd = torch.zeros(sigma_x.shape[:2])
        padd =padd.fill_(0.3)
        
        
        sigma = torch.stack([sigma_x,sigma_y,padd],dim=0) #3 x h x w
        offset = torch.stack([vec_x,vec_y,padd],dim=0) #3 x h x w
        
        prediction = torch.zeros_like(vec_x)
        for i in range(self.num_class):
            temp =torch.sigmoid(output[i+4])>0.7
            prediction[temp] =i+1
        prediction = torch.from_numpy(self.label2colormap(prediction.numpy())).permute(2,0,1)
        offset = offset*255
        sigma = sigma*255

        return offset, sigma, prediction
    def normalize(self,array):
        return (array - array.min())/(array.max()-array.min())
    
    def overlay_image(self,im1,im2):
        img_comb = ((im1*0.5) +(im2*0.5)).astype(np.uint8)
        return img_comb

class Metrics():
    def __init__(self,num_class) -> None:
        self.num_class = num_class+1 # The background class is added by the last 1
        self.CM = np.zeros((self.num_class, self.num_class),dtype=np.int64)
        self.AP = np.zeros((num_class),dtype=np.float32) # without background label
        self.F1 =  None
        self.mAP = None
        self.TP = None
        self.FP = None
        self.FN = None
        self.P = None
        self.R = None
        
        self.count = 0
        
    def _add2CM(self,gt, pred):
        cm = confusion_matrix(gt,pred,labels=list(range(self.num_class)))
        self.CM+=cm
        self._update()
        
    def _add2AP(self,gt, pred_score):
        for i in range(self.num_class - 1): # calculate AP for each class except the background class
            gt_mask = gt == i+1 # i+1 because the background class has label 0
            cl_score = pred_score[i].reshape(-1)
            ap = average_precision_score(gt_mask, cl_score)
            self.AP[i] += ap
        # self.AP = self.AP/self.count
    
    
    def _update(self):
        self.TP = np.diag(self.CM)
        self.FN = self.CM.sum(axis=0) - self.TP
        self.FP = self.CM.sum(axis=1) - self.TP
        self.P = self.TP/(self.TP+self.FP)
        self.R = self.TP/(self.TP+self.FN)
        self.F1 = 2* (self.P*self.R)/(self.P+self.R)
        
        
        
    def add(self,gt_label,pred_label,pred_score):
        if not isinstance(gt_label,np.ndarray):
            print("The given array to Metric class has to be a numpy array \
                     but a different was given. Convert the array to numpy.")
        gt_label = gt_label.reshape(-1)
        pred_label = pred_label.reshape(-1)
        
        self._add2CM(gt_label, pred_label)
        self.count+=1
        self._add2AP(gt_label, pred_score)
        
        # Calculate mAP
        self.mAP = np.mean(self.AP/self.count)
        
        
    def log(self,filename):
        with open(filename, 'w') as f:
            f.write(f"Mean Average Precision (mAP): {self.mAP:.4f}\n")
            f.write("\n")
            f.write("Class-wise Average Precision (AP):\n")
            for i in range(1,self.num_class):
                f.write(f"Class {i}: {self.AP[i-1]/self.count:.4f}\n")
            f.write("\n")
            f.write("Class-wise F1 score (F1):\n")
            for i in range(1,self.num_class):
                f.write(f"Class {i}: {self.F1[i]:.4f}\n")
            f.write("Confusion Matrix (CM):\n")
            for i in range(self.num_class):
                f.write("\t".join([str(int(x)) for x in self.CM[i]]) + "\n")
            f.write("\n")
            f.write(f"Precision (P): {self.P}\n")
            f.write(f"Recall (R): {self.R}\n")