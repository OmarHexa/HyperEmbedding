# https://github.com/juglab/EmbedSeg


import os
import shutil
from glob import glob
import subprocess as sp

import numpy as np
from skimage import io
from tqdm import tqdm
from scipy.ndimage import label
import torch

def make_dirs(data_dir, project_name):
    """
        Makes directories - `train`, `val, `test` and subdirectories under each `images` and `masks`

        Parameters
        ----------
        data_dir: string
            Indicates the path where the `data` lives.
        project_name: string
            Indicates the name of the sub-folder under the location identified by `data_dir`.

        Returns
        -------
    """
    RGB_path_train = os.path.join(data_dir, project_name, 'train', 'images/')
    classmap_path_train = os.path.join(data_dir, project_name, 'train', 'classmaps/')
    instances_path_train = os.path.join(data_dir, project_name, 'train', 'instances/')
    HS_path_train = os.path.join(data_dir, project_name, 'train', 'hs/')
    
    RGB_path_val = os.path.join(data_dir, project_name, 'val', 'images/')
    classmap_path_val = os.path.join(data_dir, project_name, 'val', 'classmaps/')
    instances_path_val = os.path.join(data_dir, project_name, 'val', 'instances/')
    HS_path_val = os.path.join(data_dir, project_name, 'val', 'hs/')

    if not os.path.exists(RGB_path_train):
        os.makedirs(os.path.dirname(RGB_path_train))
        print("Created new directory : {}".format(RGB_path_train))

    if not os.path.exists(instances_path_train):
        os.makedirs(os.path.dirname(instances_path_train))
        print("Created new directory : {}".format(instances_path_train))
    
    if not os.path.exists(classmap_path_train):
        os.makedirs(os.path.dirname(classmap_path_train))
        print("Created new directory : {}".format(classmap_path_train))
    
    if not os.path.exists(HS_path_train):
        os.makedirs(os.path.dirname(HS_path_train))
        print("Created new directory : {}".format(HS_path_train))
        

    if not os.path.exists(RGB_path_val):
        os.makedirs(os.path.dirname(RGB_path_val))
        print("Created new directory : {}".format(RGB_path_val))

    if not os.path.exists(instances_path_val):
        os.makedirs(os.path.dirname(instances_path_val))
        print("Created new directory : {}".format(instances_path_val))
        
    if not os.path.exists(classmap_path_val):
        os.makedirs(os.path.dirname(classmap_path_val))
        print("Created new directory : {}".format(classmap_path_val))
    
    if not os.path.exists(HS_path_val):
        os.makedirs(os.path.dirname(HS_path_val))
        print("Created new directory : {}".format(HS_path_val))


def split_train_val(data_dir, project_name, subset=0.15, by_fraction=True, seed=1000):
    """
        Splits the `train` directory into `val` directory using the partition percentage of `subset`.

        Parameters
        ----------
        data_dir: string
            Indicates the path where the `project` lives.
        project_name: string
            Indicates the name of the sub-folder under the location identified by `data_dir`.
        
        subset: float
            Indicates the fraction of data to be reserved for validation
        seed: integer
            Allows for the same partition to be used in each experiment.
            Change this if you would like to obtain results with different train-val partitions.
        Returns
        -------

    """
    
    #collect directory name of all data
    RGB_dir = os.path.join(data_dir, project_name, 'RGB')
    HS_dir = os.path.join(data_dir,project_name,'HS')
    classmap_dir = os.path.join(data_dir,project_name,'Classmaps')
    instance_dir = os.path.join(data_dir, project_name, 'Instances')
    
    print(classmap_dir)
    
    HS_names = sorted(glob(os.path.join(HS_dir, '*.cue')))
    HSheader_names = sorted(glob(os.path.join(HS_dir, '*.hdr')))
    RGB_names = sorted(glob(os.path.join(RGB_dir, '*.png')))
    classmap_names = sorted(glob(os.path.join(classmap_dir, '*.png')))
    instance_names = sorted(glob(os.path.join(instance_dir, '*.png')))
    print(len(classmap_names), len(instance_names))
    print(len(RGB_names))
    
    assert len(HS_names)==len(RGB_names)
    assert len(classmap_names)==len(instance_names)
    
    print("Number of samples found in '{}' equal to {}".format(RGB_dir, len(RGB_names)))
    
    indices = np.arange(len(RGB_names))
    np.random.seed(seed)
    np.random.shuffle(indices)
    if (by_fraction):
        subset_len = int(subset * len(RGB_names))
    else:
        subset_len = int(subset)
    val_indices = indices[:subset_len]
    trainIndices = indices[subset_len:]
    make_dirs(data_dir=data_dir, project_name=project_name)

    for val_index in val_indices:
        shutil.copy(RGB_names[val_index], os.path.join(
            data_dir, project_name, 'val', 'images'))
        shutil.copy(instance_names[val_index], os.path.join(
            data_dir, project_name, 'val', 'instances'))
        shutil.copy(classmap_names[val_index], os.path.join(
            data_dir, project_name, 'val', 'classmaps'))
        shutil.copy(HS_names[val_index], os.path.join(
            data_dir, project_name, 'val', 'hs'))
        shutil.copy(HSheader_names[val_index], os.path.join(
            data_dir, project_name, 'val', 'hs'))

    for trainIndex in trainIndices:
        shutil.copy(RGB_names[trainIndex], os.path.join(
            data_dir, project_name, 'train', 'images'))
        shutil.copy(instance_names[trainIndex], os.path.join(
            data_dir, project_name, 'train', 'instances'))
        shutil.copy(classmap_names[trainIndex], os.path.join(
            data_dir, project_name, 'train', 'classmaps'))
        shutil.copy(HS_names[trainIndex], os.path.join(
            data_dir, project_name, 'train', 'hs'))
        shutil.copy(HSheader_names[trainIndex], os.path.join(
            data_dir, project_name, 'train', 'hs'))

def convert_rgb2catagory(img,classMap=False):
        color_map = {0:(0,0,0),1: (21, 176, 26), 2:(5, 73, 7),3: (170, 166, 98),4: (229, 0, 0), 5: (140, 0, 15)}
        resolution = (img.shape[0],img.shape[1])
        intImage= np.zeros(resolution,dtype=np.uint8)
        
        if classMap:
            #individual color is assigned a fixed value according to the color map
            for i,color in color_map.items():
                intImage[(img==color).all(axis=2)]=i
        else:
            colors= np.unique(img.reshape(-1, img.shape[-1]), axis=0)
            for i,color in enumerate(colors):
                # every instances is given a unique interger value.
                intImage[(img==color).all(axis=2)]=i
        return intImage
    

def calculate_foreground_weight(data_dir, project_name, train_val_name, background_id=0):
    """

    Parameters
    -------

    data_dir: string
        Name of directory containing data
    project_name: string
        Name of directory containing images and instances
    train_val_name: string
        one of 'train' or 'val'
    mode: string
        one of '2d' or '3d'
    background_id: int, default
        Id which corresponds to the background.

    Returns
    -------
    float:
        Ratio of the number of foreground pixels to the background pixels, averaged over all available label masks

    Note: This is to be used for binary class segmentation.

    """
    instance_names = []
    for name in train_val_name:
        instance_dir = os.path.join(data_dir, project_name, name, 'instances')
        instance_names += sorted(glob(os.path.join(instance_dir, '*.png')))

    statistics = []
    for i in tqdm(range(len(instance_names)), position=0, leave=True):
        ma = io.imread(instance_names[i])
        ma = convert_rgb2catagory(ma)

        y, x = np.where(ma == background_id)
        len_bg = len(y)
        y, x = np.where(ma > background_id)
        len_fg = len(y)
        statistics.append(len_bg / len_fg)

    print("Foreground weight of the `{}` dataset set equal to {:.3f}".format(
        project_name, np.mean(statistics)))
    return np.mean(statistics)

# TODO


def calculate_per_class_weight(data_dir, project_name, train_val_name, background_id=0, num_classes=5):
    
    c=1.10
    classmap_names = []
    for name in train_val_name:
        classmap_dir = os.path.join(data_dir, project_name, name, 'classmaps')
        classmap_names += sorted(glob(os.path.join(classmap_dir, '*.png')))
    class_ids = np.linspace(1,num_classes,num_classes)
    class_statistics = {class_id:[] for class_id in class_ids}
    
    # loop over each image instance
    for i in tqdm(range(len(classmap_names)), position=0, leave=True):
        ma = io.imread(classmap_names[i])
        ma = convert_rgb2catagory(ma,classMap=True)
        num_pix = ma.size

        # loop over each class
        for class_id in class_ids:
            y, _ = np.where(ma==class_id)
            p_class = len(y)/num_pix
            # calculate the weight for the class using the w_class formula
            weight = 1 / np.log(c + p_class)
            
            class_statistics[class_id].append(weight)
    
    # calculate the mean weight for each class
    class_weights = {}
    for class_id in class_ids:
        class_weights[class_id] = np.mean(class_statistics[class_id])
    print("Foreground weight of the `{}` dataset set equal to '{}'".format(project_name,class_weights))
    return class_weights


def calculate_object_size(data_dir, project_name, train_val_name, process_k, background_id=0):
    """
    Calculate the mean object size from the available label masks

    Parameters
    -------

    data_dir: string
        Name of directory storing the data. For example, 'data'
    project_name: string
        Name of directory containing data specific to this project. For example, 'dsb-2018'
    train_val_name: string
        Name of directory containing 'train' and 'val' images and instance masks
    process_k: tuple (int, int)
        Parameter for speeding up the calculation of the object size by considering only a fewer number of images and objects
        The first argument in the tuple is the number of images one should consider
        The second argument in the tuple is the number of objects in the image, one should consider
    background_id: int
        Id which corresponds to the background.

    Returns
    -------
    (float, float, float, float, float, float, float)
    (minimum number of pixels in an object, mean number of pixels in an object, max number of pixels in an object,
    mean number of pixels along the x dimension, standard deviation of number of pixels along the x dimension,
    mean number of pixels along the y dimension, standard deviation of number of pixels along the y dimension

    """

    instance_names = []
    size_list_x = []
    size_list_y = []
    size_list = []
    for name in train_val_name:
        instance_dir = os.path.join(data_dir, project_name, name, 'instances')
        instance_names += sorted(glob(os.path.join(instance_dir, '*.png')))

    if process_k is not None:
        n_images = process_k[0]
    else:
        n_images = len((instance_names))
    for i in tqdm(range(len(instance_names[:n_images])), position=0, leave=True):
        ma = io.imread(instance_names[i])
        ma = convert_rgb2catagory(ma)

        # read unique id of each instances.
        ids = np.unique(ma)
        ids = ids[ids != background_id]
        for id in ids:
            y, x = np.where(ma == id)
            size_list_x.append(np.max(x) - np.min(x))
            size_list_y.append(np.max(y) - np.min(y))
            size_list.append(len(x))

    print("Minimum object size of the `{}` dataset is equal to {}".format(
        project_name, np.min(size_list)))
    print("Mean object size of the `{}` dataset is equal to {}".format(
        project_name, np.mean(size_list)))
    print("Maximum object size of the `{}` dataset is equal to {}".format(
        project_name, np.max(size_list)))
    print("Average object size of the `{}` dataset along `x` is equal to {:.3f}".format(project_name,
                                                                                        np.mean(size_list_x)))
    print("Std. dev object size of the `{}` dataset along `x` is equal to {:.3f}".format(project_name,
                                                                                         np.std(size_list_x)))
    print("Average object size of the `{}` dataset along `y` is equal to {:.3f}".format(project_name,
                                                                                        np.mean(size_list_y)))
    print("Std. dev object size of the `{}` dataset along `y` is equal to {:.3f}".format(project_name,
                                                                                         np.std(size_list_y)))

    return np.min(size_list).astype(np.float), np.mean(size_list).astype(np.float), np.max(size_list).astype(
        np.float), np.mean(size_list_y).astype(np.float), np.mean(size_list_x).astype(np.float), np.std(
        size_list_y).astype(np.float), np.std(size_list_x).astype(np.float)


def get_gpu_memory():
    """
        Identifies the max memory on the operating GPU
        https://stackoverflow.com/questions/59567226/how-to-programmatically-determine-available-gpu-memory-with-tensorflow/59571639#59571639

    """

    command = "nvidia-smi --query-gpu=memory.total --format=csv"
    memory_total_info = sp.check_output(
        command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_total_values = [int(x.split()[0])
                           for i, x in enumerate(memory_total_info)]
    return memory_total_values


def round_up_8(x):
    """
    Rounds up `x` to the next nearest multiple of 8
    for e.g. round_up_8(7) = 8

    Parameters
    -------

    x: int

    Returns
    -------
    int
    Next nearest multiple to 8

    """
    return (x.astype(int) + 7) & (-8)


def calculate_max_eval_image_size(data_dir, project_name, train_val_name, scale_factor=4.0):
    """
        Identifies the tile size to be used during training and evaluation.
        We look for the largest evaluation image.
        If the entire image could fit on the available GPU memory (based on an empirical idea), then  the tile size is set equal to those dimensions.
        If the dimensions are larger, then a smaller tile size as specified by the GPU memory is used.

        Parameters
        -------

        data_dir: string
            Name of directory storing the data. For example, 'data'
        project_name: string
            Name of directory containing data specific to this project. For example, 'electron-microscopy'
        train_val_name: string
            Name of directory containing 'train' and 'val' images and instance masks
        scale_factor: float, default
            Used to evaluate the maximum GPU memory which shall be used during evaluation

        Returns
        -------

        (int, int)
        (tile size along y, tile size along x)

        """

    image_names = []
    size_y_list = []
    size_x_list = []

    for name in train_val_name:
        instance_dir = os.path.join(data_dir, project_name, name, 'images')
        image_names += sorted(glob(os.path.join(instance_dir, '*.png')))

    for i in tqdm(range(len(image_names)), position=0, leave=True):
        im = io.imread(image_names[i])
        size_y_list.append(im.shape[0])
        size_x_list.append(im.shape[1])

    max_y = np.max(size_y_list)
    max_x = np.max(size_x_list)
    max_y, max_x = round_up_8(max_y), round_up_8(max_x)
    max_x_y = np.maximum(max_x, max_y)

    # Note: get_gpu_memory returns a list
    total_mem = get_gpu_memory()[0] * 1e6
    tile_size_temp = np.asarray(
        (total_mem / (2 * 4 * scale_factor)) ** (1 / 2))  # 2D

    if tile_size_temp < max_x_y:
        max_x, max_y = round_up_8(tile_size_temp), round_up_8(tile_size_temp)
    else:
        max_x, max_y = max_x_y, max_x_y
    print("Tile size of the `{}` dataset set equal to ({}, {})".format(
        project_name, max_y, max_x))
    return max_y.astype(np.float), max_x.astype(np.float)


def calculate_avg_background_intensity(data_dir, project_name, train_val_name, background_id=0):
    """
    Calculates the average intensity in the regions of the raw image which corresponds to the background label

    Parameters
    -------

    data_dir: str
        Path to directory containing all data
    project_name: str
        Path to directory containing project-specific images and instances
    train_val_name: str
        One of 'train' or 'val'
    background_id: int
         Label corresponding to the background

    Returns
    -------
        float
        Average background intensity of the dataset
    """

    instance_names = []
    image_names = []
    for name in train_val_name:
        instance_dir = os.path.join(data_dir, project_name, name, 'instances')
        image_dir = os.path.join(data_dir, project_name, name, 'images')
        instance_names += sorted(glob(os.path.join(instance_dir, '*.png')))
        image_names += sorted(glob(os.path.join(image_dir, '*.png')))
    statistics = []

    for i in tqdm(range(len(instance_names)), position=0, leave=True):
        ma = io.imread(instance_names[i])
        ma = convert_rgb2catagory(ma)
        
        bg_mask = ma == background_id
        im = io.imread(image_names[i])
        if im.ndim == ma.ndim:
            statistics.append(np.average(im[bg_mask]))
        elif im.ndim == ma.ndim + 1:  # multi-channel image
            statistics.append(np.average(im[bg_mask, :], axis=1))

    print("Average background intensity of the `{}` dataset set equal to {}".format(project_name,
                                                                                        np.mean(statistics, axis=0)))
    return np.mean(statistics, axis=0)

def calculate_per_class_instance(data_dir, project_name, train_val_name, background_id=0, num_class=5):
    classmap_names = []
    instance_names = []
    classmap_dir = os.path.join(data_dir, project_name,'Classmaps')
    instance_dir = os.path.join(data_dir, project_name,'Instances')
    
    classmap_names += sorted(glob(os.path.join(classmap_dir, '*.png')))
    instance_names += sorted(glob(os.path.join(instance_dir, '*.png')))
    
    class_counts = {id: 0 for id in range(1,num_class+1)}
    
    for i in tqdm(range(len(classmap_names)), position=0, leave=True):
        clmp = io.imread(classmap_names[i])
        clmp = convert_rgb2catagory(clmp,classMap=True)
        inst = io.imread(instance_names[i])
        inst = convert_rgb2catagory(inst)
        for cl in range(1,num_class+1):
            count = np.unique(inst[clmp==cl]).size
            class_counts[cl] += count
    
    print('Number of instances per class\n {}'.format(class_counts))

def get_data_properties(data_dir, project_name, train_val_name, process_k=None, background_id=0):
    """

    Parameters
    -------

    data_dir: string
            Path to directory containing all data
    project_name: string
            Path to directory containing project-specific images and instances
    train_val_name: string
            One of 'train' or 'val'
    test_name: string
            Name of test directory.
    mode: string
            One of '2d', '3d', '3d_sliced', '3d_ilp'
    one_hot: boolean
            set to True, if instances are encoded in a one-hot fashion
    process_k (int, int)
            first `int` argument in tuple specifies number of images which must be processed
            second `int` argument in tuple specifies number of ids which must be processed
    anisotropy_factor: float
            Ratio of the real-world size of the z-dimension to the x or y dimension in the raw images
            If the image is down-sampled along the z-dimension, then `anisotropy_factor` is greater than 1.0
    background_id: int
            Label id corresponding to the background

    Returns
    -------
    data_properties_dir: dictionary
            keys include `foreground_weight`, `min_object_size`, `project_name`, `avg_background_intensity` etc

    """
    data_properties_dir = {}
    data_properties_dir['inctance_per_class'] =calculate_per_class_instance(data_dir=data_dir,project_name=project_name,train_val_name=train_val_name,background_id=background_id)
    data_properties_dir['class_weight'] = calculate_per_class_weight(data_dir, project_name, train_val_name,
                                                                           background_id=background_id)
    data_properties_dir['min_object_size'], data_properties_dir['mean_object_size'], data_properties_dir[
        'max_object_size'], \
    data_properties_dir['avg_object_size_y'], data_properties_dir['avg_object_size_x'], \
    data_properties_dir['stdev_object_size_y'], data_properties_dir['stdev_object_size_x'] = \
        calculate_object_size(data_dir, project_name, train_val_name, process_k,
                                                       background_id=background_id)
        
    if torch.cuda.is_available():
        data_properties_dir['n_y'], data_properties_dir['n_x'] = calculate_max_eval_image_size(
            data_dir=data_dir, project_name=project_name, train_val_name=['train','val'])

#     data_properties_dir['avg_background_intensity'] = calculate_avg_background_intensity(data_dir, project_name,
#                                                                                          train_val_name,
#                                                                                         background_id=background_id)
    data_properties_dir['project_name'] = project_name
    return data_properties_dir



if __name__=="__main__":
    data_dir = '../Datasets'
    project_name = '20220719'
    if os.path.exists(os.path.join(data_dir,project_name)):
        print("`{}` is choosen as datapath".format(os.path.join(data_dir,project_name)))
    data_properties_dir = get_data_properties(data_dir, project_name, train_val_name=['train', 'val'])