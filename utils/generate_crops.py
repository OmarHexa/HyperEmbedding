"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import glob
import os
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm
from skimage import io
import spectral as sp
from skimage.color import rgba2rgb
import sys


def hs_to_rgb(hs,red_band=72,green_band=50,blue_band=30):
        """
        Generates RGB image from the Hyperspectral image using red, green and blue channels.
        Parameters:
        ----------
        hsi: Hyperspectral image to convert
        red_band: red band number (Depends on the hyperspectral camera). Default value taken from ENVI
        green_band: Green band number
        blue_band: Blue band number 

        Return:
        -------
        HS_to_RGB: The sliced rgb image from the hyperspectral image given 
        """
#         red_im = hs[:,:,red_band]
#         green_im = hs[:,:,green_band]
#         blue_im = hs[:,:,blue_band]
#         #Stack the channel in thrid dimension 
#         HS_to_RGB = np.dstack((red_im,green_im,blue_im))
        HS_to_RGB = hs[:,:,(160,100,45)]
        # Normalize the value and convert to uint8 image
        HS_to_RGB = cv2.normalize(HS_to_RGB, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        alpha = 2.5
        beta =10
        HS_to_RGB = cv2.convertScaleAbs(HS_to_RGB,alpha=alpha,beta=beta)
        return HS_to_RGB



def filter_matches(matches, keypoints_rgb, keypoints_hs):
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    dst_pts = np.empty((len(good_matches), 1, 2))
    src_pts = np.empty((len(good_matches), 1, 2))
    for i in range(len(good_matches)):
        dst_pts[i] = keypoints_rgb[good_matches[i].queryIdx].pt
        src_pts[i] = keypoints_hs[good_matches[i].trainIdx].pt
    return dst_pts, src_pts, len(good_matches)


    
    

def get_affine_matrix(image, spectral_image):
        """
        Calculates the affine transformation matrix of the hyperspectral image to the rgb image.

        Parameters:
        ----------

        Return:
        -------
        M: affine transformation matrix.
        """
        
        # Intiate sift object
        sift = cv2.SIFT_create(2000,sigma=3.0, contrastThreshold=0.01)
        # Detect SIFT features in the rgb image
        keypoints_rgb, descriptors_rgb = sift.detectAndCompute(image, None)


        # Detect SIFT features in the hyperspectral image
        keypoints_hs, descriptors_hs = sift.detectAndCompute(spectral_image, None)
        
        
        

        # Match the SIFT features between the RGB and hyperspectral images using the FLANN algorithm
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors_rgb, descriptors_hs, k=2)
        
        # Filter the matches to only include those with a low enough ratio of best to second-best match
        dst_pts, src_pts, good_matches_count = filter_matches(matches, keypoints_rgb, keypoints_hs)
    
        # Calculate the transformation matrix using the good matches
        MIN_MATCH_COUNT = 10
        if good_matches_count>MIN_MATCH_COUNT: 
            # We want to go from hs(src) to rgb(dst) scale and orientation
            M, _ = cv2.estimateAffine2D(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold= 10)
        else:
            print("Error: Not enough good matches found for coregistration.")
            M = None
        return M
def coregister_hs(spectral,affine_matrix,shape):
        """
        Co-register the hyperspectral images by transforming every bands using the transformation matrix.

        Parameters:
        ----------

        Returns:
        --------
        hs_registered: transformed hyperspectral image into rgb resolution.

        Note: The output resolution of the array is changed to rgb resolution to match with the ground truth segmentation.
        """
        hs_registered = np.zeros((shape[0],shape[1],spectral.shape[-1]))
        for i in range(spectral.shape[-1]):
            hs_registered[..., i] = cv2.warpAffine(spectral[..., i], affine_matrix, shape)
        return hs_registered

def convert_rgb2category(img,classMap=False):
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
    

def process_with_pool(tup):
    """
    Intput: Image names, Instance names, Classmap names, Hyperspectral cube & header, crop size
    
    """
#     assert len(tup)==6
    print("*******strating****")
    # image_name,instance_name,classmap_name,hs_name,header_name,crop_size,crops_dir,data_subset = tup
    image_name,instance_name,classmap_name,hs_name,crop_size,crops_dir,data_subset = tup
    background_id=0

    
    image_path = os.path.join(crops_dir, data_subset, 'images/')
    classmap_path = os.path.join(crops_dir, data_subset, 'classmaps/')
    instance_path = os.path.join(crops_dir, data_subset, 'instances/')
    hs_path = os.path.join(crops_dir, data_subset, 'hs/')

  
    try:  # can't use 'exists' because of threads
        os.makedirs(os.path.dirname(image_path))
        os.makedirs(os.path.dirname(instance_path))
        os.makedirs(os.path.dirname(classmap_path))
        os.makedirs(os.path.dirname(hs_path))
        
        
    except FileExistsError:
        pass

    image = io.imread(image_name).astype(np.uint8)

    
    instance = io.imread(instance_name).astype(np.uint8)

    classmap = io.imread(classmap_name).astype(np.uint8)
    
    h, w, c = image.shape
    
    # spectral_cube = sp.envi.open(header_name, hs_name).load()
    # #convert hyperspectral to rgb image
    # spectral_image = hs_to_rgb(spectral_cube)
    # # calculate the affine transformation 
    # M = get_affine_matrix(image,spectral_image)
    # # register the hyperspectral image
    # spectral_reg = coregister_hs(spectral_cube,M,(h,w))
    spectral_reg = np.load(hs_name)
    
    

    instance_np = np.array(instance, copy=False,dtype=np.uint8)
    instance_np = convert_rgb2category(instance_np)
    object_mask = instance_np > background_id
    
    ids = np.unique(instance_np[object_mask])
    ids = ids[ids!= 0]

    # loop over instances
    for j, id in enumerate(ids):
        
        y, x = np.where(instance_np == id)
        ym, xm = np.mean(y), np.mean(x)
        
        jj = int(np.clip(ym-crop_size/2, 0, h-crop_size))
        ii = int(np.clip(xm-crop_size/2, 0, w-crop_size))

        if (image[jj:jj + crop_size, ii:ii + crop_size, :].shape == (crop_size, crop_size, c)):
        
            im_crop = image[jj:jj + crop_size, ii:ii + crop_size, :]
            instance_crop = instance[jj:jj + crop_size, ii:ii + crop_size]
            classmap_crop = classmap[jj:jj + crop_size, ii:ii + crop_size]
            
            hs_crop = spectral_reg[jj:jj + crop_size, ii:ii + crop_size,:]
            
            
            
            np.save(hs_path+os.path.basename(hs_name)[:-4] + "_{:03d}".format(j),hs_crop)
            
            io.imsave(image_path + os.path.basename(hs_name)[:-4] + "_{:03d}.png".format(j), 
                      im_crop, check_contrast=False)
            io.imsave(classmap_path + os.path.basename(hs_name)[:-4] + "_{:03d}.png".format(j),
                      classmap_crop, check_contrast=False)

            io.imsave(instance_path + os.path.basename(hs_name)[:-4] + "_{:03d}.png".format(j),
                      instance_crop, check_contrast=False)


