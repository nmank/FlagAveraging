from numpy import *
import cv2
import numpy as np
from os import listdir, mkdir
from os.path import isdir
from matplotlib import pyplot as plt
import sys
sys.path.append('../scripts')
import center_algorithms as ca
from sklearn.decomposition import PCA




'''
Find the ratio that we are downsizing images by
    download from https://www.crcv.ucf.edu/data/UCF_YouTube_Action.php 
    by clicking YouTube Action Data Set 

Inputs:
    f_name: path to file from the dataset that you want to load

Outputs:
    ratio: the proportion of pixels in the new image
'''
def load_video(f_name: str) -> float:
    gr_dims = 0

    cap = cv2.VideoCapture(f_name)
    ret = True
    frames = []
    while ret:
        ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
        if ret:
            
            #convert to greyscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            width, height = img_gray.shape
            r = 25.0 / height
            dim = (25, int(width * r))
            # perform the actual resizing of the image
            img_gray_resized = cv2.resize(img_gray, dim, interpolation=cv2.INTER_AREA)
            if gr_dims == 0:
                gr_dims = dim[0]*dim[1]
            
            frames.append(img_gray_resized)
  
    video = np.stack(frames, axis=0) 

    size = video.shape[0]


    video = video.reshape(size,gr_dims)

    if video.shape[0] >= 10:
        #compute the ratio of pixels = pixels in new frame / pixels in old frame
        ratio = img_gray_resized.shape[0]*img_gray_resized.shape[1]/(width*height)  
    else:
        print(video.shape)

    if (size > 10) or video.shape[0] >= 10:
        return ratio
    else:
        return np.nan

if __name__ == '__main__':
    '''
    Caution!
        This will only work on linux systems due to file paths
        
    Before running:
        Download the UFC YouTube Action dataset
    '''

    #path to the data folder (should contain the folder: action_youtube_naudio)
    base_path = './'
    data = []
    labels = []
    ratios = []
    for label in listdir(base_path+'action_youtube_naudio/'):
        if isdir(base_path+'action_youtube_naudio/'+label):
            print('class '+label)
            for sample in listdir(base_path+'action_youtube_naudio/'+label):
                if isdir(base_path+'action_youtube_naudio/'+label+'/'+sample):
                    if 'Annotation' not in sample:
                        ratio = load_video(base_path+'action_youtube_naudio/'+label+'/'+sample+'/'+sample+'_01.avi')
                        ratios.append(ratio)
    #print the mean and standard deviation of the pixels
    ratios = np.array(ratios)
    ratios = ratios[~np.isnan(ratios)]
    mn = np.round(np.mean(ratios),2)
    std = np.round(np.std(ratios),2)
    print(f'ratio of pixels: {mn} +/- {std}')
