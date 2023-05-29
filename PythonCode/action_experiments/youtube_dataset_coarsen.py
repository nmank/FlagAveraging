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
Load video from UFC YouTube Action dataset as a point on FL(1,2,...,10; gr_dims)
    download from https://www.crcv.ucf.edu/data/UCF_YouTube_Action.php 
    by clicking YouTube Action Data Set 

Inputs:
    f_name: path to file from the dataset that you want to load
Outputs:
    gr_point: a truncated unitary matrix of size (gr_dims x 10)
    gr_dims: the number of rows of gr_point
'''
def load_video(f_name: str) -> tuple:
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
            gr_dims = dim[0]*dim[1]

            frames.append(img_gray_resized)
  
    video = np.stack(frames, axis=0) 

    size = video.shape[0]

    if video.size/size != 0:
        return [], 0

    video = video.reshape(size,gr_dims)

    if video.shape[0] >= 10:
        #make a representative for the video if it has enough frames
        gr_point = np.linalg.qr(video.T)[0][:,:10]
    
    else:
        print(video.shape)


    if (size > 10) or video.shape[0] >= 10:
        return gr_point, gr_dims
    else:
        return [], gr_dims

if __name__ == '__main__':
    '''
    Caution!
        This will only work on linux systems due to file paths
        
    Before running:
        1) Download the UFC YouTube Action dataset
        2) Make the directory <base_path>/action_youtube_gr_small
    '''


    #path to the data folder (should contain the folder: action_youtube_naudio)
    base_path = './'


    data = []
    labels = []
    #loop through each action
    for label in listdir(base_path+'action_youtube_naudio/'):
        if isdir(base_path+'action_youtube_naudio/'+label):
            print('class '+label)

            #loop through each action class
            for sample in listdir(base_path+'action_youtube_naudio/'+label):
                if isdir(base_path+'action_youtube_naudio/'+label+'/'+sample):
                    if 'Annotation' not in sample:
                        
                        point, gr_dims = load_video(base_path+'action_youtube_naudio/'+label+'/'+sample+'/'+sample+'_01.avi')
                        
                        #only take videos that sucessfully were loaded 
                        if len(point)> 1:
                            data.append(point)
                            labels.append(label)

    
    #make directories for the videos

    #specifically: /home/nmank/YouTubeData/action_youtube_gr_small
    for c in np.unique(labels):
        mkdir(base_path+'action_youtube_gr_small/'+c)
    
    #save .npy files of the video representatives
    i=1
    for x, l in zip(data, labels):
        np.save(base_path+'action_youtube_gr_small/'+l+'/'+l+'_'+str(i)+'.npy', x)
        i+=1
