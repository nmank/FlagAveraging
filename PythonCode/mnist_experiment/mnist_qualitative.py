import numpy as np
import sys
sys.path.append('../scripts')
import center_algorithms as ca
import fl_algorithms as fla
import matplotlib.pyplot as plt
import torch
import pandas as pd
import seaborn as sns
from sklearn.manifold import MDS
import time
from sklearn.neighbors import NearestNeighbors
from matplotlib import gridspec

def generate_subspace_data(X: np.array, n_nbrs: int, qr_svd: str = 'qr') -> list:

    subspaces = []

    if n_nbrs > 0:
        neigh = NearestNeighbors(n_neighbors = n_nbrs, metric = 'cosine')
        neigh.fit(X)
        nns = neigh.kneighbors(X, return_distance=False)
        
        
        for nn in nns:
            if qr_svd == 'qr':
                subspace = np.linalg.qr(X[nn,:].T)[0][:,:n_nbrs]
            elif qr_svd == 'svd':
                subspace = np.linalg.svd(X[nn,:].T)[0][:,:n_nbrs]
            else:
                print('qr_svd not recognized')

            subspaces.append(subspace)
    else:
        subspace = np.linalg.svd(X.T)[0][:,:X.shape[0]]
        subspaces.append(subspace)
    
    return subspaces


def load_mnist_data(digits, sample_size=100, dset='train', shuffle = False, data_path = './data/MNIST/'):
    '''
    Shannon Stiverson's dataloader

    Inputs:
        digits- list or int for digits from MNIST to be samples
        sample_size - number of samples of that digit
        dset - 'train' or 'test' for sampling from the training or the test datasets
    Outputs:
        out_datamat - a numpy array of dimensions (784 x sample_size)
        datamat_labels - a list of labels of the sampled points
    '''

    if type(digits) == int:
        digits = [digits]

    if type(sample_size) == int:
        sample_size = [sample_size]*len(digits)

    if len(sample_size) != len(digits):
        print('Incorrect number of sample sizes given.')
        return [], []

    return_data = []
    return_labels = []

    for i in range(len(digits)):
        digit = digits[i]
        size = sample_size[i]
        digit_data = np.loadtxt(data_path+'%s%i.csv' % (dset, digit), delimiter=',')
        if digit_data.shape[1] < size:
            print('Sample number for digit %i reduced to %i' % (digit, digit_data.shape[1]))
            return_data.append(digit_data)
            return_labels.append([digit]*digit_data.shape[1])
        else:
            if shuffle:
                idx = np.random.choice(np.arange(digit_data.shape[1]), size)
            else:
                idx = np.arange(size)
            return_data.append(digit_data[idx, :])
            return_labels.append([digit]*size)

    datamat = return_data[0]
    datamat_labels = return_labels[0]

    if len(digits) > 1:
        for i in range(1, len(digits)):
            datamat = np.vstack((datamat, return_data[i]))
            datamat_labels = np.hstack((datamat_labels, return_labels[i]))

    out_datamat = datamat.T
    
    return out_datamat, datamat_labels

def run_mnist_test(k: int, digit1: int, digit2: int, 
                   num_samples1: int, num_samples2: int,
                   n_its: int, seed: int, 
                   qr_svd: str = 'qr'):
    '''
    calculates Flag Mean, Maximum Cosine and Sine Median for a dataset

    Inputs:
        k- int for Gr(k,784)
        digit1- int for first digit
        digit2- int for second digit
        num_samples1- int for number of samples of digit1s
        num_samples2- int for number of samples of digit2s
        n_its- integer for the number of iterations of FlagIRLS
        seed- int for numpy random initialization
    Outputs:
        flagmean- numpy array that represents Flag Mean of gr_list
        sin_median- numpy array that represents Sine Median of gr_list
        max_cosine- numpy array that represents Max Cosine of gr_list
        gr_list- dataset as subspaces in Gr(k,748)
    '''
    
    data_matrix1 = load_mnist_data(digit1, k*num_samples1, dset='train')[0]
    data_matrix2 = load_mnist_data(digit2, k*num_samples2, dset='train')[0]

    gr_list = []

    if num_samples1 > 0:
        subspaces1 = generate_subspace_data(data_matrix1.T, n_nbrs = k, qr_svd = qr_svd)
    else:
        subspaces1 = []
        
    if num_samples2 > 0:
        subspaces2 = generate_subspace_data(data_matrix2.T, n_nbrs = k, qr_svd = qr_svd)
    else:
        subspaces2 = []

    gr_list = subspaces1 + subspaces2

    flag_mean = ca.flag_mean(gr_list, k)

    flag_median = ca.irls_flag(gr_list, k, n_its, 'sine', opt_err = 'sine', seed = seed)[0]

    real_flag_mean = fla.flag_mean(np.stack(gr_list, axis = 2),  flag_type = [k//2,k], oriented = True)

    real_flag_median = fla.flag_median(np.stack(gr_list, axis = 2),  flag_type = [k//2,k], oriented = True)
    
    return flag_mean, flag_median, real_flag_mean, real_flag_median

def plot_prototypes_new(flag_medians: list, flagmeans: list, real_flagmedians: list, real_flagmeans: list,
                     added_1s: int, k: int, qr_svd: str = 'qr'):
    print(len(flagmeans))
    print(added_1s)
    plt.figure('gr_mean')
    plt.imshow(flagmeans[0][:,0].reshape(28,28), cmap = 'gray')
    plt.axis('off')
    plt.savefig(f"gr_mean_{qr_svd}_{added_1s}.png", bbox_inches='tight')

    plt.figure('gr_median')
    plt.imshow(flag_medians[0][:,0].reshape(28,28), cmap = 'gray')
    plt.axis('off')
    plt.savefig(f"gr_median_{qr_svd}_{added_1s}.png", bbox_inches='tight')

    plt.figure('fl_mean')
    plt.imshow(real_flagmeans[0][:,0].reshape(28,28), cmap = 'gray')
    plt.axis('off')
    plt.savefig(f"fl_mean_{qr_svd}_{added_1s}.png", bbox_inches='tight')

    plt.figure('fl_median')
    plt.imshow(real_flagmedians[0][:,0].reshape(28,28), cmap = 'gray')
    plt.axis('off')
    plt.savefig(f"fl_median_{qr_svd}_{added_1s}.png", bbox_inches='tight')

if __name__ == '__main__':

    k = 2 #Gr(k,n)
    digit1 = 6
    digit2 = 7
    num_samples1 = 20 #number of samples of the 6 digit
    num_samples2 = 10 #number of samples of the 7 digit
    n_its = 20 #number of iterations for FlagIRLS
    seed = 1 #for initialization
    incr = 1 #samples 5s from 0 to num_samples2 in increments of incr

    for qr_or_svd in ['qr']:


        if num_samples1 % incr != 0:
            print('incr does not divide num_samples2 evenly!')
        else:
            flagmeans = []
            flag_medians = []
            real_flagmeans = []
            real_flagmedians = []
            l2_meds = []
            n2s  = []
            n5s = []
            out = run_mnist_test(k, digit1, digit2,
                                num_samples1, num_samples2,
                                n_its,seed, qr_svd = qr_or_svd)
            
            #make sure the image is oriented correctly
            #eg positive pixel values
            for i in range(4):
                for j in [0,1]:
                    sum_pt = np.sum(out[i][:,j])
                    if sum_pt <0:
                        out[i][:,j] = -out[i][:,j]      

            flagmeans.append(out[0])
            flag_medians.append(out[1])
            real_flagmeans.append(out[2])
            real_flagmedians.append(out[3])
            n2s.append(num_samples1)
            n5s.append(num_samples2)
            print(f'number of added 1s: {num_samples2}')

            plot_prototypes_new(flag_medians, flagmeans, real_flagmedians, real_flagmeans,
                        added_1s = num_samples2,  k = k, qr_svd = qr_or_svd)
                







