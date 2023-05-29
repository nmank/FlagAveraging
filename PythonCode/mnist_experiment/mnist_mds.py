import sys
sys.path.append('../scripts')

import fl_algorithms as fla
import center_algorithms as ca

from matplotlib import pyplot as plt

import numpy as np

import pandas as pd

from sklearn.manifold import MDS

from mnist_nn_test import load_mnist_data, generate_subspace_data

def chordal_dist_mat(data: list, flag_type: list) -> np.array:
    n_pts = len(data)

    distances = np.zeros((n_pts, n_pts))
    for i in range(n_pts):
        for j in range(i+1, n_pts, 1):
            distances[i,j] = fla.chordal_dist(data[i], data[j], flag_type = flag_type)
            distances[j,i] = distances[i,j].copy()
    return distances

if __name__ == '__main__':

    digit1 = 7
    digit2 = 6
    num_samples1 = 20

    num_samples2s = [0,4,8,12]

    flag_type = [1,2] 

    qr_svd = 'qr'

    n_its = 10

    k = flag_type[-1]
    n_exp = len(num_samples2s)

    flag_mean = []
    flag_median = []
    real_flag_mean = []
    real_flag_median = []
    euclidean_means = []

    for num_samples2 in num_samples2s:

        print(f'starting num_samples2 = {num_samples2}')

        data_matrix1 = load_mnist_data(digit1, num_samples1, dset='train')[0]
        data_matrix2 = load_mnist_data(digit2, num_samples2, dset='train')[0]

        gr_list = []

        if num_samples1 > 0:
            subspaces1 = generate_subspace_data(data_matrix1.T, n_nbrs = k, qr_svd = qr_svd)
        else:
            subspaces1 = []
            
        if num_samples2 > 0:
            subspaces2 = generate_subspace_data(data_matrix2.T, n_nbrs = k, qr_svd = qr_svd)
        else:
            subspaces2 = []

        data = subspaces1 + subspaces2
        n_pts = len(data)

        stacked_data = np.stack(data, axis = 2)

        flag_mean.append(ca.flag_mean(data, k))

        flag_median.append(ca.irls_flag(data, k, n_its, 'sine', opt_err = 'sine')[0])

        real_flag_mean.append(fla.flag_mean(stacked_data,  flag_type = flag_type))

        real_flag_median.append(fla.flag_median(stacked_data,  flag_type = flag_type, max_iters = 100))

#        euclidean_mean = np.mean(stacked_data, axis = 2)
#        euclidean_means.append(np.linalg.qr(euclidean_mean)[0][:,:flag_type[-1]])


#    all_data = data + flag_mean + flag_median + real_flag_mean + real_flag_median + euclidean_means
    all_data = data + flag_mean + flag_median + real_flag_mean + real_flag_median

    D = chordal_dist_mat(all_data, flag_type)

    embedding = MDS(n_components=2, dissimilarity='precomputed')
    embedded_data = embedding.fit_transform(D)

    plt.rcParams.update({'font.size': 14})    
    plt.scatter(embedded_data[:num_samples1,0], embedded_data[:num_samples1,1],  alpha = .6, color = 'k', label = '6s')
    plt.scatter(embedded_data[num_samples1:n_pts,0], embedded_data[num_samples1:n_pts,1],  alpha = .6, color = 'tab:brown', marker = 's', label = '7s')
    plt.plot(embedded_data[n_pts:n_pts+n_exp,0], embedded_data[n_pts:n_pts+n_exp,1], marker = 'x',label = 'GR-mean')
    plt.plot(embedded_data[n_pts+n_exp:n_pts+2*n_exp,0], embedded_data[n_pts+n_exp:n_pts+2*n_exp,1], marker = 'x',label = 'GR-median')
    plt.plot(embedded_data[n_pts+2*n_exp:n_pts+3*n_exp,0], embedded_data[n_pts+2*n_exp:n_pts+3*n_exp,1], marker = 'x',label = 'FL-mean')
    plt.plot(embedded_data[n_pts+3*n_exp:n_pts+4*n_exp,0], embedded_data[n_pts+3*n_exp:n_pts+4*n_exp,1], marker = 'x',label = 'FL-median')
    #plt.plot(embedded_data[n_pts+4*n_exp:,0], embedded_data[n_pts+4*n_exp:,1], marker = 'x',label = 'Euclidean mean')
    plt.legend()
    plt.xlabel('MDS 1')
    plt.ylabel('MDS 2')
    plt.tight_layout()
    plt.savefig('mds_plot.pdf')
    
    












