import sys
sys.path.append('../scripts')

import fl_algorithms as fla
import center_algorithms as ca

from matplotlib import pyplot as plt

import numpy as np

import pandas as pd

from sklearn.manifold import MDS

def generate_flag_data_noisy(n_pts: int, n: int, flag_type: list, noise: float, seed: int = 1) -> list:
    np.random.seed(seed)

    k = flag_type[-1]
    center_pt = np.linalg.qr(np.random.rand(n, k)-.5)[0][:,:k]

    data = []
    for i in range(n_pts):
        rand_mat = center_pt + noise*(np.random.rand(n, k)-.5)
        data.append(np.linalg.qr(rand_mat)[0][:,:k])

    return data, center_pt

def chordal_dist_mat(data: list, flag_type: list) -> np.array:
    n_pts = len(data)

    distances = np.zeros((n_pts, n_pts))
    for i in range(n_pts):
        for j in range(i+1, n_pts, 1):
            distances[i,j] = fla.chordal_dist(data[i], data[j], flag_type = flag_type)
            distances[j,i] = distances[i,j].copy()
    return distances

def generate_flag_data_outliers(n_inliers: int, n_outliers: int, flag_type: list, seed: int = 2) -> list:
    np.random.seed(seed)

    k = flag_type[-1]
    center_pt = np.linalg.qr(np.random.rand(n, k)-.5)[0][:,:k]

    data = []
    for i in range(n_inliers):
        rand_mat = center_pt + 0.01*(np.random.rand(n, k)-.5)
        data.append(np.linalg.qr(rand_mat)[0][:,:k])
    for i in range(n_outliers):
        rand_mat = center_pt + .5*(np.random.rand(n, k)-.5)
        data.append(np.linalg.qr(rand_mat)[0][:,:k])

    return data, center_pt

if __name__ == '__main__':

    n_pts = 100
    n = 10
    flag_type = [1,3] 

    n_its = 10

    k = flag_type[-1]

    noise = .2

    data, center_pt = generate_flag_data_outliers(70, 30, flag_type)

    stacked_data = np.stack(data, axis = 2)

    flag_mean = ca.flag_mean(data, k)

    flag_median = ca.irls_flag(data, k, n_its, 'sine', opt_err = 'sine')[0]

    real_flag_mean = fla.flag_mean(stacked_data,  flag_type = flag_type)    

    real_flag_median = fla.flag_median(stacked_data,  flag_type = flag_type, max_iters = 100)

    euclidean_mean = np.mean(stacked_data, axis = 2)
    euclidean_mean = np.linalg.qr(euclidean_mean)[0][:,:flag_type[-1]]


    all_data = data + [flag_mean] + [flag_median] + [real_flag_mean] + [real_flag_median] + [euclidean_mean]

    D = chordal_dist_mat(all_data, flag_type)

    embedding = MDS(n_components=2, dissimilarity='precomputed')
    embedded_data = embedding.fit_transform(D)

    plt.figure('embedding')

    plt.scatter(embedded_data[:n_pts,0], embedded_data[:n_pts,1],  alpha = .6, color = 'k', label = 'Data')
    plt.scatter(embedded_data[n_pts,0], embedded_data[n_pts,1], marker = 'x',label = 'Grassmannian Mean')
    plt.scatter(embedded_data[n_pts+1,0], embedded_data[n_pts+1,1], marker = 'x',label = 'Grassmannian Median')
    plt.scatter(embedded_data[n_pts+2,0], embedded_data[n_pts+2,1], marker = 'x',label = 'Flag Mean')
    plt.scatter(embedded_data[n_pts+3,0], embedded_data[n_pts+3,1], marker = 'x',label = 'Flag Median')
    plt.scatter(embedded_data[n_pts+4,0], embedded_data[n_pts+4,1], marker = 'x',label = 'Euclidean Mean')
    plt.legend()
    plt.savefig('mds_plot.pdf')












