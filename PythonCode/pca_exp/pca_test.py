import torch

from sklearn.decomposition import PCA

from matplotlib import pyplot as plt

import numpy as np

import pandas as pd

import sys
sys.path.append('../scripts')

import fl_algorithms as fl
import center_algorithms as ca


def plot_pca(data: np.array, weights: np.array, title: str):
    pca_new = PCA(n_components=3)
    pca_new.fit(data)
    pca_new.components_ = weights.T

    new_embedding = pca_new.transform(data)


    ax = plt.figure(title).add_subplot(projection='3d')
    ax.scatter(new_embedding[:99,0],new_embedding[:99,1],new_embedding[:99,2])
    ax.scatter(new_embedding[99:,0],new_embedding[99:,1],new_embedding[99:,2])

    plt.savefig(title)


if __name__ == '__main__':
    data = torch.load('myCATS.pt').numpy().T
    p, n = data.shape

    #regular PCA
    plt.figure('pca')
    pca = PCA(n_components = 3)
    out = pca.fit_transform(data)
    baseline_weights = pca.components_.T
    plot_pca(data, baseline_weights, 'pca_baseline.pdf')


    n_splits = list(range(2,7,1))
    n_trials = 20

    results = pd.DataFrame(columns = ['Number of Splits','Trial', 'Distance', 'Average']) 

    for n_split in n_splits:
        print(f'running split {n_split}')

        for trial in range(n_trials):
            np.random.seed(trial)

            weight_data = []

            left_over = np.arange(p)
            for split in range(n_split):
                left_over = np.array(list(left_over))
                # subset data
                idx0 = np.random.choice(len(left_over), (p-1)//n_split, replace = False)
                idx = left_over[idx0]
                subset_data = data[idx, :]
                # left over idx
                left_over = set(left_over).difference(set(idx))
            

                # compute pca
                pca = PCA(n_components = 3)
                out = pca.fit_transform(subset_data)
                weight_data.append(pca.components_.T)



            weight_data_stacked = np.stack(weight_data, axis = 2)

            #flag average of type (1,2,3)
            f_avg_weights = fl.flag_mean(weight_data_stacked, [1,2,3], verbosity = 0)

        #flag average of type (1,2,3)
        # f_med_weights = fl.flag_median(weight_data, [1,2,3], verbosity = 0)

            #flag average of type (3)
            #equivalent to gr(3,n) flag mean
            g_avg_weights = ca.flag_mean(weight_data, 3)
            #g_avg_weights = fl.flag_mean(weight_data_stacked, [3], verbosity = 0)

            #euclidean average
            e_avg_weights = np.mean(weight_data_stacked, axis = 2)
            e_avg_weights = np.linalg.qr(e_avg_weights)[0][:,:3]


        #pca plots
        #plot_pca(data, f_avg_weights, f'pca_cfm_{n_split}.pdf')
        #plot_pca(data, g_avg_weights, f'pca_g_{n_split}.pdf')
        #plot_pca(data, e_avg_weights, f'pca_e_{n_split}.pdf')

            #distances
            f_dist = fl.chordal_dist(f_avg_weights, baseline_weights, [1,2,3])
        
            # fmed_dists.append(fl.chordal_dist(f_med_weights, baseline_weights, [1,2,3]))

            g_dist = fl.chordal_dist(g_avg_weights, baseline_weights, [1,2,3])

            e_dist = fl.chordal_dist(e_avg_weights, baseline_weights, [1,2,3])

            rand_pt = np.random.rand(n,2)-.5
            rand_pt = np.linalg.qr(rand_pt)[0][:,:3]
            r_dist = fl.chordal_dist(rand_pt, baseline_weights, [1,2,3])


            trial_results = pd.DataFrame(columns = ['Number of Splits','Trial', 'Distance', 'Average'], 
                        data = [[n_split, trial, f_dist, 'FL-mean'],
                                [n_split, trial, g_dist, 'GR-mean'],
                                [n_split, trial, e_dist, 'Euclidean-mean'],
                                [n_split, trial, r_dist, 'Random']])
            results = pd.concat([results, trial_results])

    results.to_csv('pca_res.csv')


    #plt.figure('distances')
    #plt.plot(n_splits, e_dists, marker='o', label = 'Euclidean')
    #plt.plot(n_splits, g_dists, marker = '<', label = 'Flag Mean')
    # plt.plot(n_splits, fmed_dists, marker = 'x', label = 'Chordal Flag Median (Ours)')
    #plt.plot(n_splits, f_dists, marker = 's', label = 'Chordal Flag Mean (Ours)')
    #plt.plot(n_splits, r_dists, marker = '$R$', label = 'Random')

    #plt.xlabel('Splits')
    #plt.ylabel('Chordal Distance')

    #plt.legend()

    #plt.savefig('pca_splits.pdf')
