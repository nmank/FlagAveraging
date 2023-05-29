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


if __name__ == '__main__':

    n_pts = 100
    n = 10
    flag_type = [1,2,3]  

    n_trials = 5

    n_its = 10

    k = flag_type[-1]

    noise_data = .2
    noise_init = .2

    rfm_errs = []
    rfm_iters = []
    rfm_costs = []

    data, center_pt = generate_flag_data_noisy(n_pts, n, flag_type, noise_data)
    stacked_data = np.stack(data, axis = 2)

    noises = np.arange(0,50,4)*noise_init
    for noise in noises:

        np.random.seed(123)

        added_noise = noise*(np.random.rand(n, k)-.5)

        init_pt = np.linalg.qr(center_pt + added_noise)[0][:,:k]

        real_flag_median, rfm_it, rfm_cost = fla.flag_median(stacked_data,  
                                           flag_type = flag_type, 
                                           initial_point_median = init_pt,
                                           return_iters = True,
                                           return_cost = True)       

        #distances to center pt
        rfm_errs.append(fla.chordal_dist(real_flag_median, center_pt, flag_type))
        rfm_iters.append(rfm_it)
        rfm_costs.append(rfm_cost)

    res = pd.DataFrame()
    res['Noise'] = noises
    res['Distance from Center'] = rfm_errs
    res['Iterations'] = rfm_iters
    res['Cost'] = rfm_costs
    res.to_csv('flag_median_init.csv')

    # fig1, (ax21, ax22, ax23) = plt.subplots(3, 1)
    # ax21.plot(noises, rfm_errs, marker = 's')
    # ax21.set(ylabel='Distance From Center')

    # ax22.plot(noises, rfm_iters, marker = 's')
    # ax22.set(ylabel='Iterations')

    # ax23.plot(noises, rfm_costs, marker = 's')
    # ax23.set(ylabel='Cost')

    # ax23.set(xlabel='Noise')
    # fig1.suptitle('Initialization for Flag Median IRLS')

    # fig1.tight_layout()

    # fig1.savefig('flag_median_init.pdf')

    # res = pd.DataFrame()
    # res['Noise'] = noises
    # res['Distance from Center'] = rfm_errs
    # res['Iterations'] = rfm_its
    # res.to_csv('flag_median_init.csv')
    
    # fig1, (ax1, ax2) = plt.subplots(2, 1)
    # ax1.plot(noises, rfm_errs, marker = 's')
    # ax1.set(ylabel='Distance From Center')

    # ax2.plot(noises, rfm_its, marker = 's')
    # ax2.set(ylabel='Iterations')

    # ax2.set(xlabel='Noise')
    # fig1.suptitle('Initialization for Flag Median IRLS')

    # fig1.tight_layout()

    # fig1.savefig('flag_median_init.pdf')

    # noises = np.arange(1,1000,10)*.01
    # # for i in range(1,n_trials+1):
    #     # noise = .01*(i/n_trials)
    #     # noise = 10**(-i+2)
    #     # noises.append(noise)

    fm_errs = []
    fm_costs = []
    fm_iters = []

    for noise in noises:

        np.random.seed(123)

        added_noise = noise*(np.random.rand(n, k)-.5)

        init_pt = np.linalg.qr(center_pt+ added_noise)[0][:,:k]

        rfm_out = fla.flag_mean(stacked_data,  
                                       flag_type = flag_type, 
                                       initial_point = init_pt,
                                       return_all = True)       

        rfm = rfm_out.point
        cost = rfm_out.cost
        iters = rfm_out.iterations
        #distances to center pt

        fm_errs.append(fla.chordal_dist(rfm, center_pt, flag_type))
        fm_iters.append(iters)
        fm_costs.append(cost)

    res = pd.DataFrame()
    res['Noise'] = noises
    res['Distance from Center'] = fm_errs
    res['Iterations'] = fm_iters
    res['Cost'] = fm_costs
    res.to_csv('flag_mean_init.csv')

    fig2, (ax21, ax22, ax23) = plt.subplots(3, 1)
    ax21.plot(noises, rfm_errs, marker = 's',label = 'FL-Median (IRLS)')
    ax21.plot(noises, fm_errs, marker = 'x',label = 'FL-Mean RTR')
    ax21.set(ylabel='Error')

    ax22.plot(noises, rfm_iters, marker = 's',label = 'FL-Median (IRLS)')
    ax22.plot(noises, fm_iters, marker = 'x',label = 'FL-Mean RTR')
    ax22.set(ylabel='Iterations')

    ax23.plot(noises, rfm_costs, marker = 's',label = 'FL-Median (IRLS)')
    ax23.plot(noises, fm_costs, marker = 'x',label = 'FL-Mean (RTR)')
    ax23.set(ylabel='Cost')

    handles, labels = ax23.get_legend_handles_labels()
    ax23.legend(handles, labels, loc='center right')

    ax23.set(xlabel='Noise')
    # fig2.suptitle('Initialization')

    fig2.tight_layout()

    fig2.savefig('init_ablation.pdf')

    # plt.plot(noises, rfm_errs)
    # plt.xlabel('Noise')
    # plt.title('Flag Mean Robustness to Initialization')
    # plt.ylabel('Distance From Center')
