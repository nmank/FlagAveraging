import sys
sys.path.append('../scripts')

import fl_algorithms as fla
import center_algorithms as ca

from matplotlib import pyplot as plt

import numpy as np

import pandas as pd

def generate_flag_data_noisy(n_pts: int, n: int, flag_type: list, noise: float, seed: int = 1) -> list:
    np.random.seed(seed)

    k = flag_type[-1]
    center_pt = np.linalg.qr(np.random.rand(n, k)-.5)[0][:,:k]

    data = []
    for i in range(n_pts):
        rand_mat = center_pt + noise*(np.random.rand(n, k)-.5)
        data.append(np.linalg.qr(rand_mat)[0][:,:k])

    return data, center_pt
        

def generate_flag_data_outliers(n_inliers: int, n_outliers: int, flag_type: list, seed: int = 2) -> list:
    np.random.seed(seed)

    k = flag_type[-1]
    center_pt = np.linalg.qr(np.random.rand(n, k)-.5)[0][:,:k]

    data = []
    for i in range(n_inliers):
        rand_mat = center_pt + 0.001*(np.random.rand(n, k)-.5)
        data.append(np.linalg.qr(rand_mat)[0][:,:k])
    for i in range(n_outliers):
        rand_mat = center_pt + (np.random.rand(n, k)-.5)
        data.append(np.linalg.qr(rand_mat)[0][:,:k])

    return data, center_pt


if __name__ == '__main__':
    
    n_pts = 100
    n = 10
    flag_type = [1,3]   

    n_its = 10

    k = flag_type[-1]
 
    m_errs = []
    med_errs = []
    rfm_errs = []
    rfmed_errs = []
    n_errs = []
    e_errs = []

    noises = []

    #for exp in range(1,100,5):
        #noise = exp/50
        #noises.append(noise)

        #data, center_pt = generate_flag_data(n_pts, n, flag_type, noise)
    for n_outliers in range(20):
        noises.append(n_outliers/n_pts)
        data, center_pt = generate_flag_data_outliers(n_pts-n_outliers, n_outliers, flag_type)
        stacked_data = np.stack(data, axis = 2)

        flag_mean = ca.flag_mean(data, k)

        flag_median = ca.irls_flag(data, k, n_its, 'sine', opt_err = 'sine')[0]

        real_flag_mean = fla.flag_mean(stacked_data,  flag_type = flag_type)

        nguyen_mean = fla.flag_mean(stacked_data, flag_type = flag_type, manifold = 'flag')        

        real_flag_median = fla.flag_median(stacked_data,  flag_type = flag_type, max_iters = 100)

        euclidean_mean = np.mean(stacked_data, axis = 2)
        euclidean_mean = np.linalg.qr(euclidean_mean)[0][:,:flag_type[-1]]

        #distances to center pt
        m_errs.append(fla.chordal_dist(flag_mean, center_pt, flag_type))

        med_errs.append(fla.chordal_dist(flag_median, center_pt, flag_type))

        rfm_errs.append(fla.chordal_dist(real_flag_mean, center_pt, flag_type))

        rfmed_errs.append(fla.chordal_dist(real_flag_median, center_pt, flag_type))

        n_errs.append(fla.chordal_dist(nguyen_mean, center_pt, flag_type))

        e_errs.append(fla.chordal_dist(euclidean_mean, center_pt, flag_type))

    # plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
    ax1.plot(noises, e_errs, marker='o', label = 'Euclidean')
    ax1.plot(noises, n_errs, marker = '*', label = 'Flag Mean (Nguyen)')
    ax1.plot(noises, rfmed_errs, marker = 's', label = 'Flag Median (Ours)')
    ax1.plot(noises, rfm_errs, marker = 'x', label = 'Flag Mean (Ours)')
    ax1.plot(noises, med_errs, marker = '>', label = 'Grassmannian Median')
    ax1.plot(noises, m_errs, marker = '<', label = 'Grassmannian Mean')
    ax1.set(xlabel='Outlier Ratio', ylabel='Chordal Distance from Center')
    ax1.legend()

    outlier_results = pd.DataFrame()
    outlier_results['Outlier Ratio'] = noises
    outlier_results['Euclidean'] = e_errs
    outlier_results['FL-Mean (Nguyen)'] = n_errs
    outlier_results['FL-Mean (Ours)'] = rfm_errs
    outlier_results['FL-Median (Ours)'] = rfmed_errs
    outlier_results['GR-Mean'] = m_errs
    outlier_results['GR-Median'] = med_errs

    outlier_results.to_csv('flagmeancompare_outliers.csv')

    # plt.savefig('synth_flags_outliers.pdf')

    m_errs = []
    med_errs = []
    rfm_errs = []
    rfmed_errs = []
    n_errs = []
    e_errs = []

    noises = []


    for exp in range(1,45,5):
        noise = exp/50
        noises.append(noise)

        data, center_pt = generate_flag_data_noisy(n_pts, n, flag_type, noise)
        stacked_data = np.stack(data, axis = 2)

        flag_mean = ca.flag_mean(data, k)

        flag_median = ca.irls_flag(data, k, n_its, 'sine', opt_err = 'sine')[0]

        real_flag_mean = fla.flag_mean(stacked_data,  flag_type = flag_type)

        nguyen_mean = fla.flag_mean(stacked_data, flag_type = flag_type, manifold = 'flag')        

        real_flag_median = fla.flag_median(stacked_data,  flag_type = flag_type, max_iters = 100)

        euclidean_mean = np.mean(stacked_data, axis = 2)
        euclidean_mean = np.linalg.qr(euclidean_mean)[0][:,:flag_type[-1]]

        #distances to center pt
        m_errs.append(fla.chordal_dist(flag_mean, center_pt, flag_type))

        med_errs.append(fla.chordal_dist(flag_median, center_pt, flag_type))

        rfm_errs.append(fla.chordal_dist(real_flag_mean, center_pt, flag_type))

        rfmed_errs.append(fla.chordal_dist(real_flag_median, center_pt, flag_type))

        n_errs.append(fla.chordal_dist(nguyen_mean, center_pt, flag_type))

        e_errs.append(fla.chordal_dist(euclidean_mean, center_pt, flag_type))

    # plot results
    ax2.plot(noises, e_errs, marker='o', label = 'Euclidean')
    ax2.plot(noises, n_errs, marker = '*', label = 'FL-Mean (Nguyen)')
    ax2.plot(noises, rfmed_errs, marker = 's', label = 'FL-Median (Ours)')
    ax2.plot(noises, rfm_errs, marker = 'x', label = 'FL-Mean (Ours)')
    ax2.plot(noises, med_errs, marker = '>', label = 'GR-Median')
    ax2.plot(noises, m_errs, marker = '<', label = 'GR-Mean')
    ax2.set(xlabel='Noise')

    

    plt.tight_layout()
    plt.savefig('flagmeancompare.pdf')

    noise_results = pd.DataFrame()
    noise_results['Noise'] = noises
    noise_results['Euclidean'] = e_errs
    noise_results['FL-Mean (Nguyen)'] = n_errs
    noise_results['FL-Mean (Ours)'] = rfm_errs
    noise_results['FL-Median (Ours)'] = rfmed_errs
    noise_results['GR-Mean'] = m_errs
    noise_results['GR-Median'] = med_errs

    noise_results.to_csv('flagmeancompare_noise.csv')

