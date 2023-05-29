import sys
sys.path.append('../scripts')

import os

import fl_algorithms as fla
import center_algorithms as ca

from matplotlib import pyplot as plt

import numpy as np

import pandas as pd
        

def generate_flag_data(n_pts: int, flag_type: list, seed: int = 2) -> list:
    np.random.seed(seed)

    k = flag_type[-1]
    center_pt = np.linalg.qr(np.random.rand(n, k)-.5)[0][:,:k]

    data = []
    for i in range(n_pts):
        rand_mat = center_pt + 0.001*(np.random.rand(n, k)-.5)
        data.append(np.linalg.qr(rand_mat)[0][:,:k])

    return data, center_pt


if __name__ == '__main__':
    
    n_pts = 100
    n = 10
    flag_type = [1,3]   

    noise = .01

    n_trials = 50

    k = flag_type[-1]

    rfm_costs = []
    n_costs = [] 

    rfm_its = []
    n_its = []

    rfm_errs = []
    n_errs = []

    ii = 0
    for seed in range(n_trials):
        print(f'running trial {seed}')

        data, center_pt = generate_flag_data(n_pts, flag_type, seed = seed)
        stacked_data = np.stack(data, axis = 2)

        try:
            added_noise = noise*(np.random.rand(n, k)-.5)
            init_pt = np.linalg.qr(center_pt+ added_noise)[0][:,:k]
            rfm_out = fla.flag_mean(stacked_data,  flag_type = flag_type, return_all=True, initial_point = init_pt)

            nm_out = fla.flag_mean(stacked_data, flag_type = flag_type, manifold = 'flag', return_all=True, initial_point = init_pt)        

            real_flag_mean = rfm_out.point
            nguyen_mean = nm_out.point

            rfm_costs.append(rfm_out.cost)
            n_costs.append(nm_out.cost)

            rfm_its.append(rfm_out.iterations)
            n_its.append(nm_out.iterations)

            #distances to center pt
            rfm_errs.append(fla.chordal_dist(real_flag_mean, center_pt, flag_type))
            n_errs.append(fla.chordal_dist(nguyen_mean, center_pt, flag_type))

            ii += 1
        except np.linalg.LinAlgError:
            print('SVD didnt converge')
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    print(f'{np.round(100*ii/n_trials,2)}% of trials were successful')
    print('---------------------------')
    print('Costs')
    print(f'ours: {np.round(np.mean(rfm_costs),8)} +/- {np.round(np.std(rfm_costs),8)}')
    print(f'theirs: {np.round(np.mean(n_costs),6)} +/- {np.round(np.std(n_costs),8)}')
    print('---------------------------')
    print('Iterations')
    print(f'ours: {np.round(np.mean(rfm_its),2)} +/- {np.round(np.std(rfm_its),2)}')
    print(f'theirs: {np.round(np.mean(n_its),2)} +/- {np.round(np.std(n_its),2)}')
    print('---------------------------')
    print('Errors')
    print(f'ours: {np.round(np.mean(rfm_errs),8)} +/- {np.round(np.std(rfm_errs),8)}')
    print(f'theirs: {np.round(np.mean(n_errs),8)} +/- {np.round(np.std(n_errs),8)}')


'''
100.0% of trials were successful
---------------------------
Costs
ours: 0.00020618 +/- 4.91e-06
theirs: 0.001562 +/- 0.00159257
---------------------------
Iterations
ours: 2.0 +/- 0.0
theirs: 9.72 +/- 2.73
---------------------------
Errors
ours: 0.00014194 +/- 2.03e-05
theirs: 0.00299503 +/- 0.00214118
'''









