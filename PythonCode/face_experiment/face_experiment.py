from matplotlib import pyplot as plt
import cv2
import os
import numpy as np

import sys
sys.path.append('../scripts')
import center_algorithms as ca
import fl_algorithms as fl


def load_data(base_dir: str, im_idxs: list) -> list:

    illuminations = ['centerlight', 'leftlight', 'rightlight']

    subspaces = []
    for im_idx in im_idxs:
        subspace = []
        for illumination in illuminations:
            filename = f'subject{im_idx}.{illumination}'

            b=plt.imread(os.path.join(base_dir, filename))
    
            flat_b = b.flatten()
            tall_b = np.reshape(flat_b, (len(flat_b), 1))

            subspace.append(tall_b)

        subspace = np.hstack(subspace)
        subspaces.append(np.linalg.qr(subspace)[0][:,:3])
    
    return subspaces

def run_exp(subspaces: list, exp_id: str) -> None:
    f_mean = fl.flag_mean(np.stack(subspaces, axis = 2),  flag_type = [1,3], oriented = True, verbosity = 1)

    for i, ttl in zip(range(3),['centerlight', 'leftlight', 'rightlight']):
        plt.figure()
        plt.imshow(-np.reshape(f_mean[:,i], (243, 320)), 'gray')
        plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
        plt.savefig(f'./images/exp{exp_id}_real_flag_mean_{ttl}.pdf')

    f_mean = ca.flag_mean(subspaces,  r=3)

    for i, ttl in zip(range(3),['centerlight', 'leftlight', 'rightlight']):
        plt.figure()
        plt.imshow(-np.reshape(f_mean[:,i], (243, 320)), 'gray')
        plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
        plt.savefig(f'./images/exp{exp_id}_flag_mean_{ttl}.pdf')

if __name__ == '__main__':
    subspaces = load_data('YaleFaceDatabase', ['01','02','03'])
    run_exp(subspaces, '123')

    subspaces = load_data('YaleFaceDatabase', ['05','06','07'])
    run_exp(subspaces, '567')
 
