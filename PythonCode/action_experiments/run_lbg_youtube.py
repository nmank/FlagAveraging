import sys
sys.path.append('../scripts/')

import numpy as np
from matplotlib import pyplot as plt
import fl_algorithms as fla
import center_algorithms as ca
from os import listdir
import pandas
import seaborn as sns

'''
A script for running the youtube lbg clustering experiments.
'''



#number of iterations of LBG clustering
n_its= 10
#random seed for reproducible results
seed = 0
#number of trials for each number of centers
n_trials = 10
#type of flag (for grassmannian lbg we just use Gr(10,n))
fl_type = [1,2,3,4,5,6,7,8,9,10]

#where do save the results png
f_name = './youtube_lbg_'+str(n_trials)+'trials_small1.png'

#path to youtube action videos as numpy arrays (.npy files)
base_path = '/home/nmank/YouTubeData/action_youtube_gr_small/'


#load the numpy arrays
X = []
labels_true = []
count = 0
for label in listdir(base_path):
    if count < 5:
        current_dir = base_path+label+'/'
        for f in listdir(current_dir):
            X.append(np.load(current_dir+f))
            labels_true.append(label)
    count += 1


#stack the list of numpy arrays 
stacked_X = np.stack(X, axis = 2)

#results data frame
Purities = pandas.DataFrame(columns = ['Algorithm','Codebook Size','Cluster Purity'])

#run trials for different n centers (codebook sizes)
for n in range(4, 24, 4):
    sin_purities = []
    cos_purities = []
    flg_purities = []
    for trial in range(n_trials):
        print('cluster '+str(n)+' trial '+str(trial))
        print('.')
        print('.')
        print('.')
        print('flag median start')
        centers_flagmedian, error_flagmedian, dist_flagmedian = ca.lbg_subspace(X, .0001, n_centers = n, opt_type = 'sine', n_its = 10, seed = trial)
        flagmedian_purity = ca.cluster_purity(X, centers_flagmedian, labels_true)
        print('real flag mean start')
        centers_real_flagmean, error_real_flagmean, dist_real_flagmean = fla.lbg_flag(stacked_X, epsilon = .0001, n_centers = n, seed = trial, flag_type = fl_type)
        real_flagmean_purity = fla.cluster_purity(stacked_X, centers_real_flagmean, labels_true, flag_type = fl_type)
        print('real flag median start')
        centers_real_flagmedian, error_real_flagmedian, dist_real_flagmedian = fla.lbg_flag(stacked_X, epsilon = .0001, opt_type = 'median', n_centers = n, seed = trial, flag_type = fl_type)
        real_flagmedian_purity = fla.cluster_purity(stacked_X, centers_real_flagmedian, labels_true, flag_type = fl_type)        
        print('flag mean')
        centers_flagmean, error_flagmean, dist_flagmean = ca.lbg_subspace(X, .0001, n_centers = n, opt_type = 'sinesq', seed = trial)
        flagmean_purity = ca.cluster_purity(X, centers_flagmean, labels_true)


        Purities = Purities.append({'Algorithm': 'GR-median', 
                                'Codebook Size': n,
                                'Cluster Purity': flagmedian_purity},
                                ignore_index = True)
        

        Purities = Purities.append({'Algorithm': 'FL-mean', 
                                'Codebook Size': n,
                                'Cluster Purity': real_flagmean_purity},
                                ignore_index = True)

        Purities = Purities.append({'Algorithm': 'FL-median', 
                                'Codebook Size': n,
                                'Cluster Purity': real_flagmedian_purity},
                                ignore_index = True)

        Purities = Purities.append({'Algorithm': 'GR-mean', 
                                'Codebook Size': n,
                                'Cluster Purity': flagmean_purity},
                                ignore_index = True)
    # print(Purities)
    # Purities.to_csv(f'youtube_LBG_results_{trial}trials.csv')
    
#save the results as a csv
Purities.to_csv(f'./youtube_lbg_{trial}trials.csv')

Purities = Purities.sort_values(by = 'Algorithm')
sns.boxplot(x='Codebook Size', y='Cluster Purity', hue='Algorithm', data = Purities)
plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
plt.savefig(f_name, bbox_inches='tight')


