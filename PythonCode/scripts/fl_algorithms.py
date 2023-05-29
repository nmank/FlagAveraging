'''
by Nathan Mankovich

Algorthms for averaging on flags using chordal distance from  Pitival et al.
'''

import numpy as np
import autograd.numpy as anp
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers
from RealFlag import RealFlag

def flag_mean(data: np.array, flag_type: list , weights: np.array = [], 
              initial_point: np.array = None, oriented = False, 
              verbosity: int = 0, manifold: str = 'stiefel',
              return_all: bool = False):
    '''
    chordal flag mean on p flags living in n-space with flag type flag_type
    Inputs:
        data: np.array- n x k x p matrix of data on flag manifold
        flag_type: list- type of flag as a list (eg. 1,2,3)
        weights: np.array- weights for the flag median
        initial_point: np.array- initial point for trust regions solver
        oriented: bool- True to compute oriented average, only for Flags of type (1,2,...,n-1;n)
        verbosity: int- print out details from RTR
        manifold: str- 'stiefel' for Mankovich et al. algorithm and 'flag' for Nugyen algorithm
        return_all: bool- True to return pymanopt object with cost, number of iterations, and more
    Outputs:
        the chordal flag mean with or without extra information
    '''
    n,k,p = data.shape

    #construct weight matrix
    weight_mat = np.eye(p)
    if len(weights) > 0:
         weight_mat[np.arange(p), np.arange(p)] = np.sqrt(weights)

    
    p_mats = []
    id_mats = []

    for i in range(len(flag_type)):

        #set the initial f_type_prev to 0
        f_type = flag_type[i]
        if i-1 < 0:
            f_type_prev = 0
        else:
            f_type_prev = flag_type[i-1]
        
        #make projection matrices
        dim_d_mat = data[:,f_type_prev:f_type,:] @ weight_mat
        dim_d_mat = np.reshape(dim_d_mat, (n,(f_type-f_type_prev)*p))
        p_mat = dim_d_mat @ dim_d_mat.T 
        p_mats.append(p_mat)

        #make identity matrices
        id_mat = np.zeros((k,k))
        id_mat[np.arange(f_type_prev,f_type,1),np.arange(f_type_prev,f_type,1)] = 1
        id_mats.append(id_mat)

    if manifold == 'stiefel':
        # initialize a stiefel manifold object
        St = pymanopt.manifolds.stiefel.Stiefel(n,k)
    elif manifold == 'flag':
        # get proper flag type
        
        real_flag_type = []
        real_flag_type.append(flag_type[0])
        for i in range(1,len(flag_type)):
            real_flag_type.append(flag_type[i] - flag_type[i-1])
        real_flag_type.append(n - flag_type[-1])
        real_flag_type.reverse()

        print(real_flag_type)        

        # initialize a flag manifold object
        St = RealFlag(np.array(real_flag_type))

    #setu up the objective function
    @pymanopt.function.autograd(St)
    def cost(point):
        f = 0
        for i in np.arange(len(p_mats)):
            if i < 1:
                f_type_before = 0
            else:
                f_type_before = flag_type[i-1]

            k_i = flag_type[i] - f_type_before
            
            f += p*k_i-np.trace(id_mats[i] @ point.T @ p_mats[i] @ point)

        return f

    problem = pymanopt.Problem(St, cost)

# , max_iterations = 20, max_time = 20)
    optimizer = pymanopt.optimizers.trust_regions.TrustRegions(verbosity = verbosity)
    #optimizer = pymanopt.optimizers.conjugate_gradient.ConjugateGradient(verbosity = verbosity)
    # optimizer = pymanopt.optimizers.conjugate_gradient.ConjugateGradient(verbosity = verbosity, max_iterations = 20)

    #run the trust regions algorithm
    if initial_point is None:
        mu = np.mean(data, axis = 2)
        initial_point = np.linalg.qr(mu)[0][:,:flag_type[-1]]
        result = optimizer.run(problem, initial_point = initial_point)
    else:
        result = optimizer.run(problem, initial_point = initial_point)
    
    f_mean = result.point

    #make it an oriented flag mean
    #this might only work for flag types FL(1,2,3,...,n-1;n)
    if oriented:
        euclidean_mean = np.mean(data, axis = 2)
        for i in range(k):
            cos_theta = np.dot(euclidean_mean[:,i],f_mean[:,i])
            if cos_theta < 0:
                f_mean[:,i] = -f_mean[:,i]
    
    if not return_all:
        return f_mean
    else:
        return result


def chordal_dist(X: np.array, Y: np.array, flag_type: list)-> float:
    '''
    Compute pitival chordal distance between X and Y
    '''
    c_dist = 0
    for i in range(len(flag_type)):
        # make projection matrices
        f_type = flag_type[i]
        if i < 1:
            f_type_prev = 0
        else:
            f_type_prev = flag_type[i-1]

        k_i = f_type-f_type_prev
        
        dimX = Y[:,f_type_prev:f_type]

        dimY = X[:,f_type_prev:f_type]

        c_dist += k_i - np.trace(dimY.T @ dimX @ dimX.T @ dimY)

    if c_dist < 0:
        c_dist = 0
        print('warning: distance is close to 0')
    c_dist = np.sqrt(c_dist)

    return c_dist


def flag_median(data: np.array, flag_type: list, initial_point_median: np.array = None, 
                random_seed: int = 1, conv_eps: float = .000001, wt_eps: float = .000001,
                weights: np.array = [], max_iters: int = 10, initial_point_mean: np.array = None,
                verbosity: int = 0, oriented: bool = False, return_iters: bool = False,
                return_cost: bool = False):
    '''
    chordal flag mean on p flags living in n-space with flag type flag_type

    Inputs:
        data: np.array- n x k x p matrix of data on flag manifold
        flag_type: list- type of flag as a list (eg. 1,2,3)
        initial_point_median: np.array- initial point for the median algorithm
        random_seed: int- for reproducibility
        conv_eps: float- the convergence parameter for the flag median
        wt_eps: float- epsilon for the IRLS weights
        weights: np.array- weights for the flag median
        max_iters: int- max iterations for flag median convergence
        initial_point_mean: np.array- initial point for trust regions solver
        verbosity: int- print out details from Steifel RTR
        oriented: bool- True to compute oriented average, only for Flags of type (1,2,...,n-1;n)
        return iters: bool- True to return the number of iterations of IRLS
        return_cost: bool- True to return cost at each iteration
    Outputs:
        the chordal flag median with or without extra information
    '''

    n,k,p = data.shape

    if len(weights) < 1:
        weights = np.ones(p)

    #generate initial guess
    if initial_point_median is None:
        np.random.seed(random_seed)
        Y_new = np.linalg.qr(np.random.rand(n,k))[0][:,:k]
    else:
        Y_new = initial_point_median

    n_iters = 0
    err = 10
    errs = []

    #while not converged
    while (err > conv_eps) and (n_iters < max_iters):
        Y = Y_new

        #calculate weights
        median_weights = np.array([1/max(chordal_dist(data[:,:,i], Y, flag_type), wt_eps) for i in range(p)])

        combined_weights = weights * median_weights

        #compute weighted mean
        Y_new = flag_mean(data, 
                          flag_type, 
                          combined_weights, 
                          initial_point_mean, 
                          verbosity = verbosity, 
                          oriented = oriented)

        #compute error
        err = chordal_dist(Y, Y_new, flag_type)

        if verbosity > 0:
            print(f'iteration: {n_iters} | err: {err}')

        errs.append(err)
        n_iters +=1

    if verbosity > 0:
        print('flag median found!')

    if return_cost:
        cost = cost_value(Y_new, data, flag_type)
    
    if return_iters and return_cost:
        return Y_new, n_iters, cost
    elif return_iters:
        return Y_new, n_iters
    elif return_cost:
        return Y_new, cost
    else:
        return Y_new


def cost_value(estimate: np.array, data: np.array, flag_type: list , weights: np.array = [], median: bool = False):
    '''
    Compute the sum of the chordal distance between estimate and data with weights

    estimate: np.array- n x k one flag
    data: np.array- n x k x p matrix of data on flag manifold
    flag_type: list- type of flag as a list (eg. 1,2,3)
    weights: np.array- weights for the flag median
    median: bool- True for chordal distance, False for squared chordal distance
    '''
    
    n,k,p = data.shape

    if len(weights) < 1:
        weights = np.ones(p)

    cost = 0
    for j in range(p):
        err = 0
        for i in range(len(flag_type)):
            # make projection matrices
            f_type = flag_type[i]
            if i < 1:
                f_type_prev = 0
            else:
                f_type_prev = flag_type[i-1]

            k_i = f_type-f_type_prev
            
            dim_d_mat = data[:,f_type_prev:f_type,j]
            # dim_d_mat = np.reshape(dim_d_mat, (n,k_i))

            dim_estimate = estimate[:,f_type_prev:f_type]


            err += k_i - weights[j]*np.trace(dim_estimate.T @ dim_d_mat @ dim_d_mat.T @ dim_estimate)
        if median:
            err = np.sqrt(err)
    
        cost += err

    return cost


def distance_matrix(X: np.array, C: list, flag_type: list) -> np.array:
    '''
    Calculate a chordal distance matrix for the dataset (columns) and the centers (rows)

    Inputs:
        X: np.array- n x k x p dataset with p points of flags in n-space
        C: list- centers a.k.a. codebook of flags
        flag_type: list- type of flag
    Outputs:
        Distances
    '''
    n = X.shape[2]
    m = len(C)
    Distances = np.zeros((m,n))

    for i in range(m):
        for j in range(n):
            Distances[i,j] = chordal_dist(C[i], X[:,:,j], flag_type)

    return Distances


def cluster_purity(X: np.array, centers: list, labels_true: list, flag_type: list, similarity: bool = False, feature_labels: list = None) -> float:
    '''
    Calculate the cluster purity of the dataset

    Inputs:
        X: np.array- n x k x p dataset with p points of flags in n-space
        C: list- centers a.k.a. codebook of flags
        flag_type: list- type of flag
        labels_true: list- the true labels
    Outputs:
        the cluster purity
    '''

    #calculate distance matrix
    d_mat = distance_matrix(X, centers, flag_type)

    #find the closest center for each point
    index = np.argmin(d_mat, axis = 0)
    
    count = 0
    for i in range(len(centers)):
        idx = np.where(index == i)[0]
        if len(idx) != 0:
            cluster_labels = [labels_true[i] for i in idx]
            most_common_label = max(set(cluster_labels), key = cluster_labels.count)
            # count += cluster_labels.count(most_common_label)
            count += cluster_labels.count(most_common_label)/len(idx)

    # return count/len(X)
    purity = count/len(centers)
    return purity

    
def lbg_flag(X: np.array, epsilon: float, centers: list = [],  n_centers: int = 17, 
                 opt_type: str = 'mean', n_its: int = 10, seed: int = 1, flag_type: list = []) -> tuple:
    '''
    LBG clustering with flags
    
    Inputs:
        X: np.array- n x k x p dataset with p points of flags in n-space
        epsilon: float- convergence parameter
        centers: list- initial centers
        n_centers- int: codebook size
        opt_type- strinr: type of LBG clustering
            'mean' chordal flag-mean
            'median' chordal flag-median
        n_its: int- number of iterations of LBG clustering
        seed: int- seed for initial codebook selection
        flag_type: list- type of flag
    Outputs:
        centers: list- numpy arrays for the centers
        errors: list- normalized consecutive distortion error at each iteration
        distortions: list- cluster distortions at each iteration
    '''
    n_pts = X.shape[2]
    error = 1
    distortions = []

    #init centers if centers aren't provided
    if len(centers) == 0:
        np.random.seed(seed)
        centers = []
        for i in range(n_centers):
            centers.append(X[:,:,np.random.randint(n_pts)])

    #calculate distance matrix
    d_mat = distance_matrix(X, centers, flag_type)

    #find the closest center for each point
    index = np.argmin(d_mat, axis = 0)

    #calculate first distortion
    new_distortion = np.sum(d_mat[index])

    distortions.append(new_distortion)


    errors = []
    while error > epsilon and len(errors) < 20:
        print(f'iteration {len(errors)}')

        #set new distortion as old one
        old_distortion = new_distortion

        m = len(centers)

        #calculate new centers
        centers = []
        for c in range(m):
            idx = np.where(index == c)[0]
            if len(idx) > 0:
                if opt_type == 'mean':
                    centers.append(flag_mean(X[:,:,idx], flag_type, verbosity = 1))
                elif opt_type == 'median':
                    centers.append(flag_median(X[:,:,idx], flag_type, random_seed = seed, max_iters = n_its, verbosity = 1))
                else:
                    print('opt_type not recognized')

        #calculate distance matrix
        d_mat = distance_matrix(X, centers, flag_type)

        #find the closest center for each point
        index = np.argmin(d_mat, axis = 0)

        #new distortion
        new_distortion = np.sum(d_mat[index])

        distortions.append(new_distortion)

        if new_distortion <0.00000000001:
            error = 0
        else:
            error = np.abs(new_distortion - old_distortion)/old_distortion
        print(error)
        errors.append(error)

    return centers, errors, distortions    
