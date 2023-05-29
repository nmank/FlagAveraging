'''
by Nathan Mankovich

FlagIRLS and Weiszfeld-type algorithm for Grassmannian averaging.
Used for The Flag Median and FlagIRLS, CVPR 2023
'''
import numpy as np



def gr_log(X: np.array,Y: np.array) -> np.array:
    '''
    Log map on the Grassmannian.
    
    Inputs:
        X (np.array) a point about which the tangent space has been computed
        Y (np.array) the point on the Grassmannian manifold that's mapped to the tangent space of X
    Outputs:
        TY (np.array) Y in the tangent space of X
    '''
    m = X.shape[0]

    #temp = (np.eye(m)-X @ X.T) @ Y @ np.linalg.inv(X.T@Y)
    #The following line is a slightly faster way to compute temp.

    temp = np.eye(m) @ Y @ np.linalg.inv(X.T @ Y) - X @ (X.T @ Y) @ np.linalg.inv(X.T @ Y)
    U,S,V = np.linalg.svd(temp, full_matrices = False)
    Theta = np.arctan(S)
    
    TY = U @ np.diag(Theta) @ V.T
    
    return TY
                                             

def gr_exp(X: np.array, TY: np.array) -> np.array:
    '''
    Exponential map on the Grassmannian.

    Inputs:
        X: (np.array) is the point about which the tangent space has been
          computed.
        TY: (np.array) is a point in the tangent space of X.
    Outputs:
        Y: The output of the exponential map.
    
    '''
    
    U, S, V = np.linalg.svd(TY, full_matrices = False)
    Y = X @ V @ np.diag(np.cos(S)) + U @ np.diag(np.sin(S))

    return Y


def gr_dist(X: np.array, Y: np.array) -> np.array:
    '''
    Geodesic distance on the Grassmannian

    inputs:
        X- numpy array
        Y- numpy array
    outputs:
        dist- the geodesic distance between X and Y
    '''
    if X.shape[1] > 1:
        U,S,V = np.linalg.svd(X.T @ Y, full_matrices = False)
        S[np.where(S >1)] = 1
#         S[np.where(S < -1)] = -1
        angles = np.real(np.arccos(S))
#         print(angles)
        dist = np.linalg.norm(angles)
    else:
        dist = calc_error_1_2([X], Y, 'geodesic')
    return dist


def l2_median(data: list, alpha: float, r: int, max_itrs: int, seed: int = 0, init_datapoint: bool = False) -> tuple:
    '''
    Code adopted from Tim Marrinan (translated from matlab into python)

    inputs:
        data- list of numpy arrays 
        alpha- float for the step size
        r- integer for Gr(r,n) where the output 'lives'
        max_itrs- integer for the maximum number of iterations
        seed- integer for the numpy random seed for the algorithm initialization
        init_datapoint- boolean, True for intializing at a datapoint, False for random initialization
    outputs:
        Y- numpy array for the l2-median
        err- objective function values at each iteration
    '''
    
    n = data[0].shape[0]
    
    if init_datapoint:
        np.random.seed(seed)
        Y = data[np.random.randint(len(data))]
    else:
        np.random.seed(seed)
        Y_raw = np.random.rand(n,r)-.5
        Y = np.linalg.qr(Y_raw)[0][:,:r]
    
    itr = 0
    errs = []
    diff = 1
    
    while diff > 0.000001 and itr < max_itrs:
        d_fracs = 0
        ld_fracs = np.empty((n,r))
        dists = []
        for x in data:
            dists.append(gr_dist(x, Y))
            if dists[-1] > .0001:
                d_fracs += 1 / dists[-1]
                ld_fracs += gr_log(Y, x) / dists[-1]
            elif not init_datapoint:
                print('converged to datapoint')

        if len(ld_fracs)==0:
            return Y
        else:
            vk = ld_fracs/d_fracs
            Y = gr_exp(Y, alpha * vk)
            
            errs.append(np.sum(dists))
            
            if itr > 0:
                diff = np.abs(errs[-2] - errs[-1])
            
            if not np.allclose(Y.T @ Y, np.eye(r,r)):
                Y = np.linalg.qr(Y)[0][:,:r]
            
            itr+=1 
    
    return Y, errs


def calc_error_1_2(data: list, Y: np.array, sin_cos: str, labels: list = None) -> float:
    '''
    Calculate objective function value. 

    Inputs:
        data - a list of numpy arrays representing points in Gr(k_i,n)
        Y - a numpy array representing a point on Gr(r,n) 
        sin_cos - a string defining the objective function
                    'cosine' = Maximum Cosine
                    'sine' = Sine Median
                    'sinsq' = Flag Mean
                    'geodesic' = Geodesic Median (CAUTION: only for k_i = r = 1)
                    'l2_med' = geodesic distance if k_i or r > 1
                    'zobs' = a subspace version of zobs
        labels - labels for the features within the data
    Outputs:
        err - objective function value
    '''
    k = Y.shape[1]
    err = 0
    if sin_cos == 'sine':
        for x in data:
            r = np.min([k,x.shape[1]])
            sin_sq = r - np.trace(Y.T @ x @ x.T @ Y)
            #fixes numerical errors
            if sin_sq < 0:
                sin_sq = 0
            err += np.sqrt(sin_sq)
    elif sin_cos == 'cosine':
        for x in data:
            err += np.sqrt(np.trace(Y.T @ x @ x.T @ Y))
    elif sin_cos == 'sinesq':
        for x in data:
            r = np.min([k,x.shape[1]])
            sin_sq = r - np.trace(Y.T @ x @ x.T @ Y)
            #fixes numerical errors
            if sin_sq < 0:
                sin_sq = 0
            err += sin_sq
    elif sin_cos == 'geodesic':
        for x in data:
            cos = (Y.T @ x @ x.T @ Y)[0][0]
            #fixes numerical errors
            if cos > 1:
                cos = 1
            elif cos < 0:
                cos = 0
            geodesic_distance = np.arccos(np.sqrt(cos))
            err += geodesic_distance
    elif sin_cos == 'l2_med':
        for x in data:
            err += gr_dist(x, Y)

    elif sin_cos == 'zobs':

        idx_class0 = np.where(labels == 0)
        idx_class1 = np.where(labels == 1)

        Y0 = Y[idx_class0]
        Y1 = Y[idx_class1]

        for x in data:

            x0 = x[idx_class0]
            x1 = x[idx_class1]

            #sloppy divide by 0 fix
            x0_norm = np.trace(x0.T @ x0)
            if x0_norm == 0:
                x0_norm = 1
            x1_norm = np.trace(x1.T @ x1)
            if x1_norm == 0:
                x1_norm = 1
            Y0_norm = np.trace(Y0.T @ Y0)
            if Y0_norm == 0:
                Y0_norm = 1
            Y1_norm = np.trace(Y1.T @ Y1)
            if Y1_norm == 0:
                Y1_norm = 1


            
            r0 = np.sqrt(np.trace(Y0.T @ x0 @ x0.T @ Y0)/(Y0_norm*x0_norm))
            r1 = np.sqrt(np.trace(Y1.T @ x1 @ x1.T @ Y1)/(Y1_norm*x1_norm))

            z_class0 = np.arctanh(r0)
            z_class1 = np.arctanh(r1)

            zobs = (z_class0-z_class1) / np.sqrt( 1/(len(idx_class0[0])-3) + 1/(len(idx_class1[0])-3))

            zobs = np.abs(zobs)

            err += zobs
    return err


def flag_mean(data: list, r: int) -> np.array:
    '''
    Calculate the Flag Mean

    Inputs:
        data - list of numpy arrays representing points on Gr(k_i,n)
        r - integer number of columns in flag mean
    Outputs:
        mean - a numpy array representing the Flag Mean of the data
    ''' 
    X = np.hstack(data)
    
    mean = np.linalg.svd(X, full_matrices = False)[0][:,:r]

    return mean


def flag_mean_iteration(data: list, Y0: np.array, weight: float, eps: float = .0000001) -> np.array:
    '''
    Calculates a weighted Flag Mean of data using a weight method for FlagIRLS
    eps = .0000001 for paper examples

    Inputs:
        data - list of numpy arrays representing points on Gr(k_i,n)
        Y0 - a numpy array representing a point on Gr(r,n)
        weight - a string defining the objective function
                    'sine' = flag median
                    'sinsq' = flag mean
        eps - a small perturbation to the weights to avoid dividing by zero
    Outputs:
        Y- the weighted flag mean
    '''
    r = Y0.shape[1]
    
    aX = []
    al = []

    ii=0

    for x in data:
        if weight == 'sine':
            m = np.min([r,x.shape[1]])
            sinsq = m - np.trace(Y0.T @ x @ x.T @ Y0)
            if sinsq < 0:
                sinsq = 0
            al.append((np.sqrt(sinsq)+eps)**(-1/2))
        elif weight == 'cosine':
            cossq = np.trace(Y0.T @ x @ x.T @ Y0)
            if cossq < 0:
                cossq = 0
            al.append((np.sqrt(cossq)+eps)**(-1/2))
        elif weight == 'geodesic':
            sinsq = 1 - Y0.T @ x @ x.T @ Y0
            cossq = Y0.T @ x @ x.T @ Y0
            if sinsq < 0:
                sinsq = 0
            if cossq < 0:
                cossq = 0
            al.append((np.sqrt(sinsq)*np.sqrt(cossq)+ eps)**(-1/2))
        else:
            print('sin_cos must be geodesic, sine or cosine')
        aX.append(al[-1]*x)
        ii+= 1

    Y = flag_mean(aX, r)

    return Y


def irls_flag(data: list, r: int, n_its: int, sin_cos: str, opt_err: str = 'geodesic', 
              init: str = 'random', seed: int = 0, stochastic: int = 0, diff_eps: float = 0.0000000001) -> tuple: 
    '''
    Use FlagIRLS on data to output a representative for a point in Gr(r,n) 
    which solves the input objection function

    Repeats until iterations = n_its or until objective function values of consecutive
    iterates are within 0.0000000001 and are decreasing for every algorithm (except increasing for maximum cosine)

    Inputs:
        data - list of numpy arrays representing points on Gr(k_i,n)
        r - the number of columns in the output
        n_its - number of iterations for the algorithm
        sin_cos - a string defining the objective function for FlagIRLS
                    'sine' = flag median
        opt_err - string for objective function values in err (same options as sin_cos)
        init - string 'random' for random initlalization. 
            otherwise input a numpy array for the inital point
        seed - seed for random initialization, for reproducibility of results
    Outputs:
        Y - a numpy array representing the solution to the chosen optimization algorithm
        err - a list of the objective function values at each iteration (objective function chosen by opt_err)
    '''
    err = []
    n = data[0].shape[0]


    #initialize
    if init == 'random':
        #randomly
        np.random.seed(seed)
        Y_raw = np.random.rand(n,r)-.5
        Y = np.linalg.qr(Y_raw)[0][:,:r]
    elif init == 'data':
        np.random.seed(seed)
        Y = data[np.random.randint(len(data))]
    else:
        Y = init

    err.append(calc_error_1_2(data, Y, opt_err))

    #flag mean iteration function
    #uncomment the commented lines and 
    #comment others to change convergence criteria
    
    itr = 1
    diff = 1
    while itr <= n_its and diff > diff_eps: 
        Y0 = Y.copy()
        if stochastic > 0:
            idx = np.random.randint(len(data), size=stochastic)
            Y = flag_mean_iteration([data[i] for i in idx], Y, sin_cos)
        else:
            Y = flag_mean_iteration(data, Y, sin_cos)
        err.append(calc_error_1_2(data, Y, opt_err))
        if opt_err == 'cosine':
            diff  = err[itr] - err[itr-1]
        else:
            diff  = err[itr-1] - err[itr]
        # diff  = np.abs(err[itr-1] - err[itr])
           
        itr+=1
    


    if diff > 0:
        return Y, err
    else:
        return Y0, err[:-1]


def calc_gradient(data: list, Y0: np.array, weight: str = 'sine', eps: float = .0000001) -> float:
    '''
    Calculates the gradient of a given Y0 and data given an objective function
    Inputs:
        data - list of numpy arrays representing points on Gr(k_i,n)
        Y0 - a representative for a point on Gr(r,n)
        weight - a string defining the objective function
                    'sine' = flag median
    Output:
        grad - numpy array of the gradient

    '''
    k = Y0.shape[1]
    aX = []
    al = []
    for x in data:
        if weight == 'sine':
            r = np.min([k,x.shape[1]])
            sin_sq = r - np.trace(Y0.T @ x @ x.T @ Y0)
            # if sin_sq < eps :
            #     sin_sq = 0
            #     print('converged to datapoint')
            # else:
            #     al.append((max(sin_sq, eps))**(-1/4))
            al.append((max(sin_sq, eps))**(-1/4))
        elif weight == 'cosine':
            cos_sq = np.trace(Y0.T @ x @ x.T @ Y0)
            # if cos_sq < eps :
            #     cos_sq = 0
            #     print('converged to datapoint')
            # else:
            #     al.append((max(cos_sq, eps))**(-1/4))
            al.append((max(cos_sq, eps))**(-1/4))
        elif weight == 'geodesic':
            r = np.min([k,x.shape[1]])
            cos_sq = Y0.T @ x @ x.T @ Y0
            # if cos_sq < eps  or np.abs(cos_sq-1) < eps:
            #     cos_sq = 0
            #     print('converged to datapoint')
            # else:
            #     al.append(((1 - max(cos_sq, eps))**(-1/4))*((max(cos_sq, eps))**(-1/4)))
            al.append(((1 - max(cos_sq, eps))**(-1/4))*((max(cos_sq, eps))**(-1/4)))
        else:
            print('weight must be sine, cosine, or geodesic')
        aX.append(al[-1]*x)

    big_X = np.hstack(aX)
    
    grad = big_X @ big_X.T @ Y0

    return grad


def gradient_descent(data: list, r: int, alpha: float, n_its: int, sin_cos: str, init: str = 'random', seed: int = 0):
    '''
    Runs Grassmannian gradient descent
    Inputs:
        data - list of numpy arrays representing points on Gr(k,n)
        r - integer for the number of columns in the output
        alpha - step size
        n_its - number of iterations
        sin_cos - a string defining the objective function
                    'sine' = flag median
                    'sinsq' = flag mean
        init - string 'random' for random initlalization. 
            otherwise input a numpy array for the inital point
    Outputs:
        Y - a numpy array representing the solution to the chosen optimization algorithm
        err - a list of the objective function values at each iteration (objective function chosen by opt_err)
    '''
    n = data[0].shape[0]

    #initialize
    if init == 'random':
        np.random.seed(seed)
        #randomly
        Y_raw = np.random.rand(n,r)-.5
        Y = np.linalg.qr(Y_raw)[0][:,:r]
    else:
        Y = init

    err = []
    err.append(calc_error_1_2(data, Y, sin_cos))

    for _ in range(n_its):
        Fy = calc_gradient(data,Y,sin_cos)
        # project the gradient onto the tangent space
        G = (np.eye(n)-Y@Y.T)@Fy
        
        [U,S,V] = np.linalg.svd(G)
        cosin = np.diag(np.cos(-alpha*S))
        sin = np.vstack([np.diag(np.sin(-alpha*S)), np.zeros((n-r,r))])
        if cosin.shape[0] == 1:
            Y = Y*V*cosin*V.T+U@sin *V.T
        else:
            Y = Y@V@cosin@V.T+U@sin@V.T
        
        err.append(calc_error_1_2(data, Y, sin_cos))
    return Y, err


def distance_matrix(X: list, C: list, similarity: bool = False, labels: list = None) -> np.array:
    '''
    Calculate a chordal distance matrix for the dataset

    Inputs:
        X- list of numpy arrays for the datset
        C- list of numpy arrays for the elements of the codebook
    Outputs:
        Distances- a numpy array with 
            rows corresponding to elements of the codebook and 
            columns corresponding to data
    '''
    n = len(X)
    m = len(C)
    Distances = np.zeros((m,n))

    if labels is None:
        if similarity:
            sin_cos = 'cosine'
        else:
            sin_cos = 'sine'
    else:
        sin_cos = 'zobs'


    for i in range(m):
        for j in range(n):
            Distances[i,j] = calc_error_1_2([C[i]], X[j], sin_cos, labels)
            
    return Distances


def cluster_purity(X: list, centers: list, labels_true: list, similarity: bool = False, feature_labels: list = None) -> float:
    '''
    Calculate the cluster purity of the dataset

    Inputs:
        X- list of numpy arrays for the dataset
        centers- a list of numpy arrays for the codebook
        labels_true- a list of the true labels
    Outputs:
        purity- a float for the cluster purity
    '''

    #calculate distance matrix
    d_mat = distance_matrix(X, centers, similarity, feature_labels)

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


def lbg_subspace(X: list, epsilon: float, centers: list = [], n_centers: int = 17, 
                 opt_type: str = 'sine', n_its: int = 10, seed: int = 1, r: int = 48, 
                 similarity: bool = False, labels: np.array = None) -> tuple:
    '''
    LBG clustering with subspaces
    
    Inputs:
        X-              a list of numpy array for the dataset
        epsilon-        float for a convergence parameter
        centers-        list of initial centers
        n_centers-      int for the codebook size
        opt_type-       string for the type of LBG clustering
            'sine'          for flag median
            'sinesq'        for flag mean
            'l2_med'        for l2-median
        n_its-          int for the number of iterations
        seed-           int, seed for initial codebook selection
        r-              int, the output is in Gr(r,n)
        similarity-     bool, True to use cosine similarity, otherwise use chordal distance
        labels-         array, labels for the data, only for subspace zobs
    Outputs:
        centers- a list of numpy arrays for the centers
        errors- a list for the the normalized consecutive distortion error at each iteration
        distortions- a list for the cluster distortions at each iteration
    '''
    n_pts = len(X)
    error = 1
    distortions = []

    #init centers if centers aren't provided
    if len(centers) == 0:
        np.random.seed(seed)
        centers = []
        for i in range(n_centers):
            centers.append(X[np.random.randint(n_pts)])

    #calculate distance matrix
    d_mat = distance_matrix(X, centers, similarity, labels)

    #find the closest center for each point
    if similarity:
        index  = np.argmax(d_mat, axis = 0)
    else:
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
                if opt_type == 'sinesq':
                    centers.append(flag_mean([X[i] for i in idx], r))
                elif opt_type == 'eigengene':
                    centers.append(eigengene([X[i] for i in idx], r))
                elif opt_type == 'l2_med':
                    centers.append(l2_median([X[i] for i in idx], .1, r, 1000)[0])
                elif opt_type == 'zobs_eigengene':
                    centers.append(zobs_eigengene([X[i] for i in idx], r, labels))
                else:
                    centers.append(irls_flag([X[i] for i in idx], r, n_its, 'sine', 'sine')[0])

        #calculate distance matrix
        d_mat = distance_matrix(X, centers, similarity, labels)

        #find the closest center for each point
        if similarity:
            index  = np.argmax(d_mat, axis = 0)
        else:
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


