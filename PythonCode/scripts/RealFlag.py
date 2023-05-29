'''
by Nugyen et al.

Function names changed by Nathan Mankovich 
for usability with pymanopt RTR
'''

from __future__ import division
import numpy as np
import numpy.linalg as la
from numpy import trace, zeros, zeros_like
from numpy.random import randn
from scipy.linalg import expm, expm_frechet, null_space
from scipy.sparse.linalg import cg, LinearOperator
from NullRangeManifold import NullRangeManifold

if not hasattr(__builtins__, "xrange"):
    xrange = range


def _calc_dim(dvec):
    s = 0
    for i in range(1, len(dvec)):
        for j in range(i):
            s += dvec[i]*dvec[j]
    return s
    

class RealFlag(NullRangeManifold):
    """Class for a Real Flag manifold
    Block matrix Y with Y.T @ Y = I
    dvec is a vector defining the blocks of Y
    dvec of size p
    Y of dimension n*d

    n = dvec.sum()
    d = dvec[1:].sum()

    Metric is defined by a sets of parameters alpha of size (p-1)p

    Parameters
    ----------
    dvec     : vector defining the block size
    alpha    : array of size (p-1)p, p = dvec.shape[0]
               Defining a metric on the Flag manifold
               alpha  > 0

    """
    
    def __init__(self, dvec, alpha=None,
                 log_stats=False,
                 log_method=None):
        self.dvec = np.array(dvec)
        self.n = dvec.sum()
        self.d = dvec[1:].sum()
        self._dimension = _calc_dim(dvec)
        self._codim = self.d * self.n - self._dimension
        self._point_layout = 1
        cs = dvec[:].cumsum() - dvec[0]
        self._g_idx = dict((i+1, (cs[i], cs[i+1]))
                           for i in range(cs.shape[0]-1))
        p = self.dvec.shape[0]-1
        self.p = p
        if alpha is None:
            self.alpha = np.full((p, p+1), fill_value=1/2)
            self.alpha[:, 0] = 1
        else:
            self.alpha = alpha
        self.log_stats = log_stats
        if log_method is None:
            self.log_method = 'trust-krylov'
        elif log_method.lower() in ['trust-ncg', 'trust-krylov']:
            self.log_method = log_method.lower()
        else:
            raise(ValueError(
                'log method must be one of trust-ncg or trust-krylov'))
        self.log_gtol = None
        self.lbd = self.make_lbd()
                        
    def inner_product(self, X, Ba, Bb=None):
        """ Inner product (Riemannian metric) on the tangent space.
        The tangent space is given as a matrix of size mm_degree * m
        """
        gdc = self._g_idx
        alpha = self.alpha
        p = self.dvec.shape[0]-1
        s2 = 0
        if Bb is None:
            Bb = Ba
        for rr in range(p, 0, -1):
            br, er = gdc[rr]
            ss = trace(alpha[rr-1, 0] * Ba[:, br:er] @ Bb[:, br:er].T)
            s2 += ss

            for jj in range(1, p+1):
                bj, ej = gdc[jj]
                ss = trace(
                    (alpha[rr-1, jj] - alpha[rr-1, 0]) * (
                        (Ba[:, br:er].T @ X[:, bj:ej]) @
                        (X[:, bj:ej].T @ Bb[:, br:er])))
                s2 += ss
        return s2
    
    @property
    def dim(self):
        return self._dimension
    
    @property
    def codim(self):
        return self._codim

    def __str__(self):
        self._name = "Real flag manifold dimension vector=(%s) alpha=%s" % (
            self.dvec, str(self.alpha))
        return self._name

    @property
    def typicaldist(self):
        return np.sqrt(sum(self._dimension))

    def base_inner_ambient(self, eta1, eta2):
        return trace(eta1.T @ eta2)

    def base_inner_E_J(self, a1, a2):
        raise trace(a1.T @ a2)
    
    def g(self, X, omg):
        gdc = self._g_idx
        alpha = self.alpha
        ret = zeros_like(omg)
        p = self.p
        for rr in range(1, p+1):
            br, er = gdc[rr]
            ret[:, br:er] = alpha[rr-1, 0]*omg[:, br:er]
            for jj in range(1, p+1):
                bj, ej = gdc[jj]
                ret[:, br:er] += (alpha[rr-1, jj]-alpha[rr-1, 0]) *\
                    (X[:, bj:ej] @ (X[:, bj:ej].T @ omg[:, br:er]))
        
        return ret
        
    def g_inv(self, X, omg):
        gdc = self._g_idx
        alpha = self.alpha
        ret = zeros_like(omg)
        p = self.p
        for rr in range(1, p+1):
            br, er = gdc[rr]
            ret[:, br:er] = 1/alpha[rr-1, 0]*omg[:, br:er]
            for jj in range(1, p+1):
                bj, ej = gdc[jj]
                ret[:, br:er] += (1/alpha[rr-1, jj]-1/alpha[rr-1, 0]) *\
                    (X[:, bj:ej] @ (X[:, bj:ej].T @ omg[:, br:er]))
        
        return ret
    
    def J(self, X, eta):
        ret = {}
        ph = self.dvec
        gidx = self._g_idx
        alpha = self.alpha
        p = self.dvec.shape[0]-1
        for r in range(p, 0, -1):
            if ph[r] == 0:
                continue
            r_g_beg, r_g_end = gidx[r]
            for s in range(p, 0, -1):
                if ph[s] == 0:
                    continue
                s_g_beg, s_g_end = gidx[s]
                if r == s:
                    ret[r, r] = alpha[r-1, r]*X[:, r_g_beg:r_g_end].T @\
                        eta[:, r_g_beg:r_g_end]

                elif s > r:
                    ret[r, s] = eta[:, r_g_beg:r_g_end].T @\
                        X[:, s_g_beg:s_g_end]

                    ret[r, s] += X[:, r_g_beg:r_g_end].T @\
                        eta[:, s_g_beg:s_g_end]
        return ret

    def Jst(self, X, a):
        ret = zeros_like(X)
        alpha = self.alpha
        for r, s in a:
            br, er = self._g_idx[r]
            if r == s:
                ret[:, br:er] += alpha[r-1, r] * X[:, br:er] @ a[r, r]
            else:
                bs, es = self._g_idx[s]
                ret[:, br:er] += X[:, bs:es] @ a[r, s].T
                ret[:, bs:es] += X[:, br:er] @ a[r, s]
        return ret

    def g_inv_Jst(self, X, a):
        ret = np.zeros_like(X)
        alpha = self.alpha
        for r, s in a:
            br, er = self._g_idx[r]
            if r == s:
                ret[:, br:er] += X[:, br:er] @ a[r, r]
            else:
                bs, es = self._g_idx[s]
                ret[:, br:er] += 1/alpha[r-1, s] * X[:, bs:es] @ a[r, s].T
                ret[:, bs:es] += 1/alpha[s-1, r] * X[:, br:er] @ a[r, s]
        return ret

    def D_g(self, X, xi, eta):
        gdc = self._g_idx
        alpha = self.alpha
        ret = zeros_like(eta)
        p = self.p
        for rr in range(1, p+1):
            br, er = gdc[rr]
            for jj in range(1, p+1):
                bj, ej = gdc[jj]
                ret[:, br:er] += (alpha[rr-1, jj]-alpha[rr-1, 0]) *\
                    xi[:, bj:ej] @ (X[:, bj:ej].T @ eta[:, br:er])
                ret[:, br:er] += (alpha[rr-1, jj]-alpha[rr-1, 0]) *\
                    X[:, bj:ej] @ (xi[:, bj:ej].T @ eta[:, br:er])
        return ret

    def christoffel_form(self, X, xi, eta):
        ret = 0.5*self.D_g(X, xi, eta)
        ret += 0.5*self.D_g(X, eta, xi)
        ret -= 0.5*self.contract_D_g(X, xi, eta)
        return ret

    def D_J(self, X, xi, eta):
        """ Derivatives of J
        """
        ret = {}
        ph = self.dvec
        gidx = self._g_idx
        alpha = self.alpha
        p = self.p
        for r in range(p, 0, -1):
            if ph[r] == 0:
                continue
            r_g_beg, r_g_end = gidx[r]
            for s in range(p, 0, -1):
                if ph[s] == 0:
                    continue
                s_g_beg, s_g_end = gidx[s]
                if r == s:
                    ret[r, r] = alpha[r-1, r]*xi[:, r_g_beg:r_g_end].T @\
                        eta[:, r_g_beg:r_g_end]

                elif s > r:
                    ret[r, s] = eta[:, r_g_beg:r_g_end].T @\
                        xi[:, s_g_beg:s_g_end]

                    ret[r, s] += xi[:, r_g_beg:r_g_end].T @\
                        eta[:, s_g_beg:s_g_end]
        return ret
    
    def D_Jst(self, X, xi, a):
        ret = np.zeros_like(xi)

        alpha = self.alpha
        for r, s in a:
            br, er = self._g_idx[r]
            if r == s:
                ret[:, br:er] += alpha[r-1, r] * xi[:, br:er] @ a[r, r]
            else:
                bs, es = self._g_idx[s]
                ret[:, br:er] += xi[:, bs:es] @ a[r, s].T
                ret[:, bs:es] += xi[:, br:er] @  a[r, s]
        return ret

    def D_g_inv_Jst(self, Y, xi, a):
        ret = zeros_like(Y)
        alpha = self.alpha
        for r, s in a:
            br, er = self._g_idx[r]
            if r == s:
                ret[:, br:er] += xi[:, br:er] @ a[r, r]
            else:
                bs, es = self._g_idx[s]
                ret[:, br:er] += 1/alpha[r-1, s] * xi[:, bs:es] @ a[r, s].T
                ret[:, bs:es] += 1/alpha[s-1, r] * xi[:, br:er] @ a[r, s]
        return ret
    
    def contract_D_g(self, X, xi, eta):
        ret = zeros_like(eta)
        alpha = self.alpha
        gidx = self._g_idx
        p = self.p
        for r in range(1, p+1):
            br, er = gidx[r]
            for jj in range(1, p+1):
                bj, ej = gidx[jj]
                ret[:, br:er] += (alpha[jj-1, r] - alpha[jj-1, 0])*(
                    eta[:, bj:ej] @ (xi[:, bj:ej].T @ X[:, br:er]) +
                    xi[:, bj:ej] @ (eta[:, bj:ej].T @ X[:, br:er]))
        return ret
    
    def st(self, mat):
        """The split_transpose. transpose if real, hermitian transpose if complex
        """
        return mat.T

    def solve_J_g_inv_Jst(self, X, b):
        alf = 1/self.alpha
        a = dict()
        for r in range(1, alf.shape[1]):
            a[r, r] = alf[r-1, r] * b[r, r]
            for s in range(r+1, alf.shape[1]):
                a[r, s] = 1/(alf[r-1, s] + alf[s-1, r])*b[r, s]
        return a
    
    def projection(self, X, U):
        """projection. U is in ambient
        return one in tangent
        """
        ret = zeros_like(U)
        alpha = self.alpha
        p = self.p
        for tt in range(1, p+1):
            bt, et = self._g_idx[tt]
            ret[:, bt:et] = U[:, bt:et] -\
                X[:, bt:et] @ (X[:, bt:et].T @ U[:, bt:et])
            for uu in range(1, p+1):
                if uu == tt:
                    continue
                bu, eu = self._g_idx[uu]
                ft = alpha[uu-1, tt] / (alpha[uu-1, tt] + alpha[tt-1, uu])
                ret[:, bt:et] -= ft*X[:, bu:eu:] @ (
                    U[:, bu:eu:].T @ X[:, bt:et] +
                    X[:, bu:eu:].T @ U[:, bt:et])
        return ret

    def proj_g_inv(self, X, U):
        ret = zeros_like(U)
        alpha = self.alpha
        p = self.p
        for tt in range(1, p+1):
            bt, et = self._g_idx[tt]
            ret[:, bt:et] = 1/alpha[tt-1, 0] *\
                (U[:, bt:et] -
                 X @ (X.T @ U[:, bt:et]))
            for uu in range(1, p+1):
                if uu == tt:
                    continue
                bu, eu = self._g_idx[uu]
                ft = 1 / (alpha[uu-1, tt] + alpha[tt-1, uu])
                ret[:, bt:et] += ft*X[:, bu:eu:] @ (
                    X[:, bu:eu:].T @ U[:, bt:et] -
                    U[:, bu:eu:].T @ X[:, bt:et])
        return ret

    def euclidean_to_riemannian_gradient(self, X, U):
        return self.proj_g_inv(X, U)

    def rhess02(self, X, xi, eta, egrad, ehess):
        egcoef = np.zeros_like(eta)
        ph = self.dvec
        alpha = self.alpha
        gidx = self._g_idx
        p = ph.shape[0]-1
        ehess_val = self.base_inner_ambient(ehess, eta)
        
        for tt in range(1, p+1):
            bt, et = gidx[tt]
            egcoef[:, bt:et] += X[:, bt:et] @ (xi[:, bt:et].T @ eta[:, bt:et])
                
            for uu in range(1, p+1):
                if uu != tt:
                    bu, eu = gidx[uu]
                    """
                    ft = alpha[uu-1, tt]/(alpha[uu-1, tt] + alpha[tt-1, uu])
                    egcoef[bt:et, :] += ft*(
                        xi[bt:et, :] @ eta[bu:eu, :].T +
                        eta[bt:et, :] @ xi[bu:eu, :].T) @ W[bu:eu, :]
                    """
                    ft2 = 0.5*(alpha[tt-1, uu]+alpha[uu-1, tt] -
                               alpha[tt-1, 0]+alpha[uu-1, 0]) /\
                        (alpha[tt-1, uu]+alpha[uu-1, tt])
                    egcoef[:, bt:et] += ft2*X[:, bu:eu] @ (
                        eta[:, bu:eu].T @ xi[:, bt:et] +
                        xi[:, bu:eu].T @ eta[:, bt:et])

        for tt in range(1, p+1):
            bt, et = self._g_idx[tt]
            for jj in range(1, p+1):
                bj, ej = self._g_idx[jj]
                ftt = 0.5*(alpha[jj-1, 0]+alpha[tt-1, 0] -
                           alpha[jj-1, tt]-alpha[tt-1, jj])

                omg_t_j = ftt*(
                    eta[:, bj:ej] @ xi[:, bj:ej].T +
                    xi[:, bj:ej] @ eta[:, bj:ej].T) @ X[:, bt:et]

                egcoef[:, bt:et] += 1/alpha[tt-1, 0] *\
                    (omg_t_j - X @ (X.T @ omg_t_j))
                for uu in range(1, p+1):
                    if uu == tt:
                        continue
                    bu, eu = self._g_idx[uu]
                    ft = 1 / (alpha[uu-1, tt] + alpha[tt-1, uu])
                    # omg_u = np.zeros_like(W[bu:eu, :])
                    ftu = 0.5*(alpha[jj-1, 0]+alpha[uu-1, 0] -
                               alpha[jj-1, uu]-alpha[uu-1, jj])
                    egcoef[:, bt:et] += ft*(ftt-ftu) *\
                        X[:, bu:eu] @ X[:, bu:eu:].T @ (
                            eta[:, bj:ej] @ xi[:, bj:ej].T @ X[:, bt:et] +
                            xi[:, bj:ej] @ eta[:, bj:ej].T @ X[:, bt:et])
                    
        return ehess_val - trace(egrad.T @ egcoef)

    def euclidean_to_riemannian_hessian(self, X, egrad, ehess, H):
        """ Convert Euclidean into Riemannian Hessian.
        ehess is the Hessian product on the ambient space
        egrad is the gradient on the ambient space
        """
        first = ehess
        a = self.J(X, self.g_inv(X, egrad))
        rgrad = self.proj_g_inv(X, egrad)
        second = self.D_g(X, H, self.g_inv(X, egrad))
        aout = self.solve_J_g_inv_Jst(X, a)
        third = self.proj(X, self.D_g_inv_Jst(X, H, aout))
        fourth = self.christoffel_form(X, H, rgrad)
        return self.proj_g_inv(X, (first - second) + fourth) - third
        
    def retraction(self, X, eta):
        """ Calculate 'thin' qr decomposition of X + G
        then add point X
        then do thin lq decomposition
        """
        u, _, vh = la.svd(X+eta, full_matrices=False)
        return u @ vh

    def norm(self, X, eta):
        # Norm on the tangent space is simply the Euclidean norm.
        return np.sqrt(self.inner(X, eta, eta))

    def random_point(self):
        # Generate random  point using qr of random normally distributed
        # matrix.
        O, _ = np.linalg.qr(randn(
            self.n, self.d))
        return O
    
    def random_tangent_vector(self, X):
        """Random tangent vector at point X
        """
        U = self.proj(X, self._rand_ambient())
        return U / self.norm(X, U)

    def zero_vector(self, X):
        return zeros(X.shape)

    def _rand_ambient(self):
        return randn(self.n, self.d)

    def _rand_range_J(self):
        p = self.p
        out = {}
        dv = self.dvec
        for r in range(p, 0, -1):
            for s in range(p, r-1, -1):
                out[r, s] = randn(dv[r], dv[s])
        return out

        return randn(self.d, self.d)

    def _vec(self, E):
        """vectorize. This is usually used for sanity test in low dimension
        typically X.reshape(-1). For exampe, we can test J, g by representing
        them as matrices.
        Convenient for testing but dont expect much actual use
        """
        raise E.reshape(-1)

    def _unvec(self, vec):
        """reshape to shape of matrix - use unvech if hermitian,
        unvecah if anti hermitian. For testing, don't expect actual use
        """
        return vec.reshape(self.n, self.d)

    def _vec_range_J(self, a):
        """vectorize an elememt of rangeJ
        a.reshape(-1)
        """
        ret = zeros(self.codim)
        start = 0
        for r, s in sorted(a, reverse=True):
            tp = a[r, s].reshape(-1)
            ret[start:start+tp.shape[0]] = tp
            start += tp.shape[0]
        return ret

    def _unvec_range_J(self, vec):
        dout = {}
        start = 0
        p = self.p
        dv = self.dvec
        for r in range(p, 0, -1):
            for s in range(p, r-1, 0):
                vlen = dv[r]*dv[s]
                dout[r, s] = vec[start:start+vlen].reshape(
                    dv[r], dv[s])
                start += vlen
        return dout

    def exp(self, X, eta):
        """
        Use this only if 
        self.alpha[:, 1] = self.alpha[0, 1]
        self.alpha[:, 0] = self.alpha[0, 0]
        This is Stiefel geodesics
        """
        assert(
            1e-14 > np.max(np.abs(np.abs(self.alpha[:, 0] - self.alpha[:, 0]))))
        assert(
            1e-14 > np.max(np.abs(np.abs(self.alpha[:, 1] - self.alpha[:, 1]))))
        
        K = eta - X @ (X.T @ eta)
        Xp, R = la.qr(K)
        alf = self.alpha[0, 1]/self.alpha[0, 0]
        A = X.T @eta
        x_mat = np.bmat([[2*alf*A, -R.T],
                         [R, np.zeros((self.d, self.d))]])
        return np.array(
            np.bmat([X, Xp]) @ expm(x_mat)[:, :self.d] @ expm((1-2*alf)*A))

    def _dist(self, X, Y):
        """This function is tentative. It only works
        for two points sufficiently close, and it is not provable
        to be the length minimizing distance
        Use at your own risk
        """
        lg = self.log(X, Y, show_steps=False, init_type=0)
        return self.norm(X, lg)

    def euclidean_dist(self, X, Y):
        """ Euclidean distance. Useful to compare two
        elememt
        """
        YTX = Y.T@X
        return np.sqrt(2*(np.sum(self.lbd * self.lbd) - np.trace(
            (YTX*self.lbd[None, :])@(YTX.T*self.lbd[None, :]))))

    def make_lbd(self):
        dd = self.dvec.shape[0]

        def coef(a, dd):
            if dd % 2 == 0:
                if a < dd // 2 - 1:
                    return - (dd // 2) + a + 1
                else:
                    return - (dd // 2) + a + 2
            else:
                if a < (dd-1)//2:
                    return -(dd-1)//2 + a
                else:
                    return -(dd-1)//2 + a + 1

        dsum = self.dvec[1:].cumsum()
        lbd = np.concatenate([np.ones(self.dvec[a+1])*coef(a, dd)
                              for a in range(dsum.shape[0])])
        # return .5 + lbd / lbd.sum()
        return lbd    

    def to_tangent_space(self, X, X1, show_steps=False, init_type=0):
        """
        Use this only if 
        self.alpha[:, 1] = self.alpha[0, 1]
        self.alpha[:, 0] = self.alpha[0, 0]
        This is Stiefel geodesics
        Only use init_type 0 now = may change in the future
        """
        assert(
            1e-14 > np.max(np.abs(np.abs(self.alpha[:, 0] - self.alpha[:, 0]))))
        assert(
            1e-14 > np.max(np.abs(np.abs(self.alpha[:, 1] - self.alpha[:, 1]))))
        if init_type != 0:
            print("Will init with zero vector. Other options are not yet available")

        alf = self.alpha[0, 1]/self.alpha[0, 0]        
        d = self.dvec[1:].sum()
        sqrt2 = np.sqrt(2)
        
        def getQ():
            """ algorithm: find a basis in linear span of Y Y1
            orthogonal to Y
            """
            u, s, v = np.linalg.svd(
                np.concatenate([X, X1], axis=1), full_matrices=False)
            k = (s > 1e-14).sum()
            good = u[:, :k]@v[:k, :k]
            qs = null_space(X.T@good)
            Q, _ = np.linalg.qr(good@qs)
            return Q
        
        # Q, s, _ = la.svd(Y1 - Y@Y.T@Y1, full_matrices=False)
        # Q = Q[:, :np.sum(np.abs(s) > 1e-14)]
        Q = getQ()
        k = Q.shape[1]
        p = self.p
        lbd = self.lbd

        def asym(mat):
            return 0.5*(mat - mat.T)
        
        def vec(A, R):
            # for A, take all blocks [ij with i > j]
            lret = []
            for r in range(1, p+1):
                gdc = self._g_idx                
                if r not in gdc:
                    continue
                br, er = gdc[r]
                for s in range(r+1, p+1):
                    if s <= r:
                        continue
                    bs, es = gdc[s]
                    lret.append(A[br:er, bs:es].reshape(-1)*sqrt2)

            lret.append(R.reshape(-1))
            return np.concatenate(lret)

        def unvec(avec):
            A = np.zeros((d, d))
            R = np.zeros((k, d))
            gdc = self._g_idx
            be = 0
            for r in range(1, p+1):
                if r not in gdc:
                    continue
                br, er = gdc[r]
                for s in range(r+1, p+1):
                    if s <= r:
                        continue
                    bs, es = gdc[s]
                    dr = er - br
                    ds = es - bs
                    A[br:er, bs:es] = (avec[be: be+dr*ds]/sqrt2).reshape(dr, ds)
                    A[bs:es, br:er] = - A[br:er, bs:es].T
                    be += dr*ds
            R = avec[be:].reshape(k, d)
            return A, R

        XQ = np.array(np.bmat([X, Q]))
        # X2 = XQ.T@X1@X1.T@XQ
        X2 = XQ.T@(X1*lbd[None, :])@X1.T@XQ
        
        def dist(v):
            #  = (dist0a(v) - d)*2
            alf = self.alpha[0, 1] / self.alpha[0, 0]
            A, R = unvec(v)
            x_mat = np.array(
                np.bmat([[2*alf*A, -R.T], [R, zeros((k, k))]]))
            exh = expm(x_mat)
            ex = expm((1-2*alf)*A)
            Mid = (ex*lbd[None, :])@ex.T
            return (- trace(X2@exh[:, :d]@Mid@exh[:, :d].T))

        def jac(v):
            alf = self.alpha[0, 1] / self.alpha[0, 0]
            gdc = self._g_idx
            A, R = unvec(v)
            x_mat = np.array(
                np.bmat([[2*alf*A, -R.T], [R, zeros((k, k))]]))
            exh = expm(x_mat)
            ex = expm((1-2*alf)*A)

            blk = np.zeros_like(exh)
            blk[:d, :] = (ex*lbd[None, :])@ex.T@exh[:, :d].T
            blkA = (lbd[:, None]*ex.T)@exh[:, :d].T@X2@exh[:, :d]

            fexh = 2*expm_frechet(x_mat, blk@X2)[1]
            fex = 2*expm_frechet((1-2*alf)*A, blkA)[1]

            for r in range(1, p+1):
                if r not in gdc:
                    continue
                br, er = gdc[r]            
                fexh[br:br, br:br] = 0
                fex[br:br, br:br] = 0

            return vec(
                (1-2*alf)*asym(fex) + 2*alf*asym(fexh[:d, :d]),
                fexh[d:, :d] - fexh[:d, d:].T)    
        
        def make_vec(xi):
            return vec(X.T@xi, Q.T@xi)

        def hessp(v, xi):
            dlt = 1e-8
            return (jac(v+dlt*xi) - jac(v))/dlt

        def conv_to_tan(A, R):
            return X@A + Q@R

        from scipy.optimize import minimize
        # A0, R0 = make_init()
        # x0 = vec(A0, R0)
        adim = (self.dvec[1:].sum()*self.dvec[1:].sum() -
                (self.dvec[1:]*self.dvec[1:]).sum()) // 2
        tdim = d*k + adim

        x0 = np.zeros(tdim)
        
        def printxk(xk):
            print(la.norm(jac(xk)), dist(xk))

        if show_steps:
            callback = printxk
        else:
            callback = None
        res = {'fun': np.nan, 'x': np.zeros_like(x0),
               'success': False,               
               'message': 'minimizer exception'}
        try:
            if self.log_gtol is None:
                res = minimize(dist, x0, method=self.log_method,
                               jac=jac, hessp=hessp, callback=callback)
            else:
                res = minimize(dist, x0, method=self.log_method,
                               jac=jac, hessp=hessp, callback=callback,            
                               options={'gtol': self.log_gtol})
        except Exception:
            pass
        
        stat = [(a, res[a]) for a in res.keys() if a not in ['x', 'jac']]
        A1, R1 = unvec(res['x'])
        if self.log_stats:
            return conv_to_tan(A1, R1), stat
        else:
            return conv_to_tan(A1, R1)    
    