'''
by Nugyen et al.
'''

from __future__ import division
from pymanopt.manifolds.manifold import Manifold
import numpy as np
# from numpy import trace, zeros, zeros_like, sqrt, eye
# from numpy.random import randn, randint


if not hasattr(__builtins__, "xrange"):
    xrange = range


class NullRangeManifold(Manifold):
    """Base class, template for NullRangeManifold
    with formulas for Hessian and gradient
    once the required operators are defined
    """
    def __init__(self):
        raise NotImplementedError

    @property
    def dim(self):
        return self._dimension
    
    @property
    def codim(self):
        return self._codim

    def __str__(self):
        return "Base null range manifold"

    @property
    def typicaldist(self):
        return np.sqrt(self.dim)

    def dist(self, X, Y):
        """ Geodesic distance. Not implemented
        """
        raise NotImplementedError

    def base_inner_ambient(self, eta1, eta2):
        raise NotImplementedError

    def base_inner_E_J(self, a1, a2):
        raise NotImplementedError
    
    def g(self, X, eta):
        raise NotImplementedError

    def g_inv(self, X, eta):
        raise NotImplementedError
    
    def J(self, X, eta):
        raise NotImplementedError

    def Jst(self, X, a):
        raise NotImplementedError

    def g_inv_Jst(self, X, a):
        return self.g_inv(X, self.Jst(X, a))

    def D_g(self, X, xi, eta):
        raise NotImplementedError

    def christoffel_form(self, X, xi, eta):
        ret = 0.5*self.D_g(X, xi, eta)
        ret += 0.5*self.D_g(X, eta, xi)
        ret -= 0.5*self.contract_D_g(X, xi, eta)
        return ret

    def D_J(self, X, xi, eta):
        raise NotImplementedError
    
    def D_Jst(self, X, xi, a):
        raise NotImplementedError

    def D_g_inv_Jst(self, X, xi, a):
        djst = self.D_Jst(X, xi, a)
        return self.g_inv(
            X, -self.D_g(X, xi, self.g_inv(X, self.Jst(X, a))) + djst)
    
    def contract_D_g(self, X, xi, eta):
        raise NotImplementedError
    
    def inner(self, X, G, H):
        """ Inner product (Riemannian metric) on the tangent space.
        The tangent space is given as a matrix of size mm_degree * m
        """
        # return inner_product_tangent
        return self.base_inner_ambient(self.g(X, G), H)

    def st(self, mat):
        """The split_transpose. transpose if real, hermitian transpose if complex
        """
        raise NotImplementedError

    def J_g_inv_Jst(self, X, a):
        return self.J(X, self.g_inv_Jst(X, a))

    def solve_J_g_inv_Jst(self, X, b, tol=1e-8):
        """ base is use CG. Unlikely to use
        """
        from scipy.sparse.linalg import cg, LinearOperator
        if tol is None:
            tol = self.tol
            
        def Afunc(a):
            return self._vec_range_J(
                self.J_g_inv_Jst(X, self._unvec_range_J(a)))
        A = LinearOperator(
            dtype=float, shape=(self._codim, self._codim), matvec=Afunc)
        res = cg(A, self._vec_range_J(b), tol=tol)
        return self._unvec_range_J(res[0])

    def proj(self, X, U):
        """projection. U is in ambient
        return one in tangent
        """
        return U - self.g_inv_Jst(
            X, self.solve_J_g_inv_Jst(X, self.J(X, U)))

    def proj_g_inv(self, X, U):
        return self.proj(X, self.g_inv(X, U))

    def egrad2rgrad(self, X, U):
        return self.proj_g_inv(X, U)

    def rhess02(self, X, xi, eta, egrad, ehess):
        """ Ehess is the Hessian Vector Product
        """
        return self.inner(
            X, self.ehess2rhess(X, egrad, ehess, xi), eta)
    
    def rhess02_alt(self, X, xi, eta, egrad, ehess_val):
        """ optional
        """
        try:
            g_inv_Jst_solve_J_g_in_Jst_DJ = self.g_inv(
                X, self.Jst(X, self.solve_J_g_inv_Jst(
                    X, self.D_J(X, xi, eta))))
            proj_christoffel = self.proj_g_inv(
                X, self.christoffel_form(X, xi, eta))
            return ehess_val - self.base_inner_ambient(
                g_inv_Jst_solve_J_g_in_Jst_DJ + proj_christoffel, egrad)
        except Exception as e:
            raise(RuntimeError("%s if D_J is not implemeted try rhess02" % e))

    def christoffel_gamma(self, X, xi, eta):
        try:
            g_inv_Jst_solve_J_g_in_Jst_DJ = self.g_inv(
                X, self.Jst(X, self.solve_J_g_inv_Jst(
                    X, self.D_J(X, xi, eta))))
            proj_christoffel = self.proj_g_inv(
                X, self.christoffel_form(X, xi, eta))
            return g_inv_Jst_solve_J_g_in_Jst_DJ + proj_christoffel
        except Exception as e:
            raise(RuntimeError("%s if D_J is not implemeted try rhess02" % e))
    
    def ehess2rhess(self, X, egrad, ehess, H):
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
    
    def retr(self, X, eta):
        """ Calculate 'thin' qr decomposition of X + G
        then add point X
        then do thin lq decomposition
        """
        raise NotImplementedError

    def norm(self, X, eta):
        # Norm on the tangent space is simply the Euclidean norm.
        return np.sqrt(self.inner(X, eta, eta))

    def rand(self):
        # Generate random  point using qr of random normally distributed
        # matrix.
        raise NotImplementedError
    
    def randvec(self, X):
        """Random tangent vector at point X
        """

        """
        U = np.random.randn(self._dim)
        U = U / self.norm(X, U)
        return U
        """
        raise NotImplementedError

    def _rand_ambient(self):
        raise NotImplementedError

    def _rand_range_J(self):
        raise NotImplementedError

    def _vec(self, E):
        """vectorize. This is usually used for sanity test in low dimension
        typically X.reshape(-1). For exampe, we can test J, g by representing
        them as matrices.
        Convenient for testing but dont expect much actual use
        """
        raise NotImplementedError

    def _unvec(self, vec):
        """reshape to shape of matrix - use unvech if hermitian,
        unvecah if anti hermitian. For testing, don't expect actual use
        """
        raise NotImplementedError

    def _vec_range_J(self, a):
        """vectorize an elememt of rangeJ
        a.reshape(-1)
        """
        raise NotImplementedError

    def _unvec_range_J(self, vec):
        raise NotImplementedError

    