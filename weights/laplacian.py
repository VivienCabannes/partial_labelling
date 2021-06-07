
import numpy as np
from scipy.linalg import eigh


class Diffusion:
    """
    Regression weights of kernel Laplacian kernel regularization

    Notes
    -----
    This implementation has not been optimized in term of memory.
    Memory scales in O(p^2 nd) while it could scale in O(nd + p^2).

    Examples
    --------
    >>> import numpy as np
    >>> x_support = np.random.randn(50, 10)
    >>> lap = Diffusion()
    >>> lap.set_support(x_support)
    >>> lap.update_sigma(1, p=20)
    >>> lap.update_mu(1 / len(x_support))
    >>> lap.update_psi(lambd=1e-3)
    >>> x = np.random.randn(30, 10)
    >>> alpha = lap(x)
    """
    def __init__(self, sigma=None, lambd=None, psi=None, mu=None,
                 p=None, full=False, nl=None):
        """

        Parameters
        ----------
        sigma: float
            Bandwidth parameter for the Gaussian kernel
        lambd: float, optional
            Tikhonov regularization parameter
        psi: lambda function, optional
            Filter function
        mu: float, optional
            Regularization parameter for GSVD
        p: int, optional
            Subsampling parameter for low-rank approximation
        full: bool, optional
            Specifies if we use full representation or low-rank representation
            to compute the empirical risk minimizer (default is False).
        nl: int, optional
            Specifies if covariance should be computed only on the first `nl` training points
        """
        self.kernel = GaussianKernel(sigma)

        if lambd is not None:
            self.Tikhonov = True
            if psi is not None:
                raise ValueError('Filter and Tikhonov regularization ' \
                                 + 'can not be specified simultaneously')
            self.lambd = lambd
        else:
            self.Tikhonov = False
            self.psi = psi

        self.mu = mu
        if (p is not None) and full:
            raise NotImplementedError('`p` and `full` can not be specified simultaneously.')
        self.p = p
        self.full = full
        self.nl = nl

    def set_support(self, x_train):
        """Specified input training data"""
        self.n_train = len(x_train)
        self.kernel.set_support(x_train)

    def update_sigma(self, sigma=None, full=None, p=None, nl=None):
        """Setting bandwith parameter

        This should be call after setting the support `x_train`.
        There should be a call to update lambda after setting the bandwith
        """
        if sigma is not None:
            self.kernel.sigma = sigma
            self.kernel.reset()
        self.kernel.set_K()
        self.kernel.set_SZ()

        # Catching arguments
        if full is not None:
            self.full = full
            if self.full:
                self.p = None
        if p is not None:
            self.p = p
        if nl is not None:
            self.nl = nl
        if (self.p is not None) and self.full:
            raise NotImplementedError('`p` and `full` can not be specified simultaneously.')

        # Full representation
        if self.full:
            self.kernel.set_ST()
            self.kernel.set_ZZ()
            self.kernel.set_TZ()
            self.kernel.set_TT()

            ST = self.kernel.ST
            TZ = self.kernel.TZ

            if self.nl is None:
                self.A = ST.T @ ST
                self.A /= self.n_train
            else:
                self.A = ST.T[..., :self.nl] @ ST[:self.nl]
                self.A /= self.nl

            self.B_0 = TZ @ TZ.T
            self.B_0 /= self.n_train

        # Small representation
        else:
            SS = self.kernel.K
            SZ = self.kernel.SZ

            if self.nl is None:
                self.A = SS[:self.p, ...] @ SS[..., :self.p]
                self.A /= self.n_train
            else:
                self.A = SS[:self.p, :self.nl] @ SS[:self.nl, :self.p]
                self.A /= self.nl

            self.B_0 = SZ[:self.p] @ SZ[:self.p].T
            self.B_0 /= self.n_train

    def update_mu(self, mu=None, mu_numerical=10e-7, Tikhonov=False):
        """Setting GSVD regularization parameter"""

        if not hasattr(self, 'B_0'):
            self.update_sigma()

        if mu is not None:
            self.mu = mu
        if self.mu is None:
            raise ValueError('GSVD regularization has not been specified.')

        if self.full:
            self.B = self.B_0 + self.mu * self.kernel.TT
            self.B += mu_numerical * np.eye(self.B.shape[0])
        else:
            self.B = self.B_0 + self.mu * self.kernel.K[:self.p,:self.p]
            self.B += mu_numerical * np.eye(self.B.shape[0])

        if Tikhonov:
            self.Tikhonov = True

        # compute GSVD once and try many filter functions after
        if not self.Tikhonov:
            self.v, self.e = eigh(self.A, self.B)

    def update_psi(self, nl=None, psi=None, lambd=None):
        """Setting Filter function

        Parameters
        ----------
        nl: int, optional
            Number of labelled data among `x_train`, default is `len(x_train)`
        """

        if not hasattr(self, 'B'):
            self.update_mu()

        if nl is None:
            nl = self.nl
        if nl is None:
            nl = self.n_train

        if (psi is not None) and (lambd is not None):
            raise ValueError('Filter and Tikhonov regularization ' \
                             + 'can not be specified simultaneously')

        if psi is not None:
            self.Tikhonov = False
            self.psi = psi

        if lambd is not None:
            self.Tikhonov = True
            self.lambd = lambd

        # full representation
        if self.full:
            b = self.kernel.ST.T[..., :nl] / nl
        # small representation
        else:
            b = self.kernel.K[:self.p, :nl] / nl

        if self.Tikhonov:
            # much faster implementation than GSVD
            self.c = np.linalg.solve(self.A + self.lambd * self.B, b)
        else:
            self.c = (self.e * self.psi(self.v)) @ (self.e.T @ b)

    def __call__(self, x_test):
        """Diffusion scheme computation.

        Parameters
        ----------
        x_test : ndarray
            Points to compute kernel ridge regression weights, of shape (nb_points, input_dim).

        Returns
        -------
        out : ndarray
            Similarity matrix of size (nb_points, n_train) given by kernel Laplacian regularization.
        """
        if not hasattr(self, 'c'):
            self.train()

        if self.full:
            T_x = self.kernel.get_ST(x_test)
            return T_x @ self.c
        else:
            K_x = self.kernel.get_SS(x_test)[:self.p]
            return K_x.T @ self.c

    def train(self, sigma=None, full=None, p=None, n_cov=None, nl=None,
              mu=None, psi=None, lambd=None):
        self.update_sigma(sigma=sigma, full=full, p=p, nl=n_cov)
        if lambd is not None:
            self.Tikhonov = True
        self.update_mu(mu=mu)
        self.update_psi(nl=nl, psi=psi, lambd=lambd)

    def set_phi(self, phi):
        self.c_beta = self.c @ phi

    def call_with_phi(self, x):
        if self.full:
            T_x = self.kernel.get_ST(x)
            return T_x @ self.c_beta
        else:
            K_x = self.kernel.get_SS(x)[:self.p]
            return K_x.T @ self.c_beta


class GaussianKernel:
    """
    Computation of Gaussian kernel and its derivatives
    """

    def __init__(self, sigma=None):
        self.sigma = sigma

    def set_support(self, x):
        """Set train support for kernel method.

        Parameters
        ----------
        x : ndarray
            Training set given as a design matrix, of shape (nb_points, input_dim).
        """
        self.reset()
        self.x = x
        self.n_train = len(x)

    def get_SS(self, x=None):
        """Gaussian kernel.

        .. math:: k(x, y) = exp(-norm{x - y}^2 / (2 * sigma2)).

        Parameters
        ----------
        x: ndarray
            Design matrix of shape (nb_points, input_dim).

        Returns
        -------
        K: ndarray
            Gram matrix of size (n_train, nb_points), similarity between
            training points and testing points.
        """
        if x is None:
            x = self.x
        K = self.x @ x.T
        K *= 2
        if not hasattr(self, "_attr_1"):
            self._attr1 = np.sum(self.x ** 2, axis=1)[:, np.newaxis]
        K -= self._attr1
        K -= np.sum(x ** 2, axis=1)
        K /= 2 * (self.sigma ** 2)
        np.exp(K, out=K)
        return K

    def set_K(self):
        if not hasattr(self, 'K'):
            self.K = self.get_SS()

    def get_SZ(self, x=None, SS=None, reshape=True):
        """First derivative of the Gaussian kernel

        Returns
        -------
        SZ: ndarray
            Array of size (nb_points, d * n_train) or (nb_points, n_train, d)
            SZ[i, j*m] = :math:`\partial_{1,m}` k(x_train[j], x[i])
        """
        if x is None:
            x = self.x
        if SS is None:
            SS = self.get_SS(x).T

        SZ = np.tile(SS[...,np.newaxis], (1, 1, self.x.shape[1]))
        # diff[i,j,k] = x[i,k] - self.x[j,k]
        diff = x[:, np.newaxis, :] - self.x[np.newaxis, ...]
        diff /= self.sigma**2
        # SZ[i, j, k] = (x[i, k] - self.x[j, k]) * k(x[i], self.x[j])
        SZ *= diff

        if reshape:
            n, d = x.shape
            # return SZ.reshape(n, -1, order='F') # slower than the following
            SZ_reshape = np.empty((n, self.n_train*d), SZ.dtype)
            for i in range(d):
                SZ_reshape[:, i*self.n_train:(i+1)*self.n_train] = SZ[..., i]
            return SZ_reshape
        return SZ

    def set_SZ(self):
        if not hasattr(self, 'SZ'):
            self.set_K()
            self.SZ = self.get_SZ(SS=self.K)

    # Matrices for exact ERM computations

    def get_ST(self, x=None, SS=None, SZ=None):
        """Matrix based on derivatives of the Gaussian kernel

        Based on T = [S, Z].
        """
        if x is None:
            x = self.x
        if SS is None:
            SS = self.get_SS(x).T
        if SZ is None:
            SZ = self.get_SZ(x, SS=SS)
        n, d = x.shape
        ST = np.zeros((n, self.n_train*(d+1)), dtype=np.float)
        ST[:, :self.n_train] = SS
        ST[:, self.n_train:] = SZ
        return ST

    def set_ST(self):
        if not hasattr(self, 'ST'):
            self.set_SZ()
            self.ST = self.get_ST(SS=self.K, SZ=self.SZ)


    def get_ZZ(self, x=None, SS=None, reshape=True):
        """Double derivative of the Gaussian kernel

        Returns
        -------
        ZZ: ndarray
            Array of size (n_train * d, nb_points * d) or (n_train, nb_points, d, d)
            ZZ[i*k, j*m] = :math:`\partial_{1,k}\partial_{2,m}` k(x[i], x_train[j])
        """
        if x is None:
            x = self.x
        if SS is None:
            SS = self.get_SS(x).T
        n, d = x.shape
        ZZ = np.tile(SS[...,np.newaxis, np.newaxis], (1, 1, d, d,))
        # diff[i,j,k] = x[i,k] - self.x[j,k]
        diff = x[:, np.newaxis, :] - self.x[np.newaxis, ...]
        # prod_diff[i,j,k,l] = diff[i,j,l]*diff[i,j,k] = (x[i,l] - self.x[j,l]) * (x[i,k] - self.x[j,k])
        prod_diff = diff[:,:, np.newaxis, :]*diff[:,:,:,np.newaxis]
        prod_diff /= self.sigma**4
        prod_diff *= -1
        for i in range(d):
            prod_diff[:, :, i, i] += 1 / (self.sigma**2)
        ZZ *= prod_diff
        if reshape:
            # return ZZ.transpose((0, 2, 1, 3)).reshape(n * d, self.n_train * d, order='F') # slower
            ZZ_reshape = np.empty((n*d, self.n_train*d), ZZ.dtype)
            for i in range(d):
                for j in range(i):
                    ZZ_reshape[n*i:n*(i+1), self.n_train*j:self.n_train*(j+1)] = ZZ[..., i, j]
                    ZZ_reshape[n*j:n*(j+1), self.n_train*i:self.n_train*(i+1)] = ZZ[..., j, i]
                ZZ_reshape[n*i:n*(i+1), self.n_train*i:self.n_train*(i+1)] = ZZ[..., i, i]
            return ZZ_reshape
        return ZZ

    def set_ZZ(self):
        if not hasattr(self, 'ZZ'):
            self.set_K()
            self.ZZ = self.get_ZZ(SS=self.K)

    def get_TZ(self, x=None, SS=None, SZ=None, ZZ=None):
        if x is None:
            x = self.x
        if SS is None:
            SS = self.get_SS(x).T
        if SZ is None:
            SZ = self.get_SZ(x, SS)
        if ZZ is None:
            ZZ = self.get_ZZ(x, SS)
        n, d = x.shape
        TZ = np.zeros((n*(d+1), self.n_train*d), dtype=np.float)
        TZ[:n,:] = SZ
        TZ[n:,:] = ZZ
        return TZ

    def set_TZ(self):
        if not hasattr(self, 'TZ'):
            self.set_K()
            self.set_SZ()
            self.set_ZZ()
            self.TZ = self.get_TZ(SS=self.K, SZ=self.SZ, ZZ=self.ZZ)

    def get_TT(self, x=None, SS=None, SZ=None, ZZ=None):
        if x is None:
            x = self.x
        else:
            raise NotImplementedError('Implementation was not finished for TT')
        if SS is None:
            SS = self.get_SS(x).T
        if SZ is None:
            SZ = self.get_SZ(x)
        if ZZ is None:
            ZZ = self.get_ZZ(x)
        n, d = x.shape
        TT = np.zeros((n*(d+1), self.n_train*(d+1)), dtype=np.float)
        TT[:n, :self.n_train] = SS
        TT[:n, self.n_train:] = SZ
        TT[n:, :self.n_train] = SZ.T
        TT[n:, self.n_train:] = ZZ
        return TT

    def set_TT(self):
        if not hasattr(self, 'TT'):
            self.set_K()
            self.set_SZ()
            self.set_ZZ()
            self.TT = self.get_TT(SS=self.K, SZ=self.SZ, ZZ=self.ZZ)

    def reset(self):
        """Resetting attributes."""
        atts = ['_attr_1', 'K', 'SZ', 'ST', 'ZZ', 'TZ', 'TT']
        for att in atts:
            if hasattr(self, att):
                delattr(self, att)


if __name__=="__main__":
    x_support = np.random.randn(50, 10)
    lap = Diffusion()
    lap.set_support(x_support)
    lap.update_sigma(1, p=20)
    lap.update_sigma(1)
    lap.update_mu(1 / len(x_support))
    lap.update_psi(lambd=1e-3)
    x = np.random.randn(30, 10)
    alpha = lap(x)
    assert(alpha.shape==(30,50))
