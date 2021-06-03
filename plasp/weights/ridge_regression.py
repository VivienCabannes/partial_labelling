
import numpy as np


class RidgeRegressor:
    """
    Regression weights of kernel Ridge regression

    Useful to try several regularization parameter
    Based on linked between Tikhonov regularization and GSVD

    Examples
    --------
    >>>> import numpy as np
    >>>> kernel_computer = Kernel('Gaussian', sigma=3)
    >>>> krr = RidgeRegressor(kernel_computer, lambd=1e-3)
    >>>> x_support = np.random.randn(50, 10)
    >>>> krr.set_support(x_support, subsample_rate=.1)
    >>>> x = np.random.randn(30, 10)
    >>>> alpha = krr(x)
    """
    def __init__(self, kernel, lambd=None):
        self.kernel = kernel
        self.lambd = lambd

    def set_support(self, x_train, subsample_rate=0):
        """Specified input training data"""
        self.x_train = x_train
        self.ind = np.random.rand(x_train.shape[0]) >= subsample_rate
        self.x_train = x_train[self.ind]
        self.n_train = len(x_train)
        self.kernel.set_support(self.x_train, subsample_rate)

    def update_sigma(self, sigma=None, subsample_rate=0):
        """Setting bandwith parameter

        There should be a call to update lambda after setting the bandwith
        """
        if sigma is not None:
            self.kernel.__init__(self.kernel.kernel, sigma=sigma)
            self.kernel.set_support(self.x_train, subsample_rate=0)
        K = self.kernel.get_k()
        self.w_0, self.v = np.linalg.eigh(K)

    def update_lambda(self, lambd=None):
        """Setting Tikhonov regularization parameter"""
        if lambd is None:
            if self.lambd is None:
                raise ValueError('No specification of regularization parameter')
            lambd = self.lambd
        if not hasattr(self, 'w_0'):
            self.update_sigma()
        w = self.w_0 + self.n_train * lambd
        w **= -1
        self.K_inv = (self.v * w) @ self.v.T

    def __call__(self, x_test):
        """Neighbor computation.

        Parameters
        ----------
        x_test : ndarray
            Points to compute kernel ridge regression weights, of shape (nb_points, input_dim).

        Returns
        -------
        out : ndarray
            Similarity matrix of size (n_train, nb_points) given by kernel ridge regression.
        """
        if not hasattr(self, 'K_inv'):
            self.update_lambda()
        K_x = self.kernel(x_test)
        return K_x.T @ self.K_inv


class Kernel:
    """
    Computation of classical kernels

    Parameters
    ----------
    kernel : {'gaussian'}
        Name of the kernel to use.
    sigma : int, optional
        Parameter for various kernel: standard deviation for Gaussian kernel.

    Examples
    --------
    >>>> import numpy as np
    >>>> x_support = np.random.randn(50, 10)
    >>>> kernel_computer = Kernel('Gaussian', sigma=3)
    >>>> kernel_computer.set_support(x_support, subsample_rate=.1)
    >>>> x = np.random.randn(30, 10)
    >>>> k = kernel_computer(x)
    """

    def __init__(self, kernel, **kwargs):
        self.kernel = kernel.lower()
        if self.kernel == "gaussian":
            self.sigma2 = 2 * (kwargs['sigma'] ** 2)
        if self.kernel == "laplacian":
            self.sigma = kwargs['sigma']
        self._call_method = getattr(self, self.kernel + '_kernel')

    def set_support(self, x, subsample_rate=0):
        """Set train support for kernel method.

        Parameters
        ----------
        x : ndarray
            Training set given as a design matrix, of shape (nb_points, input_dim).
        subsample_rate : float, optional, default to 0
            Subsampling rate to compute kernel methods on a subsets of points.
        """
        self.reset()
        self.ind = np.random.rand(x.shape[0]) >= subsample_rate
        self.x = x[self.ind]

    def __call__(self, x):
        """Kernel computation.

        Parameters
        ----------
        x : ndarray
            Points to compute kernel, of shape (nb_points, input_dim).

        Returns
        -------
        out : ndarray
            kernel matrix k(x, x_support).
        """
        return self._call_method(x)

    def get_k(self):
        """Kernel computations.

        Get kernel matrix on support points.
        """
        return self(self.x)

    def gaussian_kernel(self, x):
        """Gaussian kernel.

        Implement k(x, y) = exp(-norm{x - y}^2 / (2 * sigma2)).
        """
        K = self.x @ x.T
        K *= 2
        if not hasattr(self, "_attr_1"):
            self._attr1 = np.sum(self.x ** 2, axis=1)[:, np.newaxis]
        K -= self._attr1
        K -= np.sum(x ** 2, axis=1)
        K /= self.sigma2
        np.exp(K, out=K)
        return K

    def laplacian_kernel(self, x):
        """Laplacian kernel
        return exp(-norm{x - y} / (sigma))
        sigma = kernel_parameter
        """
        K = self.x @ x.T
        K *= -2
        if not hasattr(self, "_attr_1"):
            self._attri_1 = np.sum(self.x ** 2, axis=1)[:, np.newaxis]
        K += self._attri_1
        K += np.sum(x ** 2, axis=1)
        K[K < 0] = 0
        np.sqrt(K, out=K)
        K /= - self.sigma
        np.exp(K, out=K)
        return K

    def linear_kernel(self, x):
        """Linear kernel.

        Implement k(x, y) = x^T y.
        """
        return self.x @ x.T

    def reset(self):
        """Resetting attributes."""
        if hasattr(self, "_attr_1"):
            delattr(self, "_attr_1")


if __name__=="__main__":
    kernel_computer = Kernel('Gaussian', sigma=3)
    krr = RidgeRegressor(kernel_computer, lambd=1e-3)
    x_support = np.random.randn(50, 10)
    krr.set_support(x_support, subsample_rate=0)
    x = np.random.randn(30, 10)
    alpha = krr(x)
    assert(alpha.shape==(30,50))