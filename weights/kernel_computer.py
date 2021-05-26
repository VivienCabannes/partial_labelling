
import numpy as np


class KernelComputer:
    """
    Kernel computer.

    Useful to compute kernel matrices.

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
    >>>> kernel_computer = KernelComputer('Gaussian', sigma=3)
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
