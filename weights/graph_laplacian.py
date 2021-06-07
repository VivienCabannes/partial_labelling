
import numpy as np


class GraphLaplacian:
    """Regression weights given by graph Laplacian

    Attributes
    ----------
    x: ndarray
        Training data of shape (nb_points, input_dim)
    """

    def __init__(self, sigma):
        """

        Parameters
        ----------
        sigma: float
            Bandwidth parameter
        """

        self.sigma = sigma

    def set_support(self, x):
        """Set train support

        Parameters
        ----------
        x : ndarray
            Training set given as a design matrix, of shape (nb_points, input_dim).
        """
        self.x = x

    def kernel(self, x1, x2):
        W = x1 @ x2.T
        W *= 2
        W -= np.sum(x1 ** 2, axis=1)[:, np.newaxis]
        W -= np.sum(x2 ** 2, axis=1)
        W /= 2 * (self.sigma ** 2)
        np.exp(W, out=W)
        return W

    def __call__(self, x):
        """Weighting scheme computation.

        Parameters
        ----------
        x : ndarray
            Points to compute nearest neighbors, of shape (nb_points, input_dim).

        Returns
        -------
        out : ndarray
            Similarity matrix alpha[i,j] of shape (nb_points, n_train).
        """
        W_ul = self.kernel(x, self.x)
        W_uu = self.kernel(x, x)
        D = W_uu.sum(axis=1)
        D += W_ul.sum(axis=1)
        W_uu -= np.diag(D)

        alpha = np.linalg.solve(W_uu, W_ul)
        alpha *= -1
        return alpha

    def train(self):
        pass

    def set_phi(self, phi):
        self.phi = phi

    def call_with_phi(self, x):
        alpha = self(x)
        beta = alpha @ self.phi
        return beta


if __name__=="__main__":
    glap = GraphLaplacian(sigma=1)
    x_support = np.random.randn(50, 10)
    glap.set_support(x_support)
    x = np.random.randn(30, 10)
    alpha = glap(x)
    assert(alpha.shape==(30,50))
