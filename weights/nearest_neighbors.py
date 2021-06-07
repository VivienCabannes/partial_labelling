
import numpy as np


class NearestNeighbors:
    """Regression weights given by nearest neighbors

    Attributes
    ----------
    x: ndarray
        Training data of shape (nb_points, input_dim)

    Examples
    --------
    >>> import numpy as np
    >>> nn = NearestNeighbors(k=20)
    >>> x_support = np.random.randn(50, 10)
    >>> nn.set_support(x_support)
    >>> x = np.random.randn(30, 10)
    >>> alpha = nn(x)
    """

    def __init__(self, k):
        """

        Parameters
        ----------
        k: int
            Number of neighbors to consider
        """

        self.k = k

    def set_support(self, x):
        """Set train support for nearest neighbors method.

        Parameters
        ----------
        x : ndarray
            Training set given as a design matrix, of shape (nb_points, input_dim).
        """

        self.x = x.T
        self.norm = np.sum(x**2, axis=1)[np.newaxis,:]

    def __call__(self, x):
        """Neighbor computation.

        Parameters
        ----------
        x : ndarray
            Points to compute nearest neighbors, of shape (nb_points, input_dim).

        Returns
        -------
        out : ndarray
            Similarity matrix alpha[i,j] = 1 if the train point self.x[j] is among the k nearest
            neighbors of the test point x[j].
        """

        dist = x @ self.x
        dist *= -2
        norm = np.sum(x**2, axis=1)
        dist += norm[:, np.newaxis]
        dist += self.norm
        alpha = dist < np.partition(dist, self.k)[:, self.k:self.k+1]
        return alpha.astype(np.float)

    def train(self):
        pass

    def set_phi(self, phi):
        self.phi = phi

    def call_with_phi(self, x):
        alpha = self(x)
        beta = alpha @ self.phi
        return beta


if __name__=="__main__":
    nn = NearestNeighbors(k=20)
    x_support = np.random.randn(50, 10)
    nn.set_support(x_support)
    x = np.random.randn(30, 10)
    alpha = nn(x)
    assert(alpha.shape==(30,50))
