
import numpy as np


class NearestNeighbors:
    def __init__(self, k):
        self.k = k
        
    def set_support(self, x, subsample_rate=0):
        """Set train support for nearest neighbors method.

        Parameters
        ----------
        x : ndarray
            Training set given as a design matrix, of shape (nb_points, input_dim).
        subsample_rate : float, optional, default to 0
            Subsampling rate to compute kernel methods on a subsets of points.
        """
        self.ind = np.random.rand(x.shape[0]) >= subsample_rate
        self.x = x[self.ind].T
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
            similarity matrix alpha(x, x_support).
        """                
        dist = x @ self.x
        dist *= -2
        norm = np.sum(x**2, axis=1)
        dist += norm[:, np.newaxis]
        dist += self.norm
        alpha = dist < np.partition(dist, self.k)[:, self.k:self.k+1]
        return alpha.astype(np.float)