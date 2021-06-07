"""
Generic implementations for a problem specified by a loss

.. math:: \ell(y, z) = -\psi(z)^\top\phi(y)

See Also
--------
<problem>.IL and <problem>.DF for <problem> being
`classification`, `multilabel`, `ranking` or `regression`.
"""

class IL:
    """
    Generic model for infimum loss implementation with a specific loss.

    See the paper:
    Structured Prediction with Partial Labelling through the Infimum Loss
    """
    def __init__(self, weight_computer):
        """

        Parameters
        ----------
        weight_computer: WeightComputer object
            Computer to compute similarity metric
        """
        self.computer = weight_computer

    def train(self, x_train, S_train, **kwargs):
        self.computer.set_support(x_train)
        self.computer.train(**kwargs)
        self.phi_init = S_train
        self.constraint = self.phi_init

    def __call__(self, x, solve_psi, solve_phi, tol):
        alpha = self.computer(x)
        psi = solve_psi(alpha @ self.phi_init)
        phi = solve_phi(alpha @ psi, self.constraint)
        psi_old = np.zeros(psi.shape, psi.dtype)

        # alternative minimization
        while np.abs(psi_old - psi).max() > tol:
            psi_old = psi
            psi = solve_psi(alpha @ phi)
            phi = solve_phi(alpha @ psi, self.constraint)

        scores = alpha @ phi
        return scores


class DF:
    """
    Generic model for disambiguation framework implementation with a specific loss.

    See the paper:
    Disambiguation of weak supervision with exponential convergence rates
    """
    def __init__(self, computer):
        self.computer = computer

    def train(self, x_train, S_train, tol=1e-3, **kwargs):
        self.computer.set_support(x_train)
        self.computer.train(**kwargs)
        alpha = self.computer(x_train)

        phi = self.disambiguation(alpha, phi_init, phi_init, tol)
        self.computer.set_phi(phi)

    def __call__(self, x, solve_psi, solve_phi, tol):
        scores = self.computer.call_with_phi(x)
        return scores

    @classmethod
    def disambiguation(cls, alpha, phi, constraint, tol):
        psi = cls.solve_psi(alpha @ phi)
        phi = cls.solve_phi(alpha @ psi, constraint)
        psi_old = np.zeros(psi.shape, psi.dtype)

        while np.abs(psi_old - psi).max() > tol:
            psi_old = psi
            psi = cls.solve_psi(alpha @ phi)
            phi = cls.solve_phi(alpha @ psi, constraint)
        return phi

    @staticmethod
    def solve_phi(psi, constraint):
        pass

    @staticmethod
    def solve_psi(phi):
        pass
