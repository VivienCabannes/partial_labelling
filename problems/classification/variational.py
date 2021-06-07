
import numpy as np


class Variational:
    def __init__(self, weight_computer, method):
        """

        Parameters
        ----------
        weight_computer: weigthing scheme object
        method: specify variational method
        """
        self.computer = weight_computer
        self.method = method.lower()
        if not self.method in ['ac', 'il']:
            raise NotImplementedError('method should be `AC` or `IL`, ' \
                                      + 'but was specified as `' + method + '`.')

    def train(self, x_train, S_train, **kwargs):
        self.computer.set_support(x_train)
        self.computer.train(**kwargs)

        if self.method == 'ac':
            phi = np.asfortranarray(S_train, dtype=np.float)
            phi /= phi.sum(axis=1)[:, np.newaxis]
        elif self.method == 'il':
            phi = np.asfortranarray(S_train, dtype=np.float)
        else:
            raise NotImplementedError('method should be `AC` or `IL`, ' \
                                      + 'but was specified as `' + self.method + '`.')
        self.computer.set_phi(phi)

    def __call__(self, x):
        beta = self.computer.call_with_phi(x)
        idx = beta.argmax(axis=1)
        return idx


class AC(Variational):
    def __init__(self, computer):
        super(AC, self).__init__(computer, 'ac')


class IL(Variational):
    def __init__(self, computer):
        super(IL, self).__init__(computer, 'il')


if __name__=="__main__":
    import os
    import sys

    sys.path.append(os.path.join('..', '..'))
    from weights import RidgeRegressor, Diffusion
    from dataloader import LIBSVMLoader, FoldsGenerator

    computer_il = RidgeRegressor('Gaussian', sigma=10)
    computer_ac = RidgeRegressor('Gaussian', sigma=10)

    met_il = IL(computer_il)
    met_ac = AC(computer_ac)

    loader = LIBSVMLoader('dna')
    x, y = loader.get_trainset()
    # S = loader.synthetic_corruption(y, .6)
    S = loader.skewed_corruption(y, .6, 0)

    floader = FoldsGenerator(x, y, S)

    (x, S), (xt, y) = floader()
    y = y.argmax(axis=1)
    met_il.train(x, S, lambd=1e-4)
    met_ac.train(x, S, lambd=1e-3)
    y_il = met_il(xt)
    y_ac = met_ac(xt)
    print('KRR: ', (y_il == y).mean(), (y_ac == y).mean())

    computer_il = Diffusion(sigma=10)
    computer_ac = Diffusion(sigma=10)

    met_il = IL(computer_il)
    met_ac = AC(computer_ac)

    met_il.train(x, S, lambd=1e-2, mu=1e-4)
    met_ac.train(x, S, lambd=1e-2, mu=1e-4)
    y_il = met_il(xt)
    y_ac = met_ac(xt)
    print('LAP: ', (y_il == y).mean(), (y_ac == y).mean())
