
from .classification import CLSynthesizer
from .interval_regression import IRSynthesizer
from .multilabels import MLSynthesizer
from .ranking import RKSynthesizer


class Synthesizer:
    def __init__(self, name):
        """Synthesizers Wrapper
        Useful to use the FoldsLoader.

        Input:
         - name: of the form 'CL-<n_train>', with <n_train> an integer
        """
        name_split = name.split('-')
        self.problem = name_split[0]
        nb = int(name_split[1])
        # param = name_split[2:]

        if self.problem == 'IR':
            self.loader = IRSynthesizer(nb)
        elif self.problem == 'CL':
            self.loader = CLSynthesizer(nb)
            self.Y = self.loader.Y
        elif self.problem == 'ML':
            self.loader = MLSynthesizer(nb)
            self.k = 1
        elif self.problem == 'RK':
            self.loader = RKSynthesizer(nb)
        else:
            raise ValueError(self.problem + ' is nor "IR", nor "CL", nor "ML".')

    def get_trainset(self, *args, **kwargs):
        return self.loader.get_trainset(*args, **kwargs)

    def get_testset(self, *args, **kwargs):
        return self.loader.get_testset(*args, **kwargs)

    def synthetic_corruption(self, *args, **kwargs):
        return self.loader.synthetic_corruption(*args, **kwargs)