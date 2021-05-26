
class DataLoader:
    datasets = []

    def __init__(self, name, center=True, normalized=True):
        if name not in self.datasets:
            raise NameError("Dataset name not valid")
        self.name = name

        self.center = center
        self.normalized = normalized

    def read_trainset(self):
        pass

    def read_testset(self):
        pass

    def get_trainset(self):
        data, labels = self.read_trainset()

        if self.center:
            self.mean = data.mean(axis=0)
            data -= self.mean
        if self.normalized:
            self.std = data.std(axis=0)
            self.std[self.std == 0] = 1
            data /= self.std

        return data, labels

    def get_testset(self):
        data, labels = self.read_testset()

        data -= getattr(self, 'mean', 0)
        data /= getattr(self, 'std', 1)

        return data, labels
