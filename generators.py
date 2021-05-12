from river import datasets
import random


class Gaussian(datasets.base.SyntheticDataset):
    def __init__(self, mu, sigma, seed: int = None):
        super().__init__(task=datasets.base.REG, n_features=10)
        self._rng = random.Random(seed)
        self.seed = seed
        self.mu = mu
        self.sigma = sigma

    def __iter__(self):
        while True:
            yield dict(), self._rng.gauss(self.mu, self.sigma)
