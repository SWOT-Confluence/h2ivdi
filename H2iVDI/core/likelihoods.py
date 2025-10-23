import numpy as np


class GaussianLikelihood:

    def __init__(self, sigma=0.25):

        self._sigma = sigma

    def loglikelihood_from_cost(self, N, cost):

        return N * np.log(2.0 * np.pi) + N * np.log(self._sigma**2) + 1/self._sigma**2 * cost
    

def new_likelihood(name, **kwargs):

    if name == "gaussian":
        return GaussianLikelihood(**kwargs)