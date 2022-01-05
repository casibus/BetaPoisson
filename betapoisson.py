from typing import Optional

from scipy.stats import rv_discrete
from scipy.special import gamma, hyp1f1, gammaln
from scipy.optimize import differential_evolution
from typing import List, Tuple
import numpy as np
from matplotlib import pyplot as plt

class BetaPoisson(rv_discrete):
    "Beta-Poisson mixture distribution - a bimodal distribution for modelling zero inflated or jittered distributions."
    def _pmf(self, x: List[int], mu_max:float, p:float, q:float):
        self.mu_max = mu_max
        self.p = p
        self.q = q
        x_factorial = gamma(x+1) #np.array(list(map(np.math.factorial, x)))
        x_factorial[x == 0] = 1
        return mu_max**x * np.exp(-mu_max) * gamma(p+q) * gamma(p+x) /\
               (x_factorial * gamma(p + q + x) * gamma(p)) * hyp1f1(q, p+q+x, mu_max)\

    def _mean(self, mu_max, p, q):
        return self.mean_mu(mu_max, p, q)

    def mean_mu(self, mu_max, p, q):
        return p / (p+q) * mu_max

    def mu_max(self, mean_mu, p, q):
        return mean_mu * (p+q) / p

    def _std(self):
        pass

    def _var(self):
        pass

    def loglikelihood(self, x, mu_max, p, q):
        n = len(x)
        log_x_factorial = gammaln(x+1)
        log_x_factorial[x == 0] = 0
        return n*(gammaln(p+q) - mu_max - gammaln(p)) + np.log(mu_max)*np.sum(x) + np.sum(gammaln(p+x)) +\
            np.sum(np.log(hyp1f1(q, p+q+x, mu_max)) - log_x_factorial - gammaln(p+q+x))

    @staticmethod
    def _to_minimize(params, y):
        m, p = params
        return -BetaPoisson().loglikelihood(y, m, p, q=.5)

    @classmethod
    def fit(cls, y:List, bounds:List[Tuple] = None, workers:int = 4):
        if not bounds:
            bounds = [
            [0, 2 * np.amax(y)],
            # the maximal expectation value for each draw is assumed to be max 2-times the max obs.
            [0, 4],
            # [0, 4]
            ]
        bp = cls()
        bp.y = y
        res = differential_evolution(bp._to_minimize,
                                     bounds=bounds,
                                     workers=workers,
                                     args=(y,))
        return res


if __name__ == "__main__":
    bp = BetaPoisson()
    mu_max = 8.53
    p = .68
    y_sim = bp.rvs(mu_max, p, .5, size=1000)
    y = bp.pmf(np.arange(20), mu_max, p, .5)
    plt.hist(y_sim, range=[-.5, 100.5], bins=100, density=True)
    plt.plot(y)
    plt.xlim(-.5, 20.5)
    res = bp.fit(y = y_sim, workers=1)
    """
    differential_evolution(p.to_minimize, bounds=[
        [0, 2*np.amax(y_sim)], # the maximal expectation value for each draw is assumed to be max 2-times the max obs.
        [0, 4],
        #[0, 4]
    ], workers=4)"""
    plt.plot(bp.pmf(np.arange(20), *res.x, .5))
    plt.show()

    #import pandas
    #pandas.read_sas("C:\Users\Johannes.Nagele\Downloads\LLCP2020XPT\LLCP2020.xpt")
