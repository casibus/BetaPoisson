from typing import Optional

from scipy.stats import rv_discrete
from scipy.special import gamma, hyp1f1, gammaln
from scipy.optimize import differential_evolution
import numpy as np
from matplotlib import pyplot as plt

class BetaPoisson(rv_discrete):
    "Beta-Poisson mixture distribution - a bimodal distribution for modelling zero inflated or jittered distributions."
    #def __init__(self, y: Optional[np.ndarray[int]] = None):
    #    self.y = y
    def _pmf(self, x, mu_max, p, q):
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

    def to_minimize(self, params):
        m, p = params
        return -BetaPoisson().loglikelihood(self.y, m, p, q=.5)

if __name__ == "__main__":
    p = BetaPoisson()
    y_sim = p.rvs(8.53, .68, .49, size=1000)
    y = p.pmf(np.arange(20), 8.53, .68, .49)
    plt.hist(y_sim, range=[-.5, 100.5], bins=100, density=True)
    plt.plot(y)
    plt.xlim(-.5, 20.5)
    p.y = y_sim
    res = differential_evolution(p.to_minimize, bounds=[
        [0, 2*np.amax(y_sim)], # the maximal expectation value for each draw is assumed to be max 2-times the max obs.
        [0, 4],
        #[0, 4]
    ], workers=4)
    plt.plot(p.pmf(np.arange(20), *res.x, .5))
    plt.show()
