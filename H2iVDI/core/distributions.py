import numpy as np
import scipy.stats as sps
try:
    import pymc
except:
    pymc = None


class Dist:

    def __init__(self, name, bounds=(None, None)):
        self._name = name
        self._bounds = bounds
    
    @property
    def bounds(self):
        return self._bounds
        
    @property
    def name(self):
        return self._name

    def pdf(self, x, data, **kwargs):
        raise RuntimeError("Method must be subclassed as Dist is a base class")

    def transform(self, x):
        raise RuntimeError("Method must be subclassed as Dist is a base class")

    def to_pymc(self):
        raise RuntimeError("Method must be subclassed as Dist is a base class")


class NoDist(Dist):
    """ Uniform distribution
    """

    def __init__(self, name, bounds=(None, None)):
        """ Instanciate a Beta distribution
            Parameters
            ----------
            bounds: tuple
                Bounds
        """

        super().__init__(name, bounds)
    
    def pdf(self, x, data, **kwargs):
        return 1.0

    def to_pymc(self):
        raise RuntimeError("Cannot be used with pyMC")

    def transform(self, x, data):
        return x


class UniformDist(Dist):
    """ Uniform distribution
    """

    def __init__(self, name, bounds=(0.1, 10.0)):
        """ Instanciate a Beta distribution
            Parameters
            ----------
            bounds: tuple
                Bounds
        """

        super().__init__(name, bounds)
        self._loc = self._bounds[0]
        self._scale = self._bounds[1] - self._bounds[0]
    
    def pdf(self, x, data, **kwargs):
        return sps.uniform(self._loc, self._scale).pdf(x)

    def to_pymc(self):
        if pymc is None:
            raise RuntimeError("pymc library not installed")

        return pymc.Uniform(self._name, lower=self._bounds[0], upper=self._bounds[1])

    def transform(self, x, data):
        return self._loc + x * self._scale


class BetaDist(Dist):
    """ Beta distribution
    """

    def __init__(self, name, a=1.0, b=0.0, bounds=(0.1, 10.0)):
        """ Instanciate a Beta distribution
            Parameters
            ----------
            bounds: tuple
                Bounds
        """

        super().__init__(name, bounds)
        self._a = a
        self._b = b
        self._loc = self._bounds[0]
        self._scale = self._bounds[1] - self._bounds[0]
    
    def pdf(self, x, data, **kwargs):
        xp = (x - self._bounds[0]) / (self._bounds[1] - self._bounds[0])
        return sps.beta(a=self._a, b=self._b).pdf(xp)

    def to_pymc(self):
        if pymc is None:
            raise RuntimeError("pymc library not installed")

        return pymc.Beta(name, alpha=self._a, beta=self._b)

    def transform(self, x, data):
        return self._loc + x * self._scale


class BetaScaledDist(Dist):
    """ Beta distribution
    """

    def __init__(self, name, scale=1.0, a=1.0, b=1.0, bounds=(0.2, 5.0)):
        """ Instanciate a Beta distribution
            Parameters
            ----------
            scale: float
                Scale factor
            bounds: tuple
                Bounds
        """

        super().__init__(name, bounds)
        self._scale = scale
        self._a = a
        self._b = b
    
    def pdf(self, x, data, **kwargs):
        xp = (x/self._scale - self._bounds[0]) / (self._bounds[1] - self._bounds[0])
        return sps.beta(a=self._a, b=self._b).pdf(xp)

    def to_pymc(self):
        if pymc is None:
            raise RuntimeError("pymc library not installed")

        return pymc.Beta(name, alpha=self._a, beta=self._b)

    def transform(self, x, data):
        return (self._bounds[0] + x * (self._bounds[1] - self._bounds[0])) * self._scale


class PriorK0N(Dist):
    def __init__(self, name, mu=10.0, sigma=60.0, bounds=(10.0, 70.0)):
        """ Parameters
            ----------
            bounds: tuple
                Bounds
        """

        super().__init__(name, None)
        self._mu = mu
        self._sigma = sigma
        self._bounds = bounds
    
    @property
    def bounds(self):
        return self._bounds
    
    @property
    def proxy_name(self):
        return "k0"

    def depends(self, var):
        return False
    
    def to_pymc(self):
        if self._bounds is not None:
            normal_dist = pymc.Normal.dist(mu=self._mu, sigma=self._sigma)
            k0_dist = pymc.Truncated("k0", normal_dist, lower=self._bounds[0], upper=self._bounds[1])
        else:
            k0_dist = pymc.Normal("k0", mu=self._mu, sigma=self._sigma)
        return k0_dist
    
    def pdf(self, x, data, **kwargs):
        return sps.norm(loc=self._mu, scale=self._sigma).pdf(x)
    
    def transform(self, x, data):
        return x


class LowFroudeQ0dK0Dist(Dist):
    """ Distribution of errors for the LowFroude regression
    """

    def __init__(self, name, reg=(0.537660, 2.414474), loc=0.0, scale=1.2135):
        """ Instanciate the distribution
            Parameters
            ----------
            reg: tuple
                Regression parameters
            loc: float
                Loc (mean) parameter for the gaussian distribution of regression residuals
            scale: float
                Scale factor
        """

        super().__init__(name, None)
        self._reg = reg
        self._loc = loc
        self._scale = scale
    
    def pdf(self, x, data, k0=10.0, **kwargs):
        if not hasattr(data, "_logphiABCbar"):
            data._logphiABCbar = np.mean(5./3. * np.log(np.mean(data.dAr, axis=0)) - 2./3. * np.log(np.mean(data.W, axis=0)) + 0.5 * np.log(np.mean(data.S, axis=0)))
        err = np.log(x/k0) - self._reg[0] * data._logphiABCbar + self._reg[1]
        # print("x=%f, k0=%f, err=%f" % (x, k0, err))
        return sps.norm(loc=self._loc, scale=self._scale).pdf(err)

    def to_pymc(self):
        raise NotImplementedError()
        # if pymc is None:
        #     raise RuntimeError("pymc library not installed")

        # return pymc.Beta(name, alpha=self._a, beta=self._b)

    def transform(self, x, data, h0, k0, q0):
        raise NotImplementedError()
        if not hasattr(data, "_phiABC"):
            data._phiABC = np.mean(5./3. * np.log(np.mean(data.dA, axis=0)) - 2./3. * np.log(np.mean(data.W, axis=0)) + 0.5 * np.log(np.mean(data.S, axis=0)))
        logQdK0 = self._reg[1] + data._phiABC * self._reg[0] + x
        phiABC = np.mean(data.phiABCm)
        return np.exp(logQdK0) * k0


class PriorQdK0LF(Dist):
    def __init__(self, name, reg=(0.9833, 0.0971), err_distrib={"type": "normal", "mu": 0.0, "sigma": 0.106}):
        """ Parameters
            ----------
            reg: tuple
                Regression tuple (slope, intercept)
            bounds: tuple
                Bounds
        """
        super().__init__(name, None)
        self._reg = reg
        self._err_distrib = err_distrib
    
    # def update_from_data(self, data):
    #     #print("Update phip for prior PriorQdK0LF")
    #     #self._phiABCbar = np.mean(np.mean(data.dA, axis=0)*5./3 * og(np.mean(data.W, axis=0)) + 0.5 * np.log(np.mean(data.S, axis=0)))
    #     h0p = np.maximum(0.5, -0.509758 * np.log(np.mean(data.S, axis=0)) - 3.948736)
    #     print("h0p=", h0p)
    #     self._logphip = np.mean(5./3. * np.log(data.W0 * h0p + np.mean(data.dA, axis=0)) - 2./3. * np.log(np.mean(data.W, axis=0)) + 0.5 * np.log(np.mean(data.S, axis=0)))
    #     #self._logphiABCbar = 5./3. * np.log(np.mean(data.dA[:, 0])) - 2./3. * np.log(np.mean(data.W[:, 0])) + 0.5 * np.log(np.mean(data.S[:, 0]))
    #     #print("- dA:", np.mean(data.dA, axis=0))
    #     #print("- W :", np.mean(data.W, axis=0))
    #     #print("- S :", np.mean(data.S, axis=0))
    #     #print("- log(phiABC) :", 5./3. * np.log(np.mean(data.dA, axis=0)) - 2./3. * np.log(np.mean(data.W, axis=0)) + 0.5 * np.log(np.mean(data.S, axis=0)))
    #     #print("- log(phiABCbar) = %f" % self._logphiABCbar)
    
    # @property
    # def proxy_name(self):
    #     return "qdk0LF"

    # def depends(self, var):
    #     if var == "k0":
    #         return True
    #     return False
    
    def to_pymc(self):

        # TODO
        # if not hasattr(data, "_logphiABCbar"):
            # data._logphiABCbar = np.mean(5./3. * np.log(np.mean(data.dAr, axis=0)) - 2./3. * np.log(np.mean(data.W, axis=0)) + 0.5 * np.log(np.mean(data.S, axis=0)))

        if self._err_distrib["type"] == "normal":
            return pymc.Normal("qdk0LF", mu=self._err_distrib["mu"], sigma=self._err_distrib["sigma"])
        elif self._err_distrib["type"] == "beta":
            raise NotImplementedError("Not implemented")
    
    def pdf(self, x, data, k0=10.0, **kwargs):

        if not hasattr(data, "_logphip"):
            print("Sp=", np.mean(data.S, axis=0))
            print("log(Sp)=", np.log(np.mean(data.S, axis=0)))
            h0p = np.maximum(0.5, np.exp(-0.509758 * np.log(np.mean(data.S, axis=0)) - 3.948736))
            print("h0p=", h0p)
            data._logphip = np.mean(5./3. * np.log(np.min(data.W, axis=0) * h0p + np.mean(data._dAr, axis=0)) - 2./3. * np.log(np.mean(data.W, axis=0)) + 0.5 * np.log(np.mean(data.S, axis=0)))
            print("data._logphip=", data._logphip)

        # print("np.log(x)=", np.log(x))
        # print("self._reg[0]=", self._reg[0])
        # print("self._reg[1]=", self._reg[1])
        err = np.log(x / k0) - (self._reg[0] * data._logphip + self._reg[1])
        # print("q0=%f, k0=%f, err=%f" % (x, k0, err))
        if self._err_distrib["type"] == "normal":
            return sps.norm(loc=self._err_distrib["mu"], scale=self._err_distrib["sigma"]).pdf(err)
        elif self._err_distrib["type"] == "beta":
            return sps.beta(a=self._err_distrib["a"], b=self._err_distrib["b"]).pdf((err - self._err_distrib["loc"]) / self._err_distrib["scale"])
    
    def transform(self, x, data, h0, k0):
        #phiABC = np.mean(data.phiABCm)
        logQdK0 = self._reg[1] + self._logphip * self._reg[0] + x
        print("logQdk0=%f, k0=%f, q0=%f" % (logQdK0, k0, np.exp(logQdK0) * k0))
        return np.exp(logQdK0) * k0

def new_distribution(distribution_id, name, **kwargs):
    if distribution_id == "One":
        dist = NoDist(name)
    elif distribution_id == "Uniform":
        dist = UniformDist(name, **kwargs)
    elif distribution_id == "Beta":
        dist = BetaDist(name, **kwargs)
    elif distribution_id == "BetaScaled":
        dist = BetaScaledDist(name, **kwargs)
    elif distribution_id == "PriorK0N":
        dist = PriorK0N(name, **kwargs)
    elif distribution_id == "LowFroudeQ0dK0":
    #     dist = LowFroudeQ0dK0Dist(name, **kwargs)
    # elif distribution_id == "LowFroudeQ0dK0":
        dist = PriorQdK0LF(name, **kwargs)
    else:
        raise ValueError("Wrong distribution ID: %s" % distribution_id)

    return dist