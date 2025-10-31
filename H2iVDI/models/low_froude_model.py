import numpy as np
import pymc as pm


class LowFroudeModel:
    def __init__(self, data, h0=1.0, h1=None, k0=10.0, k1=None):
        self._data = data
        self._k0 = k0
        if k1 is None:
            k1 = np.ones(data.A.shape[1])
        self._k1 = k1
        self._h0 = h0
        if h1 is None:
            h1 = np.ones(data.A.shape[1])
        self._h1 = h1

    @property
    def name(self):
        return "lowfroude"

    def set_h0(self, h0):
        self._h0 = h0

    def set_k0(self, k0):
        self._k0 = k0
        
    def compute_lowfroude_discharge(self):
        data = self._data
        h = self._h0 * self._h1
        k = self._k0 * self._k1

        W = np.nanmin(data._W, axis=0)

        dA = data._dA
        W = data._W
        S = data._S
        W0 = np.nanmin(data.W, axis=0)
        
        A0 = np.repeat((h * W0).reshape((1, -1)), data.H.shape[0], axis=0)
        k = np.repeat(k.reshape((1, -1)), data.H.shape[0], axis=0)

        phi = k * (A0 + dA)**(5./3.) * W**(-2./3.)

        Qt = phi * S**0.5
        return Qt

    def compute_slopes_old(self, q0, reach=None):
        data = self._data
        h = self._h0 * self._h1
        k = self._k0 * self._k1

        A0 = np.repeat((h * data.W0).reshape((1, -1)), data.A.shape[0], axis=0)
        k = np.repeat(k.reshape((1, -1)), data.A.shape[0], axis=0)

        phi = k * (A0 + data.dA)**(5./3.) * data.W**(-2./3.)

        Q1 = phi * data.S**0.5
        Q1m = np.mean(Q1, axis=0)
        print("Q1m=", Q1m)
        Q1mt = np.repeat(Q1m.reshape((1, -1)), data.A.shape[0], axis=0)
        Q1t = Q1 / Q1mt

        #q0 = qdk0 * k0
        Qt = q0 * Q1t
        print("Q1[2]m=", np.mean(Qt))

        Sest = (Qt / phi)**2

        return Sest

    def compute_slopes(self, Qin, reach=None):
        data = self._data
        h = self._h0 * self._h1
        k = self._k0 * self._k1

        A0 = np.repeat((h * data.W0).reshape((1, -1)), data.A.shape[0], axis=0)
        k = np.repeat(k.reshape((1, -1)), data.A.shape[0], axis=0)

        phi = k * (A0 + data.dA)**(5./3.) * data.W**(-2./3.)
        Qxt = np.repeat(Qin.reshape((-1, 1)), data.A.shape[1], axis=1)

        Sest = (Qxt / phi)**2

        return Sest
    
    def cost(self, Qin, data):
        S = self.compute_slopes(Qin)
        return np.sum((np.ravel(S) - np.ravel(data.S))**2)

    def calibrate(self, Q):

        valid_subset = np.isfinite(np.mean(Q, axis=1))
        if np.all(valid_subset == False):
            return 1

        data = self._data
        khat = np.zeros(data.x.size)
        h0hat = np.zeros(data.x.size)

        if not hasattr(self._data, "_dA"):
            self._data.compute_dA()

        for ir in range(0, data.x.size):

            dArc = data._dA[valid_subset, ir]
            Wrc = data._W[valid_subset, ir]
            Src = data._S[valid_subset, ir]
            W0 = np.min(Wrc)

            model = pm.Model()
            with model:

                # Priors for unknown parameters
                h0 = 0.07 + pm.Beta('h0p', alpha=1, beta=3.7673) * 15.68
                k = pm.Uniform('k', lower=5.0, upper=80.0)

                A0 = h0 * W0

                # h = (A0 + dArc) / Wrc
                mu = k * (A0 + dArc) ** (5. / 3.) * Wrc ** (-2. / 3.) * Src ** 0.5

                sigma = 0.3 * mu

                qest = pm.Normal('q', mu=mu, sigma=sigma, observed=Q[valid_subset, ir])
                
                try:
                    step = pm.Metropolis()
                    trace = pm.sample(10000, step=step, tune=2000, progressbar=True, random_seed=0)
                except:
                    print("A0=", A0)
                    print("dArc=", dArc)
                    print("Wrc=", Wrc)
                    print("Src=", Src)
                    print("q=", Q)
                    raise

                khat[ir] = np.median(np.ravel(trace["posterior"]["k"]))
                h0hat[ir] = np.median(np.ravel(0.07 + trace["posterior"]["h0p"] * 15.68))
                print("khat=%f, h0hat=%f" % (khat[ir], h0hat[ir]))
                A0 = h0hat[ir] * np.min(Wrc)
                qtest = khat[ir] * (A0 + dArc) ** (5. / 3.) * Wrc ** (-2. / 3.) * Src ** 0.5

        self._k0 = np.mean(khat)
        self._k1 = khat / np.mean(khat)
        self._h0 = np.mean(h0hat)
        self._h1 = h0hat / np.mean(h0hat)

        return 0

    def calibrate_rerun(self, Q):

        valid_subset = np.isfinite(np.mean(Q, axis=1))
        if np.all(valid_subset == False):
            return 1

        data = self._data
        self._alpha = np.zeros(data.x.size)
        self._beta = np.zeros(data.x.size)
        self._n = np.zeros(data.x.size)

        if not hasattr(self._data, "_dA"):
            self._data.compute_dA()

        for ir in range(0, data.x.size):

            dArc = data._dA[valid_subset, ir]
            Wrc = data._W[valid_subset, ir]
            Src = data._S[valid_subset, ir]
            W0 = np.min(Wrc)

            model = pm.Model()
            with model:

                # Priors for unknown parameters
                alpha = pm.Uniform('alpha', lower=5.0, upper=80.0)
                beta = pm.Normal('beta', mu=0.0, sigma=0.03)

                A0 = self._h0 * self._h1 * W0

                h = (A0 + dArc) / Wrc
                mu = alpha * h**beta * (A0 + dArc) ** (5. / 3.) * Wrc ** (-2. / 3.) * Src ** 0.5

                sigma = 0.3 * mu

                qest = pm.Normal('q', mu=mu, sigma=sigma, observed=Q[valid_subset, ir])
                
                try:
                    step = pm.Metropolis()
                    trace = pm.sample(10000, step=step, tune=2000, progressbar=False, random_seed=0)
                except:
                    print("A0=", A0)
                    print("dArc=", dArc)
                    print("Wrc=", Wrc)
                    print("Src=", Src)
                    print("q=", Q)
                    raise

                self._alpha[ir] = np.median(np.ravel(trace["posterior"]["alpha"]))
                self._beta[ir] = np.median(np.ravel(trace["posterior"]["beta"]))
                self._n[ir] = 1.0 / np.mean(self._alpha[ir] * h**self._beta[ir])

        return 0
        
    def compute_rerun_discharge(self):
        data = self._data
        h = self._h0 * self._h1

        W = np.nanmin(data._W, axis=0)

        dA = data._dA
        W = data._W
        S = data._S
        W0 = np.nanmin(data.W, axis=0)
        
        A0 = np.repeat((h * W0).reshape((1, -1)), data.H.shape[0], axis=0)
        alpha = np.repeat(self._alpha.reshape((1, -1)), data.H.shape[0], axis=0)
        beta = np.repeat(self._beta.reshape((1, -1)), data.H.shape[0], axis=0)

        h = (A0+dA) / W
        phi = alpha*h**beta * (A0 + dA)**(5./3.) * W**(-2./3.)

        Qt = phi * S**0.5

        return Qt
