import numpy as np


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

        A0 = np.repeat((h * data.W0).reshape((1, -1)), data.A.shape[0], axis=0)
        k = np.repeat(k.reshape((1, -1)), data.A.shape[0], axis=0)

        phi = k * (A0 + data.dA)**(5./3.) * data.W**(-2./3.)

        Qt = phi * data.S**0.5
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
