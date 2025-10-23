# import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo

from H2iVDI.H2iVDI_ext import solve_standard_step_multi
from .model import Model

class SWST3Model(Model):

    def __init__(self, data, h0=1.0, h1=None, k0=25.0, k1=None, bathy_model="h1", kch=None):
        self._data = data
        self._h0 = h0
        if h1 is None:
            h1 = np.ones(data.H.shape[1])
        elif isinstance(h1, float) or isinstance(h1, int):
            h1 = np.ones(data.H.shape[1]) * h1
        self._h1 = h1
        self._k0 = k0
        if k1 is None:
            k1 = np.ones(data.H.shape[1])
        elif isinstance(k1, float) or isinstance(k1, int):
            k1 = np.ones(data.H.shape[1]) * k1
        self._k1 = k1
        self._bathy_model = bathy_model
        self._kch = kch

    @property
    def name(self):
        return "swst3"

    def solve(self, Qin):

        h = self._h0 * self._h1
        k = self._k0 * self._k1
        if self._kch is not None:
            k = np.stack((self._kch, k), axis=0)
        else:
            k = np.stack((k, k), axis=0)
        # print("x=", self._data.x)
        # print("h=", h)
        # print("k=", k)
        Hout = self._data.H[:, -1].copy()
        # choice = input()
        # Qin1 = Qin[0:2]
        # Hout1 = Hout[0:2]
        # print("Qin1=", Qin1)
        # print("Hout1=", Hout1)
        # H = solve_standard_step_multi(self._data.x, np.ones(self._data.x.size), self._data.He, self._data.We, k, h, Qin1, Hout1)
        # print("FINAL:H=", H)
        # choice = input()
        H = solve_standard_step_multi(self._data.x, np.ones(self._data.x.size), self._data.He, self._data.We, k, h, Qin, Hout)

        return H

    def cost(self, Qin, data):
        H = self.solve(Qin)
        residuals = np.ravel(H - data.H)
        cost = np.sum(residuals[np.isfinite(np.ravel(data.H))]**2)
        if np.isnan(cost):
            cost = 1e+99
        if cost < 1e-18:
            print(H)
            print(data.H)
            choice = input()
        return cost

    def set_data(self, data):
        self._data = data

    def set_h0(self, h0):
        self._h0 = h0
        if self._bathy_model == "lf":
            href = h0
            lowp = np.argmin(self._data.H[:, 0])
            Dref = (self._data.W[lowp, 0] * np.sqrt(self._data.S[lowp, 0]))**(3./5.)
            D = (self._data.W[lowp, :] * np.sqrt(self._data.S[lowp, :]))**(-3./5.)
            h = href * Dref * D
            self._h1 = h / np.mean(h)

    def set_k0(self, k0):
        self._k0 = k0

    def set_kch(self, kch):
        self._kch = kch

    def compute_lowfroude_discharge(self):
        data = self._data
        h = self._h0 * self._h1
        #b = data.H0r - h
        K = self._k0 * self._k1

        #W = np.repeat(data.Wr.reshape((1, -1)), data.H.shape[0], axis=0)
        A0 = np.repeat((data.We[0, :] * h).reshape((1, -1)), data.H.shape[0], axis=0)
        #A = (data.H - np.repeat(b.reshape((1, -1)), data.H.shape[0], axis=0)) * W
        A = A0 + data._dAr
        K = np.repeat(K.reshape((1, -1)), data.H.shape[0], axis=0)

        phi = K * A**(5./3.) * data._Wr**(-2./3.)

        Qlf = phi * data.S**0.5
        # if np.any(Qlf > 1e+5):
        #     print("Qlf overflow:")
        #     print("- A: %f %f" % (np.min(np.ravel(A)), np.max(np.ravel(A))))
        #     print("- W: %f %f" % (np.min(np.ravel(data.We[0, :])), np.max(np.ravel(data.We[0, :]))))
        #     print("- S: %f %f" % (np.min(np.ravel(S)), np.max(np.ravel(S))))
        #     plt.plot(data.S)
        #     plt.show()
        #     # choice = input()

        return Qlf

    def compute_optim_qin(self, q0_bounds):

        def minimize_fun(x, model, Qin1):

            Qin = x * Qin1
            return model.cost(Qin, model._data)

        Qlf = self.compute_lowfroude_discharge()
        Qin = np.nanmean(Qlf, axis=1)
        Qin1 = Qin / np.mean(Qin)

        res = spo.minimize_scalar(minimize_fun, bounds=q0_bounds, args=(self, Qin1))
        Qin = res.x * Qin1

        return Qin