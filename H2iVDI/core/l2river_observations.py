import logging
import matplotlib.pyplot as plt
import numpy as np
import piecewise_regression as pwr
import scipy.stats as sps
import tqdm

from .geometry import curve_fit_lin2, curve_fit_lin2_pwr


class L2RiverScaleObservations:

    def __init__(self, nt=None, nx=None):
        self._logger = logging.getLogger("H2iVDI")

        if nt is None or nx is None:
            self._t = None
            self._dates = None
            self._x = None
            self._H = None
            self._W = None
            self._S = None
            self._K = None
            self._A = None
            self._Q = None
            self._He = None
            self._We = None
        else:
            self._t = np.repeat(np.datetime64("1900-01-01"), nt)
            self._dates = None
            self._x = np.ones(nx) * np.nan
            self._H = np.ones((nt, nx)) * np.nan
            self._W = np.ones((nt, nx)) * np.nan
            self._S = np.ones((nt, nx)) * np.nan
            self._K = None
            self._A = None
            self._Q = None
            self._He = None
            self._We = None

    @property
    def nt(self):
        if self._H is None:
            return 0
        else:
            return self._H.shape[0]

    @property
    def nx(self):
        if self._H is None:
            return 0
        else:
            return self._H.shape[1]

    @property
    def t(self):
        return self._t

    @property
    def dates(self):
        return self._dates

    @property
    def x(self):
        return self._x

    @property
    def H(self):
        return self._H

    @property
    def W(self):
        return self._W

    @property
    def S(self):
        return self._S

    @property
    def He(self):
        return self._He

    @property
    def We(self):
        return self._We

    @property
    def Wr(self):
        return self._Wr

    @property
    def dAr(self):
        return self._dAr

    def copy(self):

        new_scale_obs = L2RiverScaleObservations()
        for variable in ["t", "dates", "x", "H", "W", "S", "K", "A", "Q", "He", "We"]:
            values = getattr(self, "_%s" % variable)
            if values is not None:
                setattr(new_scale_obs, "_%s" % variable, values.copy())

        return new_scale_obs

    def compute_dA(self):
        """ Compute dA (flow area above lowest cross-section)
        """

        if self._H is None:
            raise RuntimeError("Cannot compute dA without any observation")

        self._dA = np.zeros_like(self._H)
        for ix in range(0, self.nx):
            sorted_indices = np.argsort(self._H[:, ix])
            Hs = self._H[sorted_indices, ix]
            Ws = self._W[sorted_indices, ix]
            for its in range(1, self.nt):
                Wm = 0.5 * (self._W[sorted_indices[its-1], ix] + self._W[sorted_indices[its], ix])
                dH = self._H[sorted_indices[its], ix] - self._H[sorted_indices[its-1], ix]
                self._dA[sorted_indices[its], ix] = self._dA[sorted_indices[its-1], ix] + dH * Wm

    def compute_effective_sections(self, nlevels=3):
        """ Compute dA (flow area above lowest cross-section)
        """

        if self._H is None:
            raise RuntimeError("Cannot compute effective without any observation")

        self._He = np.ones((3, self.x.size)) * np.nan
        self._We = np.ones((3, self.x.size)) * np.nan
        self._Wr = np.ones_like(self._H) * np.nan
        self._dAr = np.ones_like(self._H) * np.nan
        for r in tqdm.tqdm(range(0, self.H.shape[1])):

            self._logger.debugL2("- Compute effective section for reach %i/%i" % (r+1, self.H.shape[1]))
            Hs = self._H[:, r]
            Ws = self._W[:, r]
            isort = np.argsort(Hs)
            Hs = Hs[isort]
            Ws = Ws[isort]

            if Hs.size == 1:

                self._logger.debugL2("  - 1 observation, using rectangular section")
                Hi = np.ones(3) * Hs[0]
                Wi = np.ones(3) * Ws[0]

            elif Hs.size == 2:

                self._logger.debugL2("  - %s observations, try linear regression")

                # Try linear regression fit
                Hi, Wi = self._linear_regression_fit_(Hs, Ws)

            else:

                # Try piecewise regression fit
                self._logger.debugL2("  - %i observations, try piecewise fit" % Hs.size)
                Hi, Wi = self._piecewise_regression_fit_(Hs, Ws)

                if Hi is None:

                    # Try linear regression fit
                    self._logger.debugL2("  - Try linear regression fit")
                    Hi, Wi = self._linear_regression_fit_(Hs, Ws)

            if np.any(Wi < 0.1):
                print(Wi)
                raise RuntimeError("Wi < 0")

            self._He[:, r] = Hi
            self._We[:, r] = Wi

            # Compute reconstructed (form effective sections) widths and "dry" flow area
            self._Wr[:, r] = np.interp(Hs, Hi, Wi)
            self._dAr[isort[0], r] = 0.0
            for it in range(1, isort.size):
                dH = Hs[it] - Hs[it-1]
                Wm = 0.5 * (self._Wr[it, r] + self._Wr[it-1, r])
                self._dAr[isort[it], r] = self._dAr[isort[it-1], r] + dH * Wm

        # self.H0 = self.Hl[0, :]
        # self.W0 = self.Wl[0, :]

    def spatial_selection(self, selection):

        # Select values in 1-dimensional variables
        for variable in ["x"]:
            values = getattr(self, "_%s" % variable)
            if values is not None:
                setattr(self, "_%s" % variable, values[selection])

        # Select values in 2-dimensional variables
        for variable in ["H", "W", "S", "K", "A", "Q", "He", "We"]:
            values = getattr(self, "_%s" % variable)
            if values is not None:
                setattr(self, "_%s" % variable, values[:, selection])

    def spatial_mean(self):

        # Select values in 1-dimensional variables
        for variable in ["x"]:
            values = getattr(self, "_%s" % variable)
            if values is not None:
                setattr(self, "_%s" % variable, np.mean(values).reshape((1,)))

        # Select values in 2-dimensional variables
        for variable in ["H", "W", "S", "K", "A", "Q", "He", "We"]:
            values = getattr(self, "_%s" % variable)
            if values is not None:
                # print("SPATIAL_MEAN: %s=%s" % (variable, values))
                setattr(self, "_%s" % variable, np.mean(values, axis=1).reshape((-1, 1)))
                # print("=>: %s" % str(getattr(self, "_%s" % variable)))

    def time_selection(self, selection):

        # Select values in 1-dimensional variables
        for variable in ["t", "dates"]:
            values = getattr(self, "_%s" % variable)
            if values is not None:
                setattr(self, "_%s" % variable, values[selection])

        # Select values in 2-dimensional variables
        for variable in ["H", "W", "S", "K", "A", "Q"]:
            values = getattr(self, "_%s" % variable)
            if values is not None:
                setattr(self, "_%s" % variable, values[selection, :])

    def compute_slopes(self):

        self._S = np.zeros(self._H.shape)
        self._S[:, 0] = (self._H[:, 0] - self._H[:, 1]) / np.abs(self._x[0] - self._x[1])
        self._S[:, 1:-1] = (self._H[:, 0:-2] - self._H[:, 2:]) / np.abs(self._x[0:-2] - self._x[2:])
        self._S[:, -1] = (self._H[:, -2] - self._H[:, -1]) / np.abs(self._x[-2] - self._x[-1])
        self._S = np.maximum(self._S, 1e-8)

    def _piecewise_regression_fit_(self, Hs, Ws):
        
        try:

            # Compute piecewise linear regression
            if pwr is not None:

                Hi, Wi = curve_fit_lin2_pwr(Hs, Ws)
                # pw_fit = pwr.Fit(Hs, Ws, n_breakpoints=1)

                # pw_results = pw_fit.get_results()
                # if pw_results["converged"] == True:
                #     estimates = pw_results["estimates"]
                #     bp1 = estimates["breakpoint1"]["estimate"]
                #     Hi = np.array([Hs[0], bp1, Hs[-1]])
                #     Wi = pw_fit.predict(Hi)

                # else:
                #     Hi = None
            else:
                Hi, Wi, err = curve_fit_lin2(Hs, Ws)


            # Check that width are increasing
            if Hi is not None:
                Wi_diff = np.diff(Wi)
                if np.any(Wi_diff < 1e-12):
                    self._logger.warning("Piecewise linear regression failed (non monotonicaly increasing widths)")
                    Hi = None

        except:
            raise
            self._logger.warning("Unable to compute multilinear fit, switch to rectangular effective section")
            Hi = np.array([Hs[0], 0.5*(Hs[0]+Hs[-1]), Hs[-1]])
            Wi = np.ones(3) * np.mean(Ws)
            # plt.plot(Hs, Ws)
            # plt.title("Size: %i" % Hs.size)
            # plt.show()

        return Hi, Wi


    def _linear_regression_fit_(self, Hs, Ws):

        res = sps.linregress(Hs, Ws)
        Hi = np.array([Hs[0], 0.5*(Hs[0]+Hs[-1]), Hs[-1]])
        Wi = res.intercept + res.slope * Hi

        if res.slope < 1e-12 or Wi[0] < 1.0:
            self._logger.warning("Simple linear regression failed (negative slope)")
            Hi = np.array([Hs[0], 0.5*(Hs[0]+Hs[-1]), Hs[-1]])
            Wi = np.ones(3) * np.mean(Ws)

        return Hi, Wi


class L2RiverObservations:

    def __init__(self, reach_obs: L2RiverScaleObservations=None, node_obs: L2RiverScaleObservations=None):
        self._logger = logging.getLogger("H2iVDI")
        self._reach_obs = reach_obs
        self._node_obs = node_obs
        self._mid_obs = None

    @property
    def node(self):
        return self._node_obs

    @property
    def reach(self):
        return self._reach_obs

    @property
    def mid(self):
        return self._mid_obs

    def copy(self):

        new_obs = L2RiverObservations()
        new_obs._reach_obs = self._reach_obs.copy()
        new_obs._node_obs = self._node_obs.copy()
        if self._mid_obs is not None:
            new_obs._mid_obs = self._mid_obs.copy()
        return new_obs
    
    def compute_mid_scale(self, dx=1000):

        #TODO

        # Compute nm
        nm = 0
        for r in range(0, self._reach_obs.x.size):
            length = self._reach_obs._xds[r] - self._reach_obs._xus[r]
            # print("L=", length)
            nm_reach = int(np.round(length / dx))
            # print("N=", N)
            nm += nm_reach

        # Initialise observation object
        mid_obs = L2RiverScaleObservations(nt=self._reach_obs.t.size, nx=nm)
        im = 0
        for r in range(0, self._reach_obs.x.size):
            length = self._reach_obs._xds[r] - self._reach_obs._xus[r]
            print("L=", length)
            nm_reach = int(np.round(length / dx))
            dxm = length / nm_reach
            node_selection = np.ravel(np.argwhere(np.isin(self._node_obs._reach_index, r+1)))
            print("r=%i, node_selection=" % r, node_selection)
            for im_reach in range(0, nm_reach):
                xm_us = self._reach_obs._xus[r] - im_reach * dxm
                xm_ds = self._reach_obs._xus[r] - (im_reach+1) * dxm
                node_im_selection = np.ravel(np.argwhere(np.logical_and(self._node_obs.x <= xm_us, self._node_obs.x >= xm_ds)))
                print("-- im=%i, node_selection=" % im, node_im_selection)
                plt.plot(self._node_obs.x, self._node_obs.H[0, :], "b-+")
                plt.plot(self._node_obs.x[node_im_selection], self._node_obs.H[0, node_im_selection], "r.")
                plt.show()

            print("N=", N)
            nm += nm_reach


        
            choice = input()
            

        new_obs._reach_obs = self._reach_obs.copy()
        new_obs._node_obs = self._node_obs.copy()
        return new_obs    