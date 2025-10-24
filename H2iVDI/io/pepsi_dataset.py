import numpy as np
import xarray as xr

from H2iVDI.core import L2RiverObservations, L2RiverScaleObservations

class PepsiDataset(L2RiverObservations):

    def __init__(self, reach_obs: L2RiverScaleObservations=None, node_obs: L2RiverScaleObservations=None):
        super().__init__(reach_obs=reach_obs, node_obs=node_obs)

    def copy(self):
        new_obs = PepsiDataset(self._reach_obs.copy(), self._node_obs.copy())
        new_obs._reach_obs._good = self._reach_obs._good.copy()
        new_obs._reach_obs._A = self._reach_obs._A.copy()
        new_obs._reach_obs._Q = self._reach_obs._Q.copy()
        new_obs._node_obs._reach_index = self._node_obs._reach_index.copy()
        new_obs._node_obs._A = self._node_obs._A.copy()
        new_obs._node_obs._Q = self._node_obs._Q.copy()
        return new_obs

    def load(self, fname: str, reaches_selection="all", times_selection="all"):

        self._logger.info("Load PEPSI dataset")

        # Open netCDF file and retrieve groups
        info = xr.open_dataset(fname, group="River_Info", decode_times=False)
        reaches = xr.open_dataset(fname, group="Reach_Timeseries", decode_times=False)
        nodes = xr.open_dataset(fname, group="XS_Timeseries", decode_times=False)

        # Create and initialise reach observations
        self._reach_obs = L2RiverScaleObservations()
        self._reach_obs._QWBM = info.QWBM.values[0]
        self._reach_obs._QmeanModel = info.QWBM.values[0]
        xbnds = info.rch_bnd.values
        self._reach_obs._t = np.ravel(reaches.t.values) * 86400.0
        self._reach_obs._valid = np.ones(self._reach_obs._t.size, dtype=bool)
        # self._reach_obs._kt = np.arange(0, self._reach_obs._t.size)
        self._reach_obs._dates = np.ravel(reaches.t.values) - np.ravel(reaches.t.values)[0]
        self._reach_obs._x = 0.5 * (xbnds[:-1] + xbnds[1:])
        self._reach_obs._xus = xbnds[:-1]
        self._reach_obs._xds = xbnds[1:]
        self._reach_obs._H = reaches.H.values
        self._reach_obs._W = reaches.W.values
        self._reach_obs._S = reaches.S.values
        self._reach_obs._A = reaches.A.values
        self._reach_obs._Q = reaches.Q.values
        self._reach_obs._K = None

        # Retrieve good flag
        self._reach_obs._good = np.zeros(self._reach_obs.nx, dtype=int)
        self._reach_obs._good[info.gdrch.values.astype(int)-1] = 1

        # Create and initialise nodes observations
        self._node_obs = L2RiverScaleObservations()
        self._node_obs._QWBM = info.QWBM.values[0]
        self._node_obs._QmeanModel = info.QWBM.values[0]
        self._node_obs._t = np.ravel(nodes.t.values) * 86400.0
        self._node_obs._valid = np.ones(self._node_obs._t.size, dtype=bool)
        # self._node_obs._kt = np.arange(0, self._node_obs._t.size)
        self._node_obs._dates = np.ravel(nodes.t.values) - np.ravel(nodes.t.values)[0]
        self._node_obs._x = np.ravel(nodes.X.values)
        self._node_obs._reach_index = np.ravel(nodes.xs_rch.values.astype(int))
        self._node_obs._H = nodes.H.values
        self._node_obs._W = nodes.W.values
        self._node_obs._A = nodes.A.values
        self._node_obs._K = 1.0 / nodes.n.values
        self._node_obs._Q = nodes.Q.values

        # Restrict to selected reaches
        if reaches_selection == "good_reaches":
            self._select_good_reaches_()
        elif reaches_selection == "consecutive_good_reaches":
            self._select_consecutive_good_reaches_()
        elif isinstance(reaches_selection, tuple):
            if len(reaches_selection) != 2:
                raise ValueError("'reaches_selection' tuple must be of len=2'")
            selection = np.arange(reaches_selection[0]-1, reaches_selection[1]-1)
            self._select_reaches_(selection)
        elif isinstance(reaches_selection, list):
            reaches_selection = np.array(reaches_selection)
            if reaches_selection.dtype != int or np.any(reaches_selection < 0):
                raise ValueError("'reaches_selection' must be an array of positive integers")
            self._select_reaches_(reaches_selection)
            # if len(reaches_selection) != 2:
            #     raise ValueError("'reaches_selection' list must be of len=2'")
            # selection = np.arange(reaches_selection[0]-1, reaches_selection[1]-1)
            # self._select_reaches_(selection)
        elif isinstance(reaches_selection, np.ndarray):
            if reaches_selection.dtype != int or np.any(reaches_selection < 0):
                raise ValueError("'reaches_selection' must be an array of positive integers")
            self._select_reaches_(reaches_selection)
        elif reaches_selection != "all":
            raise ValueError("Wrong value for 'reaches_selection': %s" % reaches_selection)

        # Restrict to selected times
        if times_selection == "positive_discharges":
            self._select_times_with_positive_discharges_()
        elif times_selection == "consecutive_positive_discharges":
            self._select_consecutive_times_with_positive_discharges_()
        elif isinstance(times_selection, tuple):
            if len(reaches_selection) != 2:
                raise times_selection("'times_selection' tuple must be of len=2'")
            selection = np.arange(times_selection[0]-1, times_selection[1]-1)
            self._select_times_(selection)
        elif isinstance(times_selection, list):
            if len(times_selection) != 2:
                raise ValueError("'times_selection' list must be of len=2'")
            selection = np.arange(times_selection[0]-1, times_selection[1]-1)
            self._select_times_(selection)
        elif isinstance(times_selection, np.ndarray):
            if times_selection.dtype != int or np.any(times_selection < 0):
                raise ValueError("'times_selection' must be an array of positive integers")
            self._select_times_(times_selection)
        elif times_selection == "percentiles":
            isort = np.argsort(self._reach_obs._H[:, 0])
            selection = isort[np.arange(10, 100, 10)*isort.size//100] + 1
            # print("selection=", selection)
            # choice = input()
            self._select_times_(selection)
        elif times_selection != "all":
            raise ValueError("Wrong value for 'times_selection': %s" % reaches_selection)

        self._logger.info("- Dataset dimensions: nt=%i, nr=%i, nn=%i" % (self._reach_obs.nt, self._reach_obs.nx, self._node_obs.nx))

        return 0

    def _select_reaches_(self, selection: np.ndarray):

        # Selection of data at reach scale
        self._reach_obs.spatial_selection(selection-1)
        # self._reach_obs._A = self._reach_obs._A[:, selection-1]
        # self._reach_obs._Q = self._reach_obs._Q[:, selection-1]

        # Retrieve node selection
        node_selection = np.ravel(np.argwhere(np.isin(self._node_obs._reach_index, selection)))

        # Selection of data at node scale
        self._node_obs.spatial_selection(node_selection)
        # self._node_obs._A = self._node_obs._A[:, node_selection-1]
        # self._node_obs._Q = self._node_obs._Q[:, node_selection-1]

    def _select_consecutive_good_reaches_(self):

        # Find largest sets of consecutive good reaches
        first_r = 0
        count_max = 0
        while first_r < self._reach_obs.nx-1:
            if self._reach_obs._good[first_r] == 1:
                cur_r = first_r
                while self._reach_obs._good[cur_r+1] == 1:
                    cur_r += 1
                    if cur_r == self._reach_obs.nx-1:
                        break
                if cur_r - first_r + 1 > count_max:
                    selection = np.arange(first_r, cur_r+1)
                    count_max = cur_r - first_r + 1
                first_r = min(self._reach_obs.nx-1, cur_r+1)
            else:
                first_r += 1

        if count_max > 0:
            self._logger.info("- Selected consecutive good reaches: %s" % ", ".join(["%i" % r for r in selection]))

        self._select_reaches_(selection+1)

    def _select_times_(self, selection: np.ndarray):

        # Selection of data at reach scale
        self._reach_obs.time_selection(selection-1)
        self._reach_obs._valid = self._reach_obs._valid[selection-1]
        self._reach_obs._A = self._reach_obs._A[selection-1, :]
        self._reach_obs._Q = self._reach_obs._Q[selection-1, :]

        # Selection of data at node scale
        self._node_obs.time_selection(selection-1)
        self._node_obs._valid = self._node_obs._valid[selection-1]
        self._node_obs._A = self._node_obs._A[selection-1, :]
        self._node_obs._Q = self._node_obs._Q[selection-1, :]

    def _select_times_with_positive_discharges_(self):

        selection = np.ravel(np.argwhere(np.mean(self._reach_obs._Q, axis=1) >= 1e-12))
        if selection.size < 10:
            selection_repr = ", ".join(["%i" % r for r in selection])
        else:
            selection_repr = ", ".join(["%i" % r for r in selection[0:3]]) + ", ..., " + ", ".join(["%i" % r for r in selection[-3:]])
        self._logger.info("- Select times with postive Q: %s" % selection_repr)
        self._select_times_(selection)


    def _select_consecutive_times_with_positive_discharges_(self):

        Qmin = np.min(self._reach_obs._Q, axis=1)

        # Find largest sets of consecutive postitive discharges
        first_it = 0
        count_max = 0
        while first_it < Qmin.size-1:
            if Qmin[first_it] >= 1e-12:
                cur_it = first_it
                while Qmin[cur_it+1] > 1e-12:
                    cur_it += 1
                    if cur_it == Qmin.size-1:
                        break
                if cur_it - first_it + 1 > count_max:
                    selection = np.arange(first_it, cur_it+1)
                    count_max = cur_it - first_it + 1
                first_it = min(Qmin.size-1, cur_it+1)
            else:
                first_it += 1

        if count_max > 0:
            self._logger.info("- Selected consecutive times with positives discharges: %i-%i" % (selection[0], selection[-1]))

        self._select_times_(selection+1)
