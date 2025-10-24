import logging
import numpy as np
import os
import xarray
try:
    from sos_read.sos_read import download_sos
except:
    download_sos = None


class SoSDataset:

    def __init__(self):
        self._logger = logging.getLogger("H2iVDI")
        self._node_data = None
        self._reach_data = None
        self._model_data = None

    def load_from_nc_file(self, fname: str, reaches_selection: list=None, load_observations=True):

        # Retrieve list of observations groups
        if load_observations is True:
            continent_id = os.path.basename(fname).split("_")[0].lower()
            if continent_id == "af":
                obs_groups = ["DWA"]
            elif continent_id == "as":
                raise NotImplementedError("Not implemented yet !")
            elif continent_id == "eu":
                obs_groups = ["EAU", "DEFRA"]
            elif continent_id == "na":
                obs_groups = ["USGS", "WSC", "MEFCCWP"]
            elif continent_id == "sa":
                obs_groups = ["DGA", "Hidroweb"]
            else:
                raise ValueError("Unknown continent ID: %s" % continent_id)
        
        # Decide to load them from S3 or locally
        if 'confluence' in fname:
            if download_sos is None:
                raise RuntimeError("Module sos_read not found, cannot download from S3 bucket")
            download_sos(bucket_key=os.path.dirname(fname), sos_filepath = os.path.join('/tmp', os.path.basename(fname)))
            fname = os.path.join('/tmp', os.path.basename(fname))

        # Load groups
        nodes = xarray.open_dataset(fname, group="nodes", drop_variables=["river_name", "edit_flag"])
        reaches = xarray.open_dataset(fname, group="reaches", drop_variables=["river_name", "edit_flag"])
        model = xarray.open_dataset(fname, group="model", drop_variables=["river_name", "edit_flag"])
        if load_observations is True:
            obs = {}
            for obs_group in obs_groups:
                obs[obs_group] = xarray.open_dataset(fname, group=obs_group, decode_times=False)

        # Retrieve node data
        self._logger.debugL2("- Load nodes data")
        if reaches_selection is not None:
            selection = np.where(np.isin(nodes.reach_id, reaches_selection))[0]
        else:
            selection = slice(0, None, None)
        self._node_data = {"node_id": nodes.node_id.values[selection],
                           "reach_id": nodes.reach_id.values[selection]}

        # Retrieve reach data
        self._logger.debugL2("- Load reach data")
        if reaches_selection is not None:
            selection = np.where(np.isin(reaches.reach_id, reaches_selection))[0]
        else:
            selection = slice(0, None, None)
        self._reach_data = {"reach_id": reaches.reach_id.values[selection]}

        # Retrieve model data
        self._logger.debugL2("- Load model data")
        if reaches_selection is not None:
            selection = np.where(np.isin(reaches.reach_id, reaches_selection))[0]
        else:
            selection = slice(0, None, None)
        self._model_data = {"mean_q": model.mean_q.values[selection],
                            "monthly_q": model.monthly_q.values[selection, :]}

        # Retrieve observations data
        if load_observations is True:
            self._logger.debugL2("- Load observations data")
            self._obs_data = {reach_id: None for reach_id in self._reach_data["reach_id"]}
            for key in obs:
                selection = np.where(np.isin(obs[key]["%s_reach_id" % key], self._reach_data["reach_id"]))[0]
                if selection.size > 0:
                    obs_reach_id = obs[key]["%s_reach_id" % key][selection].values
                    print("obs_reach_id=", obs_reach_id)
                    for i in range(selection.size):
                        t = obs[key]["%s_qt" % key][selection].values
                        q = obs[key]["%s_q" % key][selection].values
                        q = q[np.isfinite(t)]
                        t = t[np.isfinite(t)]
                        dates = np.array([np.datetime64("0001-01-01") + np.timedelta64(int(days), "D") for days in t])
                        print("dates=", t)
                        print("q=", q)
                        choice = input()
                        self._obs_data[obs_reach_id[i]] = {"t": dates, "Q": q}
        else:
            self._obs_data = None

        nodes.close()
        reaches.close()
        model.close()
        if load_observations is True:
            for key in obs:
                obs[key].close()

    @property
    def reach_data(self):
        return self._reach_data

    @property
    def node_data(self):
        return self._node_data

    @property
    def model_data(self):
        return self._model_data
