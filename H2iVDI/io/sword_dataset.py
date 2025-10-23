import numpy as np
import xarray

class SwordDataset:

    def __init__(self):
        self._node_data = None
        self._reach_data = None

    def load_from_nc_file(self, fname: str, reaches_selection: list=None):

        # Load nodes and reaches groups
        nodes = xarray.open_dataset(fname, group="nodes", drop_variables=["river_name", "edit_flag"])
        reaches = xarray.open_dataset(fname, group="reaches", drop_variables=["river_name", "edit_flag"])

        # Retrieve node data
        if reaches_selection is not None:
            selection = np.where(np.isin(nodes.reach_id, reaches_selection))[0]
        else:
            selection = slice(0, None, None)
        self._node_data = {"node_id": nodes.node_id.values[selection],
                           "reach_id": nodes.reach_id.values[selection],
                           "dist_out": nodes.dist_out.values[selection]}

        # Retrieve reach data
        if reaches_selection is not None:
            selection = np.where(np.isin(reaches.reach_id, reaches_selection))[0]
        else:
            selection = slice(0, None, None)
        self._reach_data = {"reach_id": reaches.reach_id.values[selection],
                            "reach_length": reaches.reach_length.values[selection],
                            "dist_out": reaches.dist_out.values[selection]}
        
        # Sort data py decreasing dist_out
        sorted_indices = np.argsort(self._node_data["dist_out"])[::-1]
        self._node_data["node_id"] = self._node_data["node_id"][sorted_indices]
        self._node_data["reach_id"] = self._node_data["reach_id"][sorted_indices]
        self._node_data["dist_out"] = self._node_data["dist_out"][sorted_indices]
        sorted_indices = np.argsort(self._reach_data["dist_out"])[::-1]
        self._reach_data["reach_id"] = self._reach_data["reach_id"][sorted_indices]
        self._reach_data["reach_length"] = self._reach_data["reach_length"][sorted_indices]
        self._reach_data["dist_out"] = self._reach_data["dist_out"][sorted_indices]
        