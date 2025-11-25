import logging
import numpy as np
import os
import scipy.stats as sps
import xarray

import matplotlib.pyplot as plt

from H2iVDI.core import error_code_from_string

class SwotReachDataset:

    def __init__(self):

        self._logger = logging.getLogger("H2iVDI")

        self._nodes_data = None
        self._reach_data = None
        self._model_data = None

    def load_from_nc_file(self, fname: str, cycle_attr="observations", remove_missing_data=True, correct_with_nodes=True, 
                          enforce_node_slopes=True, max_q=2, include_details=False, sword=None):

        # Check file exists
        if not os.path.isfile(fname):
            self._logger.error("SWOT Reach file not found: %s" % fname)
            return error_code_from_string("swot_reach_file_not_found")

        # Load groups
        info = xarray.open_dataset(fname)
        reach = xarray.open_dataset(fname, group="reach", decode_times=False)
        nodes = xarray.open_dataset(fname, group="node")
        reach_H = reach.wse.values
        reach_W = reach.width.values
        reach_S = reach.slope2.values
        reach_S0 = reach_S.copy()
        reach_qual = reach.reach_q.values
        reach_qual.fill(2)
        reach_qual = reach_qual.astype(int)
        # print("WSE:", reach_H)
        # print("SLOPE2:", reach_S)
        nt = info.nt.values

        if correct_with_nodes:
            if not sword:
                raise ValueError("'sword' must be set for correcting reach data with nodes")
            #reach_nodes = sword._node_data[sword._node_data["reach"]]
            xn = sword._node_data["dist_out"]
            node_id = sword._node_data["node_id"].astype(int)
            # print("xn(SWORD) = ", xn)
            # print("node_id(SWORD) = ", node_id, len(node_id))
            # choice = input()
            correction_count = 0
            if enforce_node_slopes:
                reach_S[:] = np.nan
            for it in range(0, reach.wse.size):
                # print("sword._reach_data[dist_out]=", sword._reach_data["dist_out"])
                # choice = input()
                Hn = nodes.wse.values[:, it]
                node_id_swot = nodes.node_id.values
                indices_reordering = np.argsort(node_id_swot)[::-1]
                # print("node_id(SWOT) = ", node_id_swot[indices_reordering], len(node_id_swot))
                node_id_swot = node_id_swot[indices_reordering]
                xn = xn[indices_reordering]
                Hn = Hn[indices_reordering]
                valid = np.ravel(np.argwhere(np.isfinite(Hn)))
                # print("[NODES] valid=", valid.size)
                it_corrected = False
                if valid.size > 5:
                    # print("node_id[valid]=", node_id3[valid])
                    # print("node_id2[valid]=", node_id_swot[valid])
                    # print("x[valid]=", xn[valid])
                    # print("Hn[valid]=", Hn[valid])
                    # choice = input()
                    reg = sps.linregress(xn[valid], Hn[valid])
                    # print("[NODES, it=%i] slope=" % it, reg.slope)
                    if reg.slope >= 1e-9:
                        if np.isnan(reach_H[it]):
                            reach_H[it] = reg.slope * sword._reach_data["dist_out"][0] + reg.intercept
                            if not it_corrected:
                                correction_count += 1
                                it_corrected = True
                        if np.isnan(reach_W[it]):
                            reach_W[it] = np.nanmean(nodes.width.values[:, it])
                            if not it_corrected:
                                correction_count += 1
                                it_corrected = True
                        if np.isnan(reach_S[it]) or enforce_node_slopes:
                            reach_S[it] = reg.slope
                            # print(" => [REACH, it=%i] slope=" % it, reach_S[it])
                            if not it_corrected:
                                correction_count += 1
                                it_corrected = True
                if it_corrected:
                    reach_qual[it] = 0
                    # plt.title('s=%f' % reach_S[it])
                    # plt.show()
            
            self._logger.info("Number of corrections with nodes: %i" % correction_count)

        unvalid_details = {}
        if include_details:
            valid_details = {}
        if remove_missing_data:
            valid_data = np.logical_and(np.isfinite(reach_H), np.isfinite(reach_W))
            valid_data = np.logical_and(valid_data, np.isfinite(reach_S))
            valid_data = np.logical_and(valid_data, reach_S > 1e-12)
            unvalid_details["missing"] = np.logical_or(~np.isfinite(reach_H), ~np.isfinite(reach_W))
            unvalid_details["missing"] = np.sum(np.logical_or(unvalid_details["missing"], ~np.isfinite(reach_S)))
            if include_details:
                valid_details["has_data"] = np.logical_and(np.isfinite(reach_H), np.isfinite(reach_W))
                valid_details["has_data"] = np.logical_and(valid_details["has_data"], np.isfinite(reach_S))

        else:
            valid_data = slice(0, None, None)
        valid_data = np.logical_and(valid_data, reach.xovr_cal_q.values < 2)
        unvalid_details["xovr_cal_q<2"] = np.sum(reach.xovr_cal_q.values > 1)
        valid_data = np.logical_and(valid_data, reach_qual < max_q+1)
        unvalid_details["reach_q>%i" % (max_q)] = np.sum(reach_qual > max_q)
        valid_data = np.logical_and(valid_data, reach_S < 50.0)
        unvalid_details["slope<50.0"] = np.sum(reach_S >= 50.0)
        valid_data = np.logical_and(valid_data, reach_W >= 20.0)
        unvalid_details["width<10.0"] = np.sum(reach_W < 20.0)

        if include_details:
            valid_details["xovr_cal_q<2"] = reach.xovr_cal_q.values < 2
            valid_details["reach_q>%i" % (max_q)] = reach_qual <= max_q
            valid_details["slope<50.0"] = reach_S < 50.0
            valid_details["width>=10.0"] = reach_W >= 20.0
            # print(reach_S)

        # Retrieve reach data
        t = reach.time.values
        dates = []
        for i, seconds in enumerate(t):
            if np.isnan(seconds):
                dates.append(np.datetime64("NaT"))
            else:
                dates.append(np.datetime64("2000-01-01") + np.timedelta64(int(seconds), "s"))
        dates = np.array(dates)
        # dates = np.array([np.datetime64("2000-01-01") + np.timedelta64(int(seconds), "s") for seconds in t])
        # print("DATES:", dates)
        if cycle_attr == "observations":
            cycles = info.observations.values
        elif cycle_attr == "nt":
            cycles = info.nt.values
        else:
            raise RuntimeError("'cycle_attr' must be 'nt' or 'observations'")
        self._reach_data = {"reach_id": reach.reach_id.values,
                            "nt": nt,
                            "valid": valid_data,
                            "unvalid_details": unvalid_details,
                            "cycle": cycles,
                            "t": t,
                            "dates": dates,
                            "H": reach_H,
                            "W": reach_W,
                            "S": reach_S,
                            "S0": reach_S0,
                            "qual": reach_qual}
        if include_details:
            self._reach_data["valid_details"] = valid_details

        # Retrieve node data
        self._nodes_data = {"H": nodes.wse.values.T,
                            "W": nodes.width.values.T}

        return 0

    @property
    def reach(self):
        return self._reach_data

    @property
    def nodes(self):
        return self._nodes_data

    @property
    def model(self):
        return self._model_data

    def check(self):

        if np.any(np.isnan(self._reach_data["t"][self._reach_data["valid"]])):
            # raise RuntimeError("Reach times for reach %i has NaT values" % self._reach_data["reach_id"])
            self._logger.error("Reach times for reach %i has NaT values" % self._reach_data["reach_id"])
            return error_code_from_string("NaT_times")
        
        # Check that cycles are ascending
        if np.any(self._reach_data["cycle"][:-1] > self._reach_data["cycle"][1:]):
            # raise RuntimeError("Cycles for reach %i are not in ascending order" % self._reach_data["reach_id"])
            self._logger.error("Cycles for reach %i are not in ascending order" % self._reach_data["reach_id"])
            self._logger.error("- cycles: %s" % ",".join(list(map(str, self._reach_data["cycle"]))))
            return error_code_from_string("cycles_not_in_increasing_order")

        return 0

