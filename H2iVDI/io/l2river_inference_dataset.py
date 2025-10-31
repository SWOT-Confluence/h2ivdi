import logging
import numpy as np
import os
import xarray as xr

from H2iVDI.core import L2RiverObservations, L2RiverScaleObservations, error_code_from_string
from .sos_dataset import SoSDataset
from .sword_dataset import SwordDataset
from .swot_reach_dataset import SwotReachDataset


class L2RiverInferenceDataset(L2RiverObservations):
    
    def __init__(self, set_def, input_dir: str, output_dir: str):
        super().__init__()

        ## Load dataset
        #self.load(set_def, input_dir, output_dir)

    def load(self, set_def, input_dir: str, output_dir: str, s3_path: str, cycle_attr="nt"):

        if isinstance(set_def, dict):

            # Load single reach set
            error_code = self.load_single_reach_set(set_def, input_dir, output_dir, s3_path)

        elif isinstance(set_def, list):

            if len(set_def) == 1:
                # Load single reach set
                error_code = self.load_single_reach_set(set_def[0], input_dir, output_dir, s3_path, cycle_attr=cycle_attr)

            else:

                # Load multiple reaches set
                error_code = self.load_multi_reaches_set(set_def, input_dir, output_dir, s3_path, cycle_attr=cycle_attr)

        if error_code != 0: return error_code 

        self._logger.debug("- Observation set:")
        self._logger.debugL2("  P   R              Date       H       W       S [Q]")
        for i in range(0, self.reach.nt):
            for j in range(0, self.reach.nx):
                self._logger.debugL2(" %03i %03i %s %7.2f %7.2f %7.4f [%1i]" % (i+1,
                                                                                j+1,
                                                                                str(self.reach.dates[i])[:16],
                                                                                self.reach.H[i, j],
                                                                                self.reach.W[i, j],
                                                                                self.reach.S[i, j] * 1000.0,
                                                                                self.reach._qual[i, j]))

        
        # Check for section without observations
        if self.reach.nt == 0:
            self._logger.error("Dataset with no valid observation")
            return error_code_from_string("dataset_without_valid_observation")
        if np.any(np.all(np.isnan(self.reach.H), axis=0)):
            self._logger.error("Section with no valid height observation detected")
            return error_code_from_string("reach_without_valid_observation")
        if np.any(np.all(np.isnan(self.reach.W), axis=0)):
            self._logger.error("Section with no valid width observation detected")
            return error_code_from_string("reach_without_valid_observation")
        if np.any(np.all(np.isnan(self.reach.S), axis=0)):
            self._logger.error("Section with no valid slope observation detected")
            return error_code_from_string("reach_without_valid_observation")

        return 0


    def load_single_reach_set(self, set_def, input_dir: str, output_dir: str, s3_path: str, cycle_attr: str="nt"):
        # raise NotImplementedError("Not implemented yet !")

        # Retrieve reaches list
        reach_id_list = [set_def["reach_id"]]
        self._logger.info("Load set with single reach")
        self._logger.info("- ID of reach: %i" % set_def["reach_id"])

        # Load reach in sword
        self._logger.debug("Load SWORD data")
        sword_file = os.path.join(input_dir, "sword", set_def["sword"])
        sword = SwordDataset()
        sword.load_from_nc_file(sword_file, reach_id_list)

        # Load reach in SoS
        self._logger.debug("Load SoS data")

        # Set the filepath to the s3 file if not running locally
        if s3_path == 'local':
            self._logger.debug("Loading the SoS locally")
            sos_file = os.path.join(input_dir, "sos", set_def["sos"])
        else:
            self._logger.debug("Loading the SoS from S3")
            sos_file = os.path.join(s3_path, set_def["sos"])

        sos = SoSDataset()
        sos.load_from_nc_file(sos_file, reach_id_list, load_observations=False)

        # Load SWOT observations file
        self._logger.debug("Load SWOT data")
        swot_file = os.path.join(input_dir, "swot", set_def["swot"])
        swot = SwotReachDataset()
        error_code = swot.load_from_nc_file(swot_file, cycle_attr=cycle_attr, correct_with_nodes=True, remove_missing_data=True, sword=sword)
        if error_code != 0: return error_code
        error_code = swot.check()
        if error_code != 0: return error_code

        valid_count = np.sum(swot.reach["valid"])
        self._logger.debug("- SWOT obs: reach_id=%i" % int(os.path.basename(set_def["swot"]).split("_")[0]))
        self._logger.debug("  - reach nt=%i" % (swot.reach["H"].size))
        self._logger.debug("  - nodes nt=%i, nx=%i" % swot.nodes["H"].shape)
        self._logger.debug("  - valid: %i" % valid_count)
        self._logger.debug("  - unvalid details:")
        for key in swot.reach["unvalid_details"]:
            self._logger.debug("    - %s: %i" % (key, swot.reach["unvalid_details"][key]))

        # Compute data from SWOT observations data
        self._node_obs = L2RiverScaleObservations(valid_count, swot.nodes["H"].shape[1])
        self._reach_obs = L2RiverScaleObservations(valid_count, 2)
        self._reach_obs._qual = np.zeros_like(self._reach_obs._H, dtype=int)
        self._reach_obs._valid = swot.reach["valid"]
        self._cycles = swot.reach["cycle"][self._reach_obs._valid]
        self._node_obs._t[:] = swot.reach["t"][self._reach_obs._valid]
        self._node_obs._H[:, :] = swot.nodes["H"][self._reach_obs._valid, :]
        self._node_obs._W[:, :] = swot.nodes["W"][self._reach_obs._valid, :]
        reach_length = sword._reach_data["reach_length"]
        self._reach_obs._QmeanModel = np.mean(sos.model_data["mean_q"])
        self._reach_obs._nt = swot.reach["nt"]
        # print("nt=", self._reach_obs._nt)
        self._reach_obs._t[:] = swot.reach["t"][self._reach_obs._valid]
        self._reach_obs._dates = swot.reach["dates"]
        # print("dates=", swot.reach["dates"])
        self._reach_obs._x[0] = sword._reach_data["dist_out"] + reach_length
        self._reach_obs._H[:, 0] = swot.reach["H"][self._reach_obs._valid] + 0.5 * reach_length * swot.reach["S"][self._reach_obs._valid]
        self._reach_obs._W[:, 0] = swot.reach["W"][self._reach_obs._valid]
        self._reach_obs._S[:, 0] = swot.reach["S"][self._reach_obs._valid]
        self._reach_obs._qual[:, 0] = swot.reach["qual"][self._reach_obs._valid]
        self._reach_obs._x[1] = sword._reach_data["dist_out"]
        self._reach_obs._H[:, 1] = swot.reach["H"][self._reach_obs._valid] - 0.5 * reach_length * swot.reach["S"][self._reach_obs._valid]
        self._reach_obs._W[:, 1] = swot.reach["W"][self._reach_obs._valid]
        self._reach_obs._S[:, 1] = swot.reach["S"][self._reach_obs._valid]
        self._reach_obs._qual[:, 1] = swot.reach["qual"][self._reach_obs._valid]
        self._logger.debugL2("  - data:")
        self._logger.debugL2("Cycle             Date       H       W       S [V] [Q]")
        for i in range(0, swot.reach["t"].size):
            self._logger.debugL2(" %04i %s %7.2f %7.2f %7.4f [%s] [%1i]" % (swot.reach["cycle"][i], 
                                                                       str(swot.reach["dates"][i])[:16],
                                                                       swot.reach["H"][i],
                                                                       swot.reach["W"][i],
                                                                       swot.reach["S"][i] * 1000.0,
                                                                       "x" if swot.reach["valid"][i] else " ",
                                                                       swot.reach["qual"][i]))

        self._logger.info("- Set dimensions: nt=%i, nr=%i, nn=%i" % (self._reach_obs.nt, self._reach_obs.nx, self._node_obs.nx))
        self._logger.info("  - reach scale: nt=%i, nx=%i" % (self._reach_obs.nt, self._reach_obs.nx))
        self._logger.info("  - node scale : nt=%i, nx=%i" % (self._node_obs.nt, self._node_obs.nx))

        self._initial_reach_count = 1

        return 0


    def load_multi_reaches_set(self, set_def, input_dir: str, output_dir: str, s3_path: str, cycle_attr: str="observations"):

        # Retrieve reaches list
        reach_id_list = [reach_def["reach_id"] for reach_def in set_def]
        self._logger.info("Load set with multiple reaches")
        self._logger.info("- IDs of reach: %s" % ",".join(["%i" % reach_id for reach_id in reach_id_list]))

        # Load reaches in sword
        self._logger.debug("Load SWORD data")
        sword_file = os.path.join(input_dir, "sword", set_def[0]["sword"])
        sword = SwordDataset()
        sword.load_from_nc_file(sword_file, reach_id_list)

        # Load reaches in SoS
        self._logger.debug("Load SoS data")

        # Set the filepath to the s3 file if not running locally
        if s3_path == 'local':
            self._logger.debug("Loading the SoS locally")
            sos_file = os.path.join(input_dir, "sos", set_def[-1]["sos"])
        else:
            self._logger.debug("Loading the SoS from S3")
            sos_file = os.path.join(s3_path, set_def[-1]["sos"])

        sos = SoSDataset()
        sos.load_from_nc_file(sos_file, reach_id_list, load_observations=False)

        # Load last (downstream) SWOT observations file
        self._logger.debug("Load SWOT data")
        swot_file = os.path.join(input_dir, "swot", set_def[-1]["swot"])
        swot = SwotReachDataset()
        error_code = swot.load_from_nc_file(swot_file, cycle_attr=cycle_attr, correct_with_nodes=True, remove_missing_data=True, sword=sword)
        if error_code != 0: return error_code
        error_code = swot.check()
        if error_code != 0: return error_code

        valid_count = np.sum(swot.reach["valid"])
        self._logger.debug("- Downstream SWOT obs [%i/%i]: reach_id=%i" % (len(set_def), len(set_def), int(set_def[-1]["swot"].split("_")[0])))
        self._logger.debug("  - reach nt=%i" % (swot.reach["H"].size))
        self._logger.debug("  - nodes nt=%i, nx=%i" % swot.nodes["H"].shape)
        self._logger.debug("  - valid: %i" % valid_count)
        self._logger.debug("  - unvalid details:")
        for key in swot.reach["unvalid_details"]:
            self._logger.debug("    - %s: %i" % (key, swot.reach["unvalid_details"][key]))

        # Initialise data with last (downstream) SWOT observations data
        self._node_obs = L2RiverScaleObservations(valid_count, swot.nodes["H"].shape[1])
        self._reach_obs = L2RiverScaleObservations(valid_count, 1)
        self._reach_obs._qual = np.zeros_like(self._reach_obs._H, dtype=int)
        self._reach_obs._valid = swot.reach["valid"]
        self._cycles = swot.reach["cycle"][self._reach_obs._valid]
        self._node_obs._t[:] = swot.reach["t"][self._reach_obs._valid]
        self._node_obs._H[:, :] = swot.nodes["H"][self._reach_obs._valid, :]
        self._node_obs._W[:, :] = swot.nodes["W"][self._reach_obs._valid, :]
        self._reach_obs._x = sword._reach_data["dist_out"]
        self._reach_obs._QmeanModel = np.mean(sos.model_data["mean_q"])
        self._reach_obs._nt = swot.reach["nt"]
        self._reach_obs._valid = swot.reach["valid"]
        self._reach_obs._t[:] = swot.reach["t"][self._reach_obs._valid]
        self._reach_obs._dates = swot.reach["dates"]
        self._reach_obs._H[:, 0] = swot.reach["H"][self._reach_obs._valid]
        self._reach_obs._W[:, 0] = swot.reach["W"][self._reach_obs._valid]
        self._reach_obs._S[:, 0] = swot.reach["S"][self._reach_obs._valid]
        self._reach_obs._qual[:, 0] = swot.reach["qual"][self._reach_obs._valid]
        self._logger.debugL2("  - data:")
        self._logger.debugL2("Cycle             Date       H       W       S [Q]")
        for i in range(0, swot.reach["t"].size):
            self._logger.debugL2(" %04i %s %7.2f %7.2f %7.4f [%1i]" % (swot.reach["cycle"][i],
                                                                       str(swot.reach["dates"][i])[:16],
                                                                       swot.reach["H"][i],
                                                                       swot.reach["W"][i],
                                                                       swot.reach["S"][i] * 1000.0,
                                                                       swot.reach["qual"][i]))

        # Load other SWOT observations data and update data
        for i in range(len(set_def)-2, -1, -1):
            swot_file = os.path.join(input_dir, "swot", set_def[i]["swot"])
            swot = SwotReachDataset()
            error_code = swot.load_from_nc_file(swot_file, cycle_attr=cycle_attr, correct_with_nodes=True, remove_missing_data=True, sword=sword)
            if error_code != 0: return error_code
            error_code = swot.check()
            if error_code != 0: return error_code

            # Find cycles that are in reference SWOT observation and their indices
            valid_cycles_flag = np.isin(swot.reach["cycle"][swot.reach["valid"]], self._cycles)
            valid_cycles = swot.reach["cycle"][swot.reach["valid"]][valid_cycles_flag]
            indices = np.searchsorted(self._cycles, valid_cycles)

            # swot.check()
            valid_count = np.sum(swot.reach["valid"])
            self._logger.debug("- Next SWOT obs [%i/%i]: reach_id=%i" % (i+1, len(set_def), int(set_def[i]["swot"].split("_")[0])))
            self._logger.debug("  - reach nt=%i" % (swot.reach["H"].size))
            self._logger.debug("  - nodes nt=%i, nx=%i" % swot.nodes["H"].shape)
            self._logger.debug("  - valid: %i" % valid_count)
            self._logger.debug("  - unvalid details:")
            for key in swot.reach["unvalid_details"]:
                self._logger.debug("    - %s: %i" % (key, swot.reach["unvalid_details"][key]))

            self._logger.debugL2("  - data:")
            self._logger.debugL2("Cycle             Date       H       W       S [Q]")
            for i in range(0, swot.reach["t"].size):
                self._logger.debugL2(" %04i %s %7.2f %7.2f %7.4f [%1i]" % (swot.reach["cycle"][i],
                                                                           str(swot.reach["dates"][i])[:16],
                                                                           swot.reach["H"][i],
                                                                           swot.reach["W"][i],
                                                                           swot.reach["S"][i] * 1000.0,
                                                                           swot.reach["qual"][i]))

            corrected_H = np.ones((self._node_obs.H.shape[0], swot.nodes["H"].shape[1])) * np.nan
            # print("indices=", indices, ", corrected_H.shape=", corrected_H.shape)
            # print("valid_cycles=", valid_cycles)
            # print("valid_cycles_flag=", valid_cycles_flag)
            # print(swot.nodes["H"].shape)
            # print(swot.nodes["H"][swot.reach["valid"], :][valid_cycles_flag, :])
            # choice = input()
            corrected_H[indices, :] = swot.nodes["H"][swot.reach["valid"], :][valid_cycles_flag, :]
            self._node_obs._H = np.concatenate((corrected_H, self._node_obs.H), axis=1)
            corrected_W = np.ones((self._node_obs.H.shape[0], swot.nodes["H"].shape[1])) * np.nan
            corrected_W[indices, :] = swot.nodes["W"][swot.reach["valid"], :][valid_cycles_flag, :]
            self._node_obs._W = np.concatenate((corrected_W, self._node_obs.W), axis=1)

            corrected_H = np.ones((self._reach_obs.H.shape[0], 1)) * np.nan
            corrected_H[indices, :] = swot.reach["H"][swot.reach["valid"]][valid_cycles_flag].reshape((-1, 1))
            self._reach_obs._H = np.concatenate((corrected_H, self._reach_obs.H), axis=1)
            corrected_W = np.ones((self._reach_obs.W.shape[0], 1)) * np.nan
            corrected_W[indices, :] = swot.reach["W"][swot.reach["valid"]][valid_cycles_flag].reshape((-1, 1))
            self._reach_obs._W = np.concatenate((corrected_W, self._reach_obs.W), axis=1)
            corrected_S = np.ones((self._reach_obs.S.shape[0], 1)) * np.nan
            corrected_S[indices, :] = swot.reach["S"][swot.reach["valid"]][valid_cycles_flag].reshape((-1, 1))
            self._reach_obs._S = np.concatenate((corrected_S, self._reach_obs.S), axis=1)
            corrected_qual = np.zeros((self._reach_obs._qual.shape[0], 1), dtype=int)
            corrected_qual[indices] = swot.reach["qual"][swot.reach["valid"]][valid_cycles_flag].reshape((-1, 1))
            self._reach_obs._qual = np.concatenate((corrected_qual, self._reach_obs._qual), axis=1)
        # if np.any(self.reach.W)

        # Store in-situ observations
        if sos._obs_data is not None:
            self._reach_obs._insitu_data = sos._obs_data
        else:
            self._obs_data = None

        self._logger.info("- Set dimensions: nt=%i, nr=%i, nn=%i" % (self._reach_obs.nt, self._reach_obs.nx, self._node_obs.nx))
        self._logger.info("  - reach scale: nt=%i, nx=%i" % (self._reach_obs.nt, self._reach_obs.nx))
        self._logger.info("  - node scale : nt=%i, nx=%i" % (self._node_obs.nt, self._node_obs.nx))

        # # Interpolate and store in-situ observations
        # if sos._obs_data is not None:
        #     self._obs_data = {}
        #     for reach_id in sos._obs_data:
        #         dt_swot = (self._reach_obs._t - np.datetime64("2000-01-01")) / np.timedelta64(1, "S")
        #         dt_obs = (sos._obs_data["t"] - np.datetime64("2000-01-01")) / np.timedelta64(1, "S")
        #         q = np.interp(dt_swot, dt_obs, sos._obs_data["Q"])
        #         self._obs_data[reach_id] = {"dates": sos._obs_data["t"],
        #                                     "Q": sos._obs_data["Q"],
        #                                     "Qi": q}
        # else:
        #     self._obs_data = None
        self._initial_reach_count = self._reach_obs.nx

        return 0