import datetime
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import os

from .case_processor import CaseProcessor
from H2iVDI.chains import BayesianChain
from H2iVDI.core import error_string_from_code
from H2iVDI.io import L2RiverInferenceDataset


class SwotCaseProcessor(CaseProcessor):

    def __init__(self, set_def, run_mode: str, input_dir:str, output_dir:str, s3_path:str):
        super().__init__()

        # Store set definition
        self._set_def = set_def

        # Store run mode definition
        self._run_mode = run_mode

        # Store directories
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._s3_path = s3_path

        # Store reaches ID
        if isinstance(set_def, dict):
            self._reaches_id = [set_def["reach_id"]]
        else:                
            self._reaches_id = [reach_def["reach_id"] for reach_def in set_def]

        # Create rundir
        if "AWS_BATCH_JOB_ARRAY_INDEX" in os.environ:
            tmp_rundir = ".rundir_set%i" % int(os.environ["AWS_BATCH_JOB_ARRAY_INDEX"])
        elif "EOHYDROLAB_SET_INDEX" in os.environ:
            tmp_rundir = ".rundir_set%i" % int(os.environ["EOHYDROLAB_SET_INDEX"])
        else:
            tmp_rundir = ".rundir"
        if not os.path.isdir(os.path.join(self._output_dir, tmp_rundir)):
            os.mkdir(os.path.join(self._output_dir, tmp_rundir))
        self._rundir = os.path.join(self._output_dir, tmp_rundir, "%i-%i" % (self._reaches_id[0], self._reaches_id[-1]))
        if not os.path.isdir(self._rundir):
            os.mkdir(self._rundir)

    def resume(self, output_dir):

        self._calibration_results = self._read_output_(output_dir)
        return (self._calibration_results is not None)

    def prepro(self):

        # Load dataset
            # def __init__(self, set_def, input_dir: str, output_dir: str):
        self._data = L2RiverInferenceDataset(set_def=self._set_def, input_dir=self._input_dir, output_dir=self._output_dir)
        # print(self._data, 'here is data')
        error_code = self._data.load(self._set_def, self._input_dir, self._output_dir, self._s3_path)
        if error_code != 0: return error_code

        # Compute effective section
        if self._data.reach.nt > 0:
            self._logger.info("- Compute effective sections:")
            self._data.reach.compute_effective_sections()
            if self._logger._debug_level > 0:
                cmap, norm = mcolors.from_levels_and_colors([0.5, 1.5, 2.5, 3.5], ['green', 'orange', 'red'])
                for r in range(0, self._data.reach.H.shape[1]):
                    plt.scatter(self._data.reach.H[:, r], self._data.reach.W[:, r], c=self._data.reach._qual[:, r], cmap=cmap, norm=norm, label="raw data")
                    plt.plot(self._data.reach.He[:, r], self._data.reach.We[:, r], "b--", label="effective section")
                    plt.plot(self._data.reach.He[0, r], self._data.reach.We[0, r], "bd")
                    plt.plot(self._data.reach.He[1, r], self._data.reach.We[1, r], "rd")
                    plt.plot(self._data.reach.He[2, r], self._data.reach.We[2, r], "gd")
                    if len(self._reaches_id) == 1:
                        if r == 0:
                            plt.title("%s - upstream" % self._reaches_id[0])
                        else:
                            plt.title("%s - downstream" % self._reaches_id[0])
                    else:
                        plt.title(self._reaches_id[r])
                    plt.legend()
                    plt.savefig(os.path.join(self._rundir, "effective_section_reach%03i.png" % (r+1)))
                    plt.close(plt.gcf())

        return 0

    def run(self):

        self._logger.debug("[START] Run SWOT case processor")

        # # Create tmpdir
        # rundir = os.path.join(self._output_dir, tmp_rundir)
        # if not os.path.isdir(rundir):
        #     os.mkdir(rundir)
        # tmpdir = os.path.join(rundir, "%i-%i" % (self._reaches_id[0], self._reaches_id[-1]))
        # if not os.path.isdir(tmpdir):
        #     os.mkdir(tmpdir)
        self._logger.debug("rundir is %s" % self._rundir)

        # Create chain parameters
        parameters = {"model": "swst3lfb",
                      "run_mode": self._run_mode,
                      "q0_method": "optim",
                      "calibrate_sigma_obs": False,
                      "plots": {"inference": True,
                                "validation": False}}
        if np.isnan(self._data._reach_obs._QmeanModel):
            self._logger.debug("- Set Qin method to low-froude (QMeanModel is NaN)")
            parameters["q0_method"] = "low-froude"

        # Create chain
        chain = BayesianChain(self._data.reach, parameters)

        # Calibration
        self._calibration_results, error_code = chain.calibrate(rundir=self._rundir)
        if error_code != 0: return error_code

        # plt.plot(self._data._reach_obs.t, self._data._reach_obs.H)
        # plt.show()

        self._logger.debug("[ END ] Run SWOT case processor")

        return 0

    def postpro(self, output_dir="/mnt/data/output"):

        # Store results in rundir
        # TODO
        # self._store_calibration_results_(results)

        # Create output file
        if self._data.reach.nt > 0:
            self._write_output_(self._calibration_results, output_dir)
        
        return self._calibration_results, 0

    def write_failed_output(self, output_dir="/mnt/data/output", error_code=999):

        # Store results in rundir
        # TODO
        # self._store_calibration_results_(results)

        # Create output file
        self._write_failed_output_(output_dir, error_code)

    # def load_data(self, set_def: str):
    #     self._data = PepsiDataset(set_def, self._input_dir, self._output_dir)

    #     plt.plot(self._data.reaches.x, self._data.reaches.H)
    #     plt.show()

    def _store_calibration_results_(self):
        pass

    def _read_output_(self, output_dir, single_output_per_reach=True):
        # TODO use xarray instead of netcdf ?

        # results = 

        if single_output_per_reach is True:
            for i, reach_id in enumerate(self._reaches_id):
                self._logger.debug("- Read results for reach %i" % reach_id)
                # out_file = os.path.join(output_dir, "%s_h2ivdi.nc" % reach_id)
                out_file = os.path.join(output_dir, "%s_h2ivdi.nc" % reach_id)
                if not os.path.isfile(out_file):
                    out_file = os.path.join(output_dir, "%s_hivdi.nc" % reach_id)
                    if not os.path.isfile(out_file):
                        return None
                single_reach_results = self._read_single_reach_output_(out_file)

                if i == 0:

                    results = single_reach_results
                    results["posterior"]["A0"] = np.array([results["posterior"]["A0"]])
                    results["posterior"]["n"] = np.array([results["posterior"]["n"]])
                    results["posterior"]["Q"] = results["posterior"]["Q"].reshape((-1, 1))

                else:

                    # # Check model is the same
                    # if single_reach_results["model"] != results["model"]:
                    #     raise RuntimeError("Model is different accross reaches")

                    results["posterior"]["A0"] = np.concatenate((results["posterior"]["A0"], np.array([single_reach_results["posterior"]["A0"]])))
                    results["posterior"]["n"] = np.concatenate((results["posterior"]["n"], np.array([single_reach_results["posterior"]["n"]])))
                    results["posterior"]["Q"] = np.concatenate((results["posterior"]["Q"], single_reach_results["posterior"]["Q"].reshape((-1, 1))), axis=1)

                # single_reach_results = {"model": results["model"],
                #                         "status": results["status"],
                #                         "error_code": results["error_code"],
                #                         "error_string": results["error_string"],
                #                         "prior": {"A0": results["prior"]["A0"][i],
                #                                   "n": results["prior"]["n"][i],
                #                                   "Q": results["prior"]["Q"][:]},
                #                         "posterior": {"A0": results["posterior"]["A0"][i],
                #                                       "n": results["posterior"]["n"][i],
                #                                       "Q": results["posterior"]["Q"][:],
                #                                       "Q_ci": results["posterior"]["Q_ci"][:, :]}}

        else:
            raise NotImplementedError("Not implemented yet !")

        return results

    def _read_single_reach_output_(self, out_file):
        # TODO use xarray instead of netcdf ?


        # Open dataset
        dataset = nc.Dataset(out_file, 'r', format="NETCDF4")

        # Initialise results with global attributes
        results = {"model": "Undefined",
                   "status": dataset.status,
                   "error_code": dataset.error_code,
                   "error_string": str(dataset.error_string),
                   "prior": {"A0": None,
                             "n": None,
                             "Q": None},
                   "posterior": {"A0": None,
                                 "n": None,
                                 "Q": None}}
        if hasattr(dataset, "model"):
            results["model"] = str(dataset.model)

        if dataset.error_code == 1:
            results["posterior"]["A0"] = []
            results["posterior"]["n"] = []
            results["posterior"]["Q"] = np.zeros(0)
            results["posterior"]["Q_ci"] = np.zeros(0)
            dataset.close()
            return results

        # Load data in reach group
        reach_group = dataset.groups["reach"]
        results["posterior"]["A0"] = reach_group.variables["A0"][:].filled(np.nan)
        results["posterior"]["n"] = reach_group.variables["n"][:].filled(np.nan)
        results["posterior"]["Q"] = reach_group.variables["Q"][:].filled(np.nan)
        results["posterior"]["Q_ci"] = reach_group.variables["Q_ci"][:, :].filled(np.nan)
        
        # Close output dataset
        dataset.close()

        return results

    def _write_output_(self, results, output_dir, single_output_per_reach=True):
        # TODO use xarray instead of netcdf ?

        if single_output_per_reach is True:
            for i, reach_id in enumerate(self._reaches_id):
                self._logger.debug("- Save results for reach %i" % reach_id)
                single_reach_results = {"model": results["model"],
                                        "status": results["status"],
                                        "error_code": results["error_code"],
                                        "error_string": results["error_string"],
                                        "prior": {"A0": results["prior"]["A0"][i],
                                                  "n": results["prior"]["n"][i],
                                                  "Q": results["prior"]["Q"][:]},
                                        "posterior": {"A0": results["posterior"]["A0"][i],
                                                      "n": results["posterior"]["n"][i],
                                                      "Q": results["posterior"]["Q"][:],
                                                      "Q_ci": results["posterior"]["Q_ci"][:, :]}}
                # TODO HERE 
                # out_file = os.path.join(output_dir, "%s_h2ivdi.nc" % reach_id)
                out_file = os.path.join(output_dir, "%s_hivdi.nc" % reach_id)
                self._write_single_reach_output_(out_file, reach_id, single_reach_results)
        else:
            raise NotImplementedError("Not implemented yet !")

    def _write_single_reach_output_(self, out_file, reach_id, results):
        # TODO use xarray instead of netcdf ?

        # Open dataset
        dataset = nc.Dataset(out_file, 'w', format="NETCDF4")
        
        # Set global attributes
        if isinstance(self._set_def, dict):
            set_name = "%s" % self._set_def["reach_id"]
        else:
            set_name = "%s-%s" % (self._set_def[0]["reach_id"], self._set_def[-1]["reach_id"])
        dataset.setncatts({"title" : "H2IVDI output",
                           "production_date" : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           "set": set_name,
                           # "exception" : exception,
                           # "obs_validity" : flags[0],
                           # "inference_status" : flags[1],
                           "status": int(results["status"]),
                           "error_code": results["error_code"],
                           "error_string": results["error_string"]})

        # Create dimensions
        dataset.createDimension("nchar", 12)
        dataset.createDimension("nt", len(self._data._reach_obs._nt))
        dataset.createDimension("nci", 2)
        dataset.createDimension("chartime", 20)
        
        # Create global variables
        nt = dataset.createVariable("nt", "i4", ("nt",))
        nt.units = "day"
        nt.long_name = "nt"
        # nt[:] = ((self._data._reach_obs.t - self._data._reach_obs.t[0]) / 86400.0).astype(int)
        nt[:] = self._data._reach_obs._nt[:]
        valid = self._data._reach_obs._valid
        if self._data._reach_obs.dates is not None:
            time = dataset.createVariable("time", "f8", ("nt",), fill_value=-999999999999.)
            time.units = "seconds since 2000-01-01 00:00:00.000"
            time.long_name = "time (UTC)"
            time[:] = (self._data._reach_obs.dates - np.datetime64("2000-01-01")) / np.timedelta64(1, "s")

            time_str = dataset.createVariable("time_str", "c", ("nt","chartime"))
            # valid_index = 0
            # for i in range(0, len(self._data._reach_obs._nt)):
            #     if np.isnat(self._data._reach_obs.dates)
            #     if bool(valid[i]) is False:
            #         time_str[i, 0] = 'N'
            #         time_str[i, 1] = 'a'
            #         time_str[i, 2] = 'T'
            #         for j in range(3, 20):
            #             time_str[i, j] = '\0'
            #     else:
            #         current_time_str = "%sZ" % str(self._data._reach_obs.dates[valid_index])
            #         valid_index += 1
            #         for j in range(0, 20):
            #             time_str[i, j] = current_time_str[j]
            for i in range(0, len(self._data._reach_obs._nt)):
                if np.isnat(self._data._reach_obs.dates[i]):
                    time_str[i, 0] = 'N'
                    time_str[i, 1] = 'a'
                    time_str[i, 2] = 'T'
                    for j in range(3, 20):
                        time_str[i, j] = '\0'
                else:
                    current_time_str = "%sZ" % str(self._data._reach_obs.dates[i])
                    for j in range(0, 20):
                        time_str[i, j] = current_time_str[j]

        # Create group
        reach_group = dataset.createGroup("reach")
        
        # Create variables in reach group
        reach_id = reach_group.createVariable("reach_id", "c", ("nchar",))
        reach_id_str = "%s" % str(reach_id)
        for i in range(0, min(len(reach_id_str), 12)):
            reach_id[i] = reach_id_str[i]
        A0 = reach_group.createVariable("A0", "f8", (), fill_value=-999999999999.)
        A0.long_name = "unobserved cross-sectional area"
        A0.units = "m^2"
        A0[0] = np.nan_to_num(results["posterior"]["A0"], copy=True, nan=-999999999999.)
        # Abar = reach_group.createVariable("Abar", "f8", (), fill_value=-999999999999.)
        # Abar.long_name = "median cross-sectional area"
        # Abar.units = "m^2"
        # Abar[0] = np.nan_to_num(estimates["Abar"], copy=True, nan=-999999999999.)
        manning = reach_group.createVariable("n", "f8", (), fill_value=-999999999999.)
        manning.long_name = "Manning coefficient"
        manning.units = "s.m^(-1/3)" 
        manning[0] = np.nan_to_num(results["posterior"]["n"], copy=True, nan=-999999999999.)
        # alpha = reach_group.createVariable("alpha", "f8", (), fill_value=-999999999999.)
        # alpha.long_name = "coefficient for the Strickler power law"
        # alpha.units = "m^(1/3)/s" 
        # alpha[0] = np.nan_to_num(results["posterior"]["alpha"], copy=True, nan=-999999999999.)
        # beta = reach_group.createVariable("beta", "f8", (), fill_value=-999999999999.)
        # beta[0] = np.nan_to_num(estimates["beta"], copy=True, nan=-999999999999.)
        # beta.long_name = "exponent for the Strickler power law"
        # beta.units = "-" 
        Q = reach_group.createVariable("Q", "f8", ("nt",), fill_value=-999999999999.)
        Q.long_name = "discharge"
        Q.units = "m^3/s" 
        Q[valid] = np.nan_to_num(results["posterior"]["Q"][:], copy=True, nan=-999999999999.)
        Q_ci = reach_group.createVariable("Q_ci", "f8", ("nt", "nci"), fill_value=-999999999999.)
        Q_ci.long_name = "discharge confidence interval (lower, upper)"
        Q_ci.units = "m^3/s" 
        Q_ci[valid, :] = np.nan_to_num(results["posterior"]["Q_ci"][:, :].T, copy=True, nan=-999999999999.)
        Qmanning = reach_group.createVariable("Qmanning", "f8", ("nt",), fill_value=-999999999999.)
        Qmanning.long_name = "discharge computed using Manning Equation (Low-Froude)"
        Qmanning.units = "m^3/s" 
        # Qmanning[:] = np.nan_to_num(estimates["Qmanning"][:], copy=True, nan=-999999999999.)

        # # Create obs group
        # # print("reach: %s" % str(reach_id))
        # # print("obs_flags:", estimates["obs_flags"])
        # obs_group = dataset.createGroup("obs_flags")
        # reach_times = obs_group.createVariable("reach_times", "i2", (), fill_value=-9)
        # reach_times[:] = estimates["obs_flags"]["reach_times"]
        # reach_times.long_name = "Flag for times values at reach scale"
        # reach_times.flag_meanings = "ok, some are missing, all are missing"
        # reach_times.flag_values = "0, 1, 2"
        # reach_wse = obs_group.createVariable("reach_wse", "i2", (), fill_value=-9)
        # reach_wse[:] = estimates["obs_flags"]["reach_wse"]
        # reach_wse.long_name = "Flag for wse values at reach scale"
        # reach_wse.flag_meanings = "ok, some are missing, all are missing"
        # reach_wse.flag_values = "0, 1, 2"
        # reach_widths = obs_group.createVariable("reach_widths", "i2", (), fill_value=-9)
        # reach_widths[:] = estimates["obs_flags"]["reach_widths"]
        # reach_widths.long_name = "Flag for widths values at reach scale"
        # reach_widths.flag_meanings = "ok, some are missing, all are missing"
        # reach_widths.flag_values = "0, 1, 2"
        # reach_slopes = obs_group.createVariable("reach_slopes", "i2", (), fill_value=-9)
        # reach_slopes[:] = estimates["obs_flags"]["reach_slopes"]
        # reach_slopes.long_name = "Flag for slopes values at reach scale"
        # reach_slopes.flag_meanings = "ok, some are missing, all are missing"
        # reach_slopes.flag_values = "0, 1, 2"
        # nodes_times = obs_group.createVariable("nodes_times", "i2", (), fill_value=-9)
        # nodes_times[:] = estimates["obs_flags"]["nodes_times"]
        # nodes_times.long_name = "Flag for times values at node scale"
        # nodes_times.flag_meanings = "ok, some are missing, all are missing"
        # nodes_times.flag_values = "0, 1, 2"
        # nodes_wse = obs_group.createVariable("nodes_wse", "i2", (), fill_value=-9)
        # nodes_wse[:] = estimates["obs_flags"]["nodes_wse"]
        # nodes_wse.long_name = "Flag for wse values at node scale"
        # nodes_wse.flag_meanings = "ok, some are missing, all are missing"
        # nodes_wse.flag_values = "0, 1, 2"
        # nodes_widths = obs_group.createVariable("nodes_widths", "i2", (), fill_value=-9)
        # nodes_widths[:] = estimates["obs_flags"]["nodes_widths"]
        # nodes_widths.long_name = "Flag for widths values at node scale"
        # nodes_widths.flag_meanings = "ok, some are missing, all are missing"
        # nodes_widths.flag_values = "0, 1, 2"
        
        # Close output dataset
        dataset.close()

    def _write_multiple_reaches_output_(self, out_file, reach_ids, results):
        # TODO use xarray instead of netcdf ?

        # Open dataset
        dataset = Dataset(out_file, 'w', format="NETCDF4")
        
        # Set global attributes
        dataset.setncatts({"title" : "H2IVDI output",
                           "production_date" : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           "set": set_name,
                           # "exception" : exception,
                           # "obs_validity" : flags[0],
                           # "inference_status" : flags[1],
                           "status" : results["status"],
                           "error_code": results["error_code"],
                           "error_string": results["error_string"]})

        # Create dimensions
        dataset.createDimension("nchar", 12)
        dataset.createDimension("nt", len(t))
        dataset.createDimension("num_reaches", len(reach_ids))
        
        # Create global variables
        nt = dataset.createVariable("nt", "i4", ("nt",))
        nt.units = "day"
        nt.long_name = "nt"
        nt[:] = ((self._data._reach_obs.t - self._data._reach_obs.t[0]) / 86400.0).astype(int)
        if self._data._reach_obs.dates is not None:
            time = dataset.createVariable("time", "f8", ("nt",), fill_value=-999999999999.)
            time.units = "seconds since 2000-01-01 00:00:00.000"
            time.long_name = "time (UTC)"
            time[:] = (self._data._reach_obs.dates - np.datetime64("2000-01-01")) / np.timedelta64(1, "s")

        # Create group
        reach_group = dataset.createGroup("reach")
        
        # Create variables in reach group
        reach_id = reach_group.createVariable("reach_id", "c", ("nchar", "num_reaches"))
        for i in range(0, len(reach_ids)):
            reach_id_str = "%s" % str(reach_ids[i])
            for j in range(0, min(len(reach_id_str), 12)):
                reach_id[j, i] = reach_id_str[j]
        A0 = reach_group.createVariable("A0", "f8", ("num_reaches",), fill_value=-999999999999.)
        A0.long_name = "unobserved cross-sectional area"
        A0.units = "m^2"
        A0[:] = np.nan_to_num(results["posterior"]["A0"], copy=True, nan=-999999999999.)
        # Abar = reach_group.createVariable("Abar", "f8", (), fill_value=-999999999999.)
        # Abar.long_name = "median cross-sectional area"
        # Abar.units = "m^2"
        # Abar[0] = np.nan_to_num(estimates["Abar"], copy=True, nan=-999999999999.)
        manning = reach_group.createVariable("n", "f8", ("num_reaches",), fill_value=-999999999999.)
        manning.long_name = "Manning coefficient"
        manning.units = "s.m^(-1/3)" 
        manning[:] = np.nan_to_num(results["posterior"]["n"], copy=True, nan=-999999999999.)
        # alpha = reach_group.createVariable("alpha", "f8", (), fill_value=-999999999999.)
        # alpha.long_name = "coefficient for the Strickler power law"
        # alpha.units = "m^(1/3)/s" 
        # alpha[0] = np.nan_to_num(estimates["alpha"], copy=True, nan=-999999999999.)
        # beta = reach_group.createVariable("beta", "f8", (), fill_value=-999999999999.)
        # beta[0] = np.nan_to_num(estimates["beta"], copy=True, nan=-999999999999.)
        # beta.long_name = "exponent for the Strickler power law"
        # beta.units = "-" 
        Q = reach_group.createVariable("Q", "f8", ("nt", "num_reaches"), fill_value=-999999999999.)
        Q.long_name = "discharge"
        Q.units = "m^3/s" 
        # print("reachid=", reachid)
        # # print("estimates['Q'].size=", estimates["Q"].size)
        # # print("nt=", kt)
        # Q[:, :] = np.nan_to_num(results["posterior"]["Q"][:, :], copy=True, nan=-999999999999.)
        # Qmanning = reach_group.createVariable("Qmanning", "f8", ("nt",), fill_value=-999999999999.)
        # Qmanning.long_name = "discharge computed using Manning Equation (Low-Froude)"
        # Qmanning.units = "m^3/s" 
        # # Qmanning[:] = np.nan_to_num(estimates["Qmanning"][:], copy=True, nan=-999999999999.)

        # # Create obs group
        # # print("reach: %s" % str(reach_id))
        # # print("obs_flags:", estimates["obs_flags"])
        # obs_group = dataset.createGroup("obs_flags")
        # reach_times = obs_group.createVariable("reach_times", "i2", (), fill_value=-9)
        # reach_times[:] = estimates["obs_flags"]["reach_times"]
        # reach_times.long_name = "Flag for times values at reach scale"
        # reach_times.flag_meanings = "ok, some are missing, all are missing"
        # reach_times.flag_values = "0, 1, 2"
        # reach_wse = obs_group.createVariable("reach_wse", "i2", (), fill_value=-9)
        # reach_wse[:] = estimates["obs_flags"]["reach_wse"]
        # reach_wse.long_name = "Flag for wse values at reach scale"
        # reach_wse.flag_meanings = "ok, some are missing, all are missing"
        # reach_wse.flag_values = "0, 1, 2"
        # reach_widths = obs_group.createVariable("reach_widths", "i2", (), fill_value=-9)
        # reach_widths[:] = estimates["obs_flags"]["reach_widths"]
        # reach_widths.long_name = "Flag for widths values at reach scale"
        # reach_widths.flag_meanings = "ok, some are missing, all are missing"
        # reach_widths.flag_values = "0, 1, 2"
        # reach_slopes = obs_group.createVariable("reach_slopes", "i2", (), fill_value=-9)
        # reach_slopes[:] = estimates["obs_flags"]["reach_slopes"]
        # reach_slopes.long_name = "Flag for slopes values at reach scale"
        # reach_slopes.flag_meanings = "ok, some are missing, all are missing"
        # reach_slopes.flag_values = "0, 1, 2"
        # nodes_times = obs_group.createVariable("nodes_times", "i2", (), fill_value=-9)
        # nodes_times[:] = estimates["obs_flags"]["nodes_times"]
        # nodes_times.long_name = "Flag for times values at node scale"
        # nodes_times.flag_meanings = "ok, some are missing, all are missing"
        # nodes_times.flag_values = "0, 1, 2"
        # nodes_wse = obs_group.createVariable("nodes_wse", "i2", (), fill_value=-9)
        # nodes_wse[:] = estimates["obs_flags"]["nodes_wse"]
        # nodes_wse.long_name = "Flag for wse values at node scale"
        # nodes_wse.flag_meanings = "ok, some are missing, all are missing"
        # nodes_wse.flag_values = "0, 1, 2"
        # nodes_widths = obs_group.createVariable("nodes_widths", "i2", (), fill_value=-9)
        # nodes_widths[:] = estimates["obs_flags"]["nodes_widths"]
        # nodes_widths.long_name = "Flag for widths values at node scale"
        # nodes_widths.flag_meanings = "ok, some are missing, all are missing"
        # nodes_widths.flag_values = "0, 1, 2"
        
        # Close output dataset
        dataset.close()

    def _write_failed_output_(self, output_dir, error_code, single_output_per_reach=True):
        # TODO use xarray instead of netcdf ?

        if single_output_per_reach is True:
            for i, reach_id in enumerate(self._reaches_id):
                self._logger.debug("- Save empty results for reach %i" % reach_id)
                # out_file = os.path.join(output_dir, "%s_h2ivdi.nc" % reach_id)
                out_file = os.path.join(output_dir, "%s_hivdi.nc" % reach_id)
                self._write_single_reach_failed_output_(out_file, reach_id, error_code)
        else:
            raise NotImplementedError("Not implemented yet !")

    def _write_single_reach_failed_output_(self, out_file, reach_id, error_code):
        # TODO use xarray instead of netcdf ?

        # Open dataset
        dataset = nc.Dataset(out_file, 'w', format="NETCDF4")
        
        # Set global attributes
        if isinstance(self._set_def, dict):
            set_name = "%s" % self._set_def["reach_id"]
        else:
            set_name = "%s-%s" % (self._set_def[0]["reach_id"], self._set_def[-1]["reach_id"])
        dataset.setncatts({"title" : "H2IVDI output",
                           "production_date" : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           "set": set_name,
                           # "exception" : exception,
                           # "obs_validity" : flags[0],
                           # "inference_status" : flags[1],
                           "status": 0,
                           "error_code": error_code,
                           "error_string": error_string_from_code(error_code)})
        
        if error_code == 1:
            dataset.createDimension("nchar", 12)
            dataset.createDimension("nt", 0)
            dataset.createDimension("nci", 2)
            dataset.createDimension("chartime", 20)
            dataset.close()
            return

        # Create dimensions
        dataset.createDimension("nchar", 12)
        dataset.createDimension("nt", len(self._data._reach_obs._nt))
        dataset.createDimension("nci", 2)
        dataset.createDimension("chartime", 20)
        
        # Create global variables
        nt = dataset.createVariable("nt", "i4", ("nt",))
        nt.units = "day"
        nt.long_name = "nt"
        # nt[:] = ((self._data._reach_obs.t - self._data._reach_obs.t[0]) / 86400.0).astype(int)
        nt[:] = self._data._reach_obs._nt[:]
        valid = self._data._reach_obs._valid
        if self._data._reach_obs.dates is not None:
            time = dataset.createVariable("time", "f8", ("nt",), fill_value=-999999999999.)
            time.units = "seconds since 2000-01-01 00:00:00.000"
            time.long_name = "time (UTC)"
            time[:] = (self._data._reach_obs.dates - np.datetime64("2000-01-01")) / np.timedelta64(1, "s")

            time_str = dataset.createVariable("time_str", "c", ("nt","chartime"))
            # valid_index = 0
            # for i in range(0, len(self._data._reach_obs._nt)):
            #     if bool(valid[i]) is False:
            #         time_str[i, 0] = 'N'
            #         time_str[i, 1] = 'a'
            #         time_str[i, 2] = 'T'
            #         for j in range(3, 20):
            #             time_str[i, j] = '\0'
            #     else:
            #         current_time_str = "%sZ" % str(self._data._reach_obs.dates[valid_index])
            #         valid_index += 1
            #         for j in range(0, 20):
            #             time_str[i, j] = current_time_str[j]
            for i in range(0, len(self._data._reach_obs._nt)):
                if np.isnat(self._data._reach_obs.dates[i]):
                    time_str[i, 0] = 'N'
                    time_str[i, 1] = 'a'
                    time_str[i, 2] = 'T'
                    for j in range(3, 20):
                        time_str[i, j] = '\0'
                else:
                    current_time_str = "%sZ" % str(self._data._reach_obs.dates[i])
                    for j in range(0, 20):
                        time_str[i, j] = current_time_str[j]

        # Create group
        reach_group = dataset.createGroup("reach")
        
        # Create variables in reach group
        reach_id = reach_group.createVariable("reach_id", "c", ("nchar",))
        reach_id_str = "%s" % str(reach_id)
        for i in range(0, min(len(reach_id_str), 12)):
            reach_id[i] = reach_id_str[i]
        A0 = reach_group.createVariable("A0", "f8", (), fill_value=-999999999999.)
        A0.long_name = "unobserved cross-sectional area"
        A0.units = "m^2"
        manning = reach_group.createVariable("n", "f8", (), fill_value=-999999999999.)
        manning.long_name = "Manning coefficient"
        manning.units = "s.m^(-1/3)" 
        Q = reach_group.createVariable("Q", "f8", ("nt",), fill_value=-999999999999.)
        Q.long_name = "discharge"
        Q.units = "m^3/s" 
        Q_ci = reach_group.createVariable("Q_ci", "f8", ("nt", "nci"), fill_value=-999999999999.)
        Q_ci.long_name = "discharge confidence interval (lower, upper)"
        Q_ci.units = "m^3/s" 
        Qmanning = reach_group.createVariable("Qmanning", "f8", ("nt",), fill_value=-999999999999.)
        Qmanning.long_name = "discharge computed using Manning Equation (Low-Froude)"
        Qmanning.units = "m^3/s" 
        
        # Close output dataset
        dataset.close()

    def _plot_validation_(self, trace, log_file):
        data = self._data
        fig = plt.figure()
        if trace is not None:
            qm_post_ci = np.zeros((2, qm_post.size))
            for it in range(0, qm_post.size):
                qm_post_ci[:, it] = az.hdi(np.ravel(trace["Qin"][:, :, it]), hdi_prob=0.95)
            plt.fill_between(np.arange(0, qm_post_ci.shape[1]), qm_post_ci[0, :], qm_post_ci[1, :], color="gray", alpha=0.5)
        if qm_post_ci is not None:
            plt.fill_between(data.dates, qm_post_ci[0, :], qm_post_ci[1, :], color="gray", alpha=0.5)
        if hasattr(data, "_insitu_data"):
            for key in data._insitu_data:
                t_obs = data._insitu_data["key"]["t"]
                q_obs = data._insitu_data["key"]["Q"]
                plt.plot(t_obs, q_obs, 'r.', label="in-situ (%i)" % key)
        # if show_overbank_discharge:
        #     channel_indices = []
        #     for it in range(0, data.H.shape[0]):
        #         if np.all(data.H[it, :] <= data.Hl[1, :]):
        #             channel_indices.append(it)
        #     plt.plot(np.mean(data.Q, axis=1), ".", color="orange", label="target (overbank flow)")
        #     plt.plot(np.arange(0, data.H.shape[0])[channel_indices], np.mean(data.Q, axis=1)[channel_indices], 'r.', label="target (channel flow)")
        # else:
            # plt.plot(np.mean(data.Q, axis=1), 'r.', label="target")
        # plt.plot(np.mean(data.Q, axis=1), 'r.', label="target")
        # plt.axhline(np.mean(np.mean(data.Q, axis=1)), color='r', ls='--')
        plt.axhline(data._QmeanModel, color='k', ls='--', label="QMeanModel")
        plt.plot(data.dates, qm_prior, 'b-', label="prior")
        plt.axhline(np.mean(qm_prior), color='b', ls='--')
        plt.plot(data.dates, qm_post, 'g-', label="posterior")
        plt.axhline(np.mean(qm_post), color='g', ls='--')
        plt.legend()
        # if title is not None:
        #     plt.title(title)
        plt.tight_layout()
        plt.savefig(fname)
        plt.close(fig)        