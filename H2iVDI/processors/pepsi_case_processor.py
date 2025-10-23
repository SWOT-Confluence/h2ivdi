import configparser
import json
import matplotlib.pyplot as plt
import numpy as np
import os

from H2iVDI.core.metrics import compute_metric
from H2iVDI.processors.case_processor import CaseProcessor
from H2iVDI.chains import BayesianChain, Bayesian2StepsChain
from H2iVDI.io.pepsi_dataset import PepsiDataset

class PepsiCaseProcessor(CaseProcessor):

    def __init__(self, config):
        super().__init__()

        # Store configuration file
        if isinstance(config, str):
            self._config_file = config
            self._config = self._default_config_dict_()
        elif isinstance(config, dict):
            self._config_file = None
            self._config = config
            self._check_config_()
        else:
            raise ValueError("'config' must be either a path to file or a dictionary")

    def prepro(self):
        # raise NotImplementedError("Not implemented yet !")

        # Load configuration file
        if self._config_file is not None:
            self._logger.info("Load configuration file: %s" % self._config_file)
            self._load_config_file_(self._config_file)

        # Create run_dir
        if self._config["run_dir"] is None:
            raise RuntimeError("'run_dir' is not defined in the case json file")
        if not os.path.isdir(self._config["run_dir"]):
            dirname = os.path.dirname(self._config["run_dir"])
            if not os.path.isdir(dirname):
                raise IOError("Cannot create run directory in non existing directory: %s" % dirname)
            os.mkdir(self._config["run_dir"])

        print(self._config)

        # Load PEPSI data
        data_file = os.path.expandvars(self._config["data_file"])
        self._data = PepsiDataset()
        error_code = self._data.load(data_file, reaches_selection=self._config["reaches_selection"],
                                     times_selection=self._config["times_selection"])
        if error_code != 0:
            return error_code

        # Compute effective section
        if self._data.reach.nt > 0:
            self._logger.info("- Compute effective sections")
            self._data.reach.compute_effective_sections()
            if self._logger._debug_level > 0 or self._config["plots"]["effective_sections"] is True:
                for r in range(0, self._data.reach.H.shape[1]):
                    plt.scatter(self._data.reach.H[:, r], self._data.reach.W[:, r], c="gray", label="raw data")
                    plt.plot(self._data.reach.He[:, r], self._data.reach.We[:, r], "b--", label="effective section")
                    plt.plot(self._data.reach.He[0, r], self._data.reach.We[0, r], "bd")
                    plt.plot(self._data.reach.He[1, r], self._data.reach.We[1, r], "rd")
                    plt.plot(self._data.reach.He[2, r], self._data.reach.We[2, r], "gd")
                    plt.title("Reach #%i" % (r+1))
                    plt.legend()
                    if self._config["run_dir"] is not None:
                        plt.savefig(os.path.join(self._config["run_dir"], "effective_section_reach%03i.png" % (r+1)))
                    else:
                        plt.show()
                    plt.close(plt.gcf())

        return 0

    def run(self):

        self._logger.debug("[START] Run PEPSI case processor")

        if self._config["run_dir"] is not None:
            self._logger.debug("rundir is %s" % self._config["run_dir"])

        # Create chain parameters
        chain_parameters = {"model": self._config["model"],
                            "plots": {"inference": True,
                                      "validation": True}}
        if "priors" in self._config:
            chain_parameters["priors"] = self._config["priors"]
        if "q0_method" in self._config:
            chain_parameters["q0_method"] = self._config["q0_method"]

        # Create chain
        self._logger.info("Inference chain: %s" % self._config["inference_chain"])
        if self._config["inference_scale"] == "reach":
            if self._config["inference_chain"] == "bayesian":
                chain = BayesianChain(self._data.reach, chain_parameters)
            elif self._config["inference_chain"] == "bayesian2steps":
                chain = Bayesian2StepsChain(self._data.reach, chain_parameters)
            else:
                raise ValueError("Wrong inference chain: %s" % repr(self._config["inference_chain"]))    
        elif self._config["inference_scale"] == "node":
            self._logger.info("- Compute nodes slopes:")
            self._data.node.compute_slopes()
            self._logger.info("- Compute effective sections:")
            self._data.node.compute_effective_sections()
            chain = BayesianChain(self._data.node, chain_parameters)
        elif self._config["inference_scale"] == "mid":
            self._data.compute_mid_scale()
            self._data.node.compute_effective_sections()
            chain = BayesianChain(self._data.node, chain_parameters)
        else:
            raise ValueError("Wrong inference scale: %s" % repr(self._config["inference_scale"]))

        # Run calibration
        self._calibration_results, error_code = chain.calibrate(rundir=self._config["run_dir"])
        if error_code != 0: return error_code

        self._logger.debug("[ END ] Run PEPSI case processor")

        return 0

    def postpro(self, output_dir):

        # print(self._calibration_results)

        # Create output file
        if output_dir is None:
            if self._config["output_dir"] is not None:
                output_dir = self._config["output_dir"]
        # if output_dir is not None:
        #     if self._data.reach.nt > 0:
        #         self._write_output_(self._calibration_results, output_dir)

        # Create validation plot
        qm_prior = self._calibration_results["prior"]["Q"]
        qm_post = self._calibration_results["posterior"]["Q"]
        qm_post_ci = self._calibration_results["posterior"]["Q_ci"]
        if self._config["inference_scale"] == "reach":
            plot_file = None
            if output_dir is not None:
                plot_file = os.path.join(self._config["run_dir"], "discharge_validation.png")
            print("PLOT_FILE:", plot_file)
            self._plot_discharge_validation_(self._data.reach, qm_post, plot_file=plot_file, qm_prior=qm_prior, qm_post_ci=qm_post_ci)
        elif self._config["inference_scale"] == "node":
            plot_file = None
            if output_dir is not None:
                plot_file = os.path.join(self._config["run_dir"], "discharge_validation.png")
            # print("PLOT_FILE:", plot_file)
            # qm_prior_reaches = np.zeros(self._data.reach.t.size, self._data.reach.x.size)
            # for r in range(0, self._data.reach.x.size):
            #     nodes_selection = 
            #     qm_prior_reaches[:, r] = 
            self._plot_discharge_validation_(self._data.reach, qm_post, plot_file=plot_file, qm_prior=qm_prior, qm_post_ci=qm_post_ci)

        else:
            raise NotImplementedError("Not implemented yet !")

        # Compute validation metrics
        if self._config["inference_scale"] == "reach":
            plot_file = None
            if self._config["run_dir"] is not None:
                plot_file = os.path.join(self._config["run_dir"], "discharge_validation.png")
            nrmse, nse = self._compute_validation_metrics_(self._data.reach, qm_post)
            self._calibration_results["validation"] = {"nrmse": nrmse, "nse": nse}
        elif self._config["inference_scale"] == "node":
            plot_file = None
            if self._config["run_dir"] is not None:
                plot_file = os.path.join(self._config["run_dir"], "discharge_validation.png")
            nrmse, nse = self._compute_validation_metrics_(self._data.reach, qm_post)
            self._calibration_results["validation"] = {"nrmse": nrmse, "nse": nse}
        else:
            raise NotImplementedError("Not implemented yet !")
        
        return self._calibration_results, 0

    # def load_data(self, fname: str):
    #     self._data = PepsiDataset(fname)

    def _default_config_dict_(self):
        config_dict = {"data_file": None,
                       "reaches_selection": "all",
                       "times_selection": "all",
                       "inference_chain": "bayesian",
                       "inference_scale": "reach",
                       "q0_method": "low-froude",
                       "model": "swst3lfb",
                       "run_dir": None,
                       "output_dir": None,
                       "plots": {"effective_sections": False,
                                 "inference": True,
                                 "validation": True}}
        return config_dict

    def _load_config_file_(self, config_file):

        # Default configuration
        self._config = self._default_config_dict_()

        if os.path.splitext(config_file)[1].lower() == ".cfg":
            raise NotImplementedError("Not implemented yet !")

            # # Read configuration file using configParser
            # config = configparser.ConfigParser()
            # config.read(config_file)
            # print("config read:", config["DEFAULT"])
            # print("conf.sections()=", config.sections())
            # # default_section = config["DEFAULT"]
            # # print(dir(default_section))
            # # print(list(default_section.keys()))
            # choice = input()
            # for (key, value) in config.items("DEFAULT"):
            # # for key in default_section.keys():
            #     print(key, key in self._config)
            #     # TODO handle sections key and subkeys
            #     # if isinstance(config[key], dict):
            #     #     if key in self._config:
            #     #     for section_key in config[key]:
                    
            #     if key in self._config:
            #         self._config = value

        elif os.path.splitext(config_file)[1].lower() == ".json":

            # Read configuration file using configParser
            with open(config_file, "r") as fp:
                config = json.load(fp)

            # print("Default config keys:", self._config.keys())
            for key in config.keys():
                # print(key, key in self._config)
                if key == "priors":
                    self._config[key] = config[key]
                elif key in self._config:
                    self._config[key] = config[key]

        self._check_config_()

    def _check_config_(self):
        default_config_dict = self._default_config_dict_()

        for key in default_config_dict:
            if key not in self._config:
                
                if isinstance(default_config_dict[key], dict):

                    # Add section
                    self._config[key] = default_config_dict[key].copy()

                else:

                    self._config[key] = default_config_dict[key]

            else:

                if isinstance(default_config_dict[key], dict):

                    for section_key in default_config_dict[key]:
                        if section_key not in self._config[key]:
                            self._config[key][section_key] = default_config_dict[key][section_key]



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
                out_file = os.path.join(output_dir, "%s_h2ivdi.nc" % reach_id)
                self._write_single_reach_output_(out_file, reach_id, single_reach_results)
        else:
            raise NotImplementedError("Not implemented yet !")

    def _plot_discharge_validation_(self, data, qm_post, plot_file:str=None, qm_prior=None, qm_post_ci=None):

        nrows = int(np.ceil(data.nx/4))
        ncols = min(data.nx, 4)
        fig = plt.figure(figsize=(ncols*4, nrows*4))
        gs = fig.add_gridspec(nrows, ncols)
        for r in range(0, data.nx):
            row = r//4
            col = r%4

            # Compute metrics
            nrmse = compute_metric(qm_post, data._Q[:, r], "nrmse")
            nse = compute_metric(qm_post, data._Q[:, r], "nse")
            ax = fig.add_subplot(gs[row, col])
            if qm_post_ci is not None:
                ax.fill_between(data.dates, qm_post_ci[0, :], qm_post_ci[1, :], color="gray", alpha=0.5)
            ax.plot(data.dates, data._Q[:, r], 'r.', label="target")
            ax.axhline(data._QmeanModel, color='k', ls='--', label="QMeanModel")
            ax.plot(data.dates, qm_prior, 'b-', label="prior")
            ax.axhline(np.mean(qm_prior), color='b', ls='--')
            ax.plot(data.dates, qm_post, 'g-', label="posterior")
            ax.axhline(np.mean(qm_post), color='g', ls='--')
            if r == 0:
                ax.legend()
            ax.set_title("#%i, nrme=%.3f, nse=%.3f" % (r+1, nrmse, nse))
            # if title is not None:
            #     plt.title(title)
        plt.tight_layout()
        if plot_file is not None:
            plt.savefig(plot_file)
        else:
            plt.show()
        plt.close(fig)

    def _compute_validation_metrics_(self, data, qm_post):

        # Compute metrics on spatially averaged discharge
        nrmse = compute_metric(qm_post, np.mean(data._Q, axis=1), "nrmse")
        nse = compute_metric(qm_post, np.mean(data._Q, axis=1), "nse")

        return nrmse, nse


        
