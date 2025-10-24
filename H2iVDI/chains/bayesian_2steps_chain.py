import arviz as az
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
#from tqdm import tqdm
from tqdm.autonotebook import tqdm
import xarray as xr

from H2iVDI.core import error_code_from_string, error_string_from_code
from H2iVDI.core.distributions import new_distribution
from H2iVDI.models import new_model
from .inference_chain import InferenceChain

class Bayesian2StepsChain(InferenceChain):

    def __init__(self, data, parameters):

        self._logger = logging.getLogger("H2iVDI")
        self._data = data
        self._parameters = parameters
        self._model = None
        self._priors = None

    def calibrate(self, rundir: str=None):

        self._logger.debug("[START] Bayesian chain calibration")

        if self._model is None:
            self._apply_parameters_()

        # STEP 0: check data
        if self._data.nt == 0:
            self._logger.error("No valid data profile")
            results = {"model": self._model.name,
                       "error_code": error_code_from_string("no_valid_observation_profile"),
                       "error_string": "No valid observation profile",
                       "status": False,
                       "prior": {"A0": np.ones(self._data.nx) * np.nan,
                                 "n": np.ones(self._data.nx) * np.nan,
                                 "Q": np.ones(self._data.nt) * np.nan},
                       "posterior": {"A0": np.ones(self._data.nx) * np.nan,
                                     "n": np.ones(self._data.nx) * np.nan,
                                     "Q": np.ones(self._data.nt) * np.nan,
                                     "Q_ci": np.ones((2, self._data.nt)) * np.nan}}
            return results, error_code_from_string("no_valid_observation_profile")

        # STEP 1: Sample for channel flow profiles
        self._logger.info("Compute kch")
        self._logger.debug("- Retrieve times with flow restricted in channel")
        channel_flow_indices = []
        for it in range(0, self._data.H.shape[0]):
            if np.all(self._data.H[it, :] <= self._data.He[1, :]):
                channel_flow_indices.append(it)
        data_channel_flow = self._data.copy()
        data_channel_flow.time_selection(channel_flow_indices)
        data_channel_flow._dAr = self._data._dAr[channel_flow_indices, :]
        data_channel_flow._Wr = self._data._Wr[channel_flow_indices, :]
        self._original_data = self._data
        self._data = data_channel_flow
        self._model.set_data(data_channel_flow)
        self._logger.debug("- Sample variables space")
        trace, error_code = self._sample_(*tuple(self._parameters["sample_sizes"]), rundir=rundir)
        if error_code != 0:
            results = {"model": self._model.name,
                       "error_code": error_code,
                       "error_string": error_string_from_code(error_code),
                       "status": False,
                       "prior": {"A0": np.ones(self._data.nx) * np.nan,
                                 "n": np.ones(self._data.nx) * np.nan,
                                 "Q": np.ones(self._data.nt) * np.nan},
                       "posterior": {"A0": np.ones(self._data.nx) * np.nan,
                                     "n": np.ones(self._data.nx) * np.nan,
                                     "Q": np.ones(self._data.nt) * np.nan,
                                     "Q_ci": np.ones((2, self._data.nt)) * np.nan}}
            return results, error_code

        if rundir is not None:
            self._write_trace_(os.path.join(rundir, "mc_trace1.nc"), trace)
        if np.all(np.ravel(np.isnan(trace["cost"]))):
            self._logger.error("All NaN cost detected")
        if np.any(np.ravel(np.isnan(trace["cost"]))):
            self._logger.error("NaN cost detected")

        # Compute prior and posterior for kch
        self._logger.debug("- Compute prior and posterior for kch")
        C_prior = np.sum(np.ravel(trace["prior_pdf"]))
        if C_prior < 1e-15:
            self._logger.error("C_prior is zero")
            results = {"model": self._model.name,
                       "error_code": error_code_from_string("null_cprior_or_cpost"),
                       "error_string": "Null C_prior or C_post",
                       "status": False,
                       "prior": {"A0": np.ones(self._data.nx) * np.nan,
                                 "n": np.ones(self._data.nx) * np.nan,
                                 "Q": np.ones(self._data.nt) * np.nan},
                       "posterior": {"A0": np.ones(self._data.nx) * np.nan,
                                     "n": np.ones(self._data.nx) * np.nan,
                                     "Q": np.ones(self._data.nt) * np.nan,
                                     "Q_ci": np.ones((2, self._data.nt)) * np.nan}}
            return results, error_code_from_string("null_cprior_or_cpost")

        k0ch_prior = np.sum(np.ravel(trace["k0"]) * np.ravel(trace["prior_pdf"])) / C_prior
        lh = np.exp(-trace["cost"] / np.min(np.ravel(trace["cost"])))
        if np.any(np.ravel(np.isnan(lh))):
            self._logger.error("NaN Likelihood detected")
        C_post = np.sum(np.ravel(lh * trace["prior_pdf"]))
        k0ch_post = np.sum(np.ravel(trace["k0"]) * np.ravel(lh * trace["prior_pdf"])) / C_post

        # STEP 2: Sample for all profiles
        self._logger.info("Compute other variables")
        self._data = self._original_data
        self._model.set_data(self._data)
        kch = k0ch_post * self._model._k1
        self._model.set_kch(kch)
        self._priors["k0"] = new_distribution("Beta", "k0", a=1.25, b=1.25, bounds=(10.0, k0ch_post))
        self._logger.debug("- Sample variables space")
        trace, error_code = self._sample_(*tuple(self._parameters["sample_sizes"]), rundir=rundir)
        if error_code != 0:
            results = {"model": self._model.name,
                       "error_code": error_code,
                       "error_string": error_string_from_code(error_code),
                       "status": False,
                       "prior": {"A0": np.ones(self._data.nx) * np.nan,
                                 "n": np.ones(self._data.nx) * np.nan,
                                 "Q": np.ones(self._data.nt) * np.nan},
                       "posterior": {"A0": np.ones(self._data.nx) * np.nan,
                                     "n": np.ones(self._data.nx) * np.nan,
                                     "Q": np.ones(self._data.nt) * np.nan,
                                     "Q_ci": np.ones((2, self._data.nt)) * np.nan}}
            return results, error_code

        if rundir is not None:
            self._write_trace_(os.path.join(rundir, "mc_trace1.nc"), trace)
        if np.all(np.ravel(np.isnan(trace["cost"]))):
            self._logger.error("All NaN cost detected")
        if np.any(np.ravel(np.isnan(trace["cost"]))):
            self._logger.error("NaN cost detected")

        # Compute priors and posteriors
        self._logger.debug("- Compute prior and posterior pdfs (Step 1)")
        C_prior = np.sum(np.ravel(trace["prior_pdf"]))
        if C_prior < 1e-15:
            self._logger.error("C_prior is zero")
            results = {"model": self._model.name,
                       "error_code": error_code_from_string("null_cprior_or_cpost"),
                       "error_string": "Null C_prior or C_post",
                       "status": False,
                       "prior": {"A0": np.ones(self._data.nx) * np.nan,
                                 "n": np.ones(self._data.nx) * np.nan,
                                 "Q": np.ones(self._data.nt) * np.nan},
                       "posterior": {"A0": np.ones(self._data.nx) * np.nan,
                                     "n": np.ones(self._data.nx) * np.nan,
                                     "Q": np.ones(self._data.nt) * np.nan,
                                     "Q_ci": np.ones((2, self._data.nt)) * np.nan}}
            return results, error_code_from_string("null_cprior_or_cpost")

        h0_prior = np.sum(np.ravel(trace["h0"]) * np.ravel(trace["prior_pdf"])) / C_prior
        k0_prior = np.sum(np.ravel(trace["k0"]) * np.ravel(trace["prior_pdf"])) / C_prior
        qm_prior = np.zeros(self._data.H.shape[0])
        # print("COST:", np.min(np.ravel(trace["cost"]))), np.nanmin(np.ravel(trace["cost"]))
        lh = np.exp(-trace["cost"] / np.min(np.ravel(trace["cost"])))
        if np.any(np.ravel(np.isnan(lh))):
            self._logger.error("NaN Likelihood detected")
        C_post = np.sum(np.ravel(lh * trace["prior_pdf"]))
        h0_post = np.sum(np.ravel(trace["h0"]) * np.ravel(lh * trace["prior_pdf"])) / C_post
        k0_post = np.sum(np.ravel(trace["k0"]) * np.ravel(lh * trace["prior_pdf"])) / C_post
        qm_post = np.zeros(self._data.H.shape[0])
        for it in range(0, self._data.H.shape[0]):
            qm_prior[it] = np.sum(np.ravel(trace["Qin"][:, :, it]) * np.ravel(trace["prior_pdf"]))
            qm_post[it] = np.sum(np.ravel(trace["Qin"][:, :, it]) * np.ravel(lh * trace["prior_pdf"]))
            #qm[it] = np.sum(100.0 * np.ravel(tracek["prior_pdf"]))
        qm_prior /= C_prior
        qm_post /= C_post
        trace["lh"] = lh
        self._logger.debug("  - h0(0):%f, h0(*): %f" % (h0_prior, h0_post))
        self._logger.debug("  - k0(0):%f, k0(*): %f" % (k0_prior, k0_post))
        self._logger.debug("  - q0(0):%f, q0(*): %f" % (np.mean(qm_prior), np.mean(qm_post)))

        # Compute confidence interval
        post_pdf = np.ravel(lh * trace["prior_pdf"]) / C_post
        isort = np.argsort(post_pdf)
        post_cdf = np.cumsum(post_pdf[isort])
        # print(post_cdf[-1])
        # plt.plot(np.cumsum(post_pdf[isort]), "k-")
        selected67 = np.ravel(np.argwhere(post_cdf > 0.3))
        selected95 = np.ravel(np.argwhere(post_cdf > 0.05))
        # selected = isort[post_pdf[isort] > 0.05]
        # plt.axvline(selected67[0], c="b", ls="--")
        # plt.axvline(selected95[0], c="orange", ls="--")
        # plt.show()
        qm_post_95 = np.zeros((2, self._data.H.shape[0]))
        shape01 = trace["Qin"].shape[0] * trace["Qin"].shape[1]
        shape2 = trace["Qin"].shape[2]
        Qin_selected = trace["Qin"].reshape((shape01, shape2))[isort[selected95[0]:], :]
        qm_post_95[0, :] = np.min(Qin_selected, axis=0)
        qm_post_95[1, :] = np.max(Qin_selected, axis=0)

        # Make plots
        if np.any(np.array([self._parameters["plots"][key] for key in self._parameters["plots"]])) and rundir is not None:
            self._logger.debug("- Generate plots")
        plot_file = None
        if self._parameters["plots"]["inference"] is True:
            if rundir is not None:
                plot_file = os.path.join(rundir, "distributions.png")
            self._plot_distributions_(trace, plot_file=plot_file)
            if rundir is not None:
                plot_file = os.path.join(rundir, "discharge_comparison.png")
            self._plot_discharge_comparison_(qm_prior, qm_post, plot_file=plot_file, qm_post_ci=qm_post_95)
        # if self._parameters["plots"]["validation"] is True:
        #     if rundir is not None:
        #         plot_file = os.path.join(rundir, "validation.png")
        #     self._plot_validation_(qm_prior, qm_post, plot_file, qm_post_ci=qm_post_95)
        self._logger.debug("[ END ] Bayesian chain calibration")

        # Compute manning and A0 priors and posteriors
        A0_prior = h0_prior * self._model._h1 * self._data._We[0]
        n_prior = 1.0 / (k0_prior * self._model._k1)
        A0_post = h0_post * self._model._h1 * self._data._We[0]
        n_post = 1.0 / (k0_post * self._model._k1)

        status = np.isfinite(C_prior) and np.isfinite(C_post)
        if status == True:
            error_code = 0
            error_string = "No error"
        else:
            error_code = error_code_from_string("nan_cprior_or_cpost")
            error_string = "NaN C_prior or C_post",

        results = {"model": self._model.name,
                   "status": status,
                   "error_code": error_code,
                   "error_string": error_string,
                   "prior": {"A0": A0_prior,
                             "n": n_prior,
                             "Q": qm_prior},
                   "posterior": {"A0": A0_post,
                                 "n": n_post,
                                 "Q": qm_post,
                                 "Q_ci": qm_post_95}}

        return results, error_code


    def predict(self):
        raise RuntimeError("Method must be subclassed as BayesianChain is a base class")

    def _apply_parameters_(self):

        # Check sample sizes
        if not "sample_sizes" in self._parameters:
            self._parameters["sample_sizes"] = [50, 50]

        # Create model
        if "model" in self._parameters:
            model_id = self._parameters["model"]
        else:
            model_id = "swst3lfb"
        self._model = new_model(model_id, self._data, **self._parameters)

        # Initialise prior distributions
        if not "priors" in self._parameters:
            self._priors = {"h0": new_distribution("Beta", "h0", a=1.01, b=1.16, bounds=(0.1, 10.0)),
                            "k0": new_distribution("Beta", "k0", a=1.25, b=1.25, bounds=(10.0, 70.0)),
                            "q0": new_distribution("BetaScaled", "q0", a=2.0, b=5.5, scale=self._data._QmeanModel, bounds=(0.2, 5.0))}
        else:
            if "h0" in self._parameters["priors"]:
                distribution_parameters = self._parameters["priors"]["h0"]
                distribution_id = distribution_parameters["distribution"]
                distribution_kwargs = {key:distribution_parameters[key] for key in distribution_parameters if key != "distribution"}
                self._priors["h0"] = new_distribution(distribution_id, "h0", **distribution_kwargs)
            else:
                self._priors["h0"] = new_distribution("Beta", "h0", a=1.01, b=1.16, bounds=(0.1, 10.0))
            if "k0" in parameters["priors"]:
                distribution_parameters = self._parameters["priors"]["k0"]
                distribution_id = distribution_parameters["distribution"]
                distribution_kwargs = {key:distribution_parameters[key] for key in distribution_parameters if key != "distribution"}
                self._priors["k0"] = new_distribution(distribution_id, "k0", **parameters["priors"]["k0"])
            else:
                self._priors["k0"] = new_distribution("Beta", "k0", a=1.25, b=1.25, bounds=(0.1, 10.0)),
            if "q0" in parameters["priors"]:
                distribution_parameters = self._parameters["priors"]["q0"]
                distribution_id = distribution_parameters["distribution"]
                distribution_kwargs = {key:distribution_parameters[key] for key in distribution_parameters if key != "distribution"}
                if "QmeanModel" in distribution_kwargs:
                    distribution_kwargs["QmeanModel"] = self._data._QmeanModel
                self._priors["q0"] = new_distribution(distribution_id, "q0", **parameters["priors"]["q0"])
            else:
                self._priors["q0"] = new_distribution("BetaScaled", "q0", a=2.0, b=5.5, scale=self._data._QmeanModel, bounds=(0.2, 5.0))

    def _sample_(self, N: int, M: int, rundir: str=None):

        data = self._data
        model = self._model
        priors = self._priors

        h0 = np.zeros((N, M))
        k0 = np.zeros((N, M))
        q0 = np.zeros((N, M))
        Qlf = np.zeros((N, M, *data.H.shape))
        Qin = np.zeros((N, M, data.H.shape[0]))
        Hsp = np.zeros((N, M, *data.H.shape))
        prior_pdf = np.zeros((N, M))
        cost = np.zeros((N, M))
        
        for ij in tqdm(range(0, N*M)):
            i = ij//M
            j = ij%M
            h0[i, j] = priors["h0"].bounds[0] + i / float(N-1) * (priors["h0"].bounds[1] - priors["h0"].bounds[0])
            k0[i, j] = priors["k0"].bounds[0] + j / float(M-1) * (priors["k0"].bounds[1] - priors["k0"].bounds[0])
            model.set_h0(h0[i, j])
            model.set_k0(k0[i, j])
    
            Qlf[i, j, :, :] = model.compute_lowfroude_discharge()

            # Check that Qlf are not all nan for some profiles
            if np.any(np.all(np.isnan(Qlf[i, j, :, :]), axis=1)):
                if rundir is not None:
                    for it in range(0, Qlf.shape[2]):
                        if np.all(np.isnan(Qlf[i, j, it, :])):
                            print("all nan Qlf for profile it=%i" % it)
                            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
                            ax1.plot(data.H[it, :], "-.")
                            ax1.set_ylabel("H")
                            ax2.plot(data.Wr[it, :], "-.")
                            ax2.set_ylabel("Wr")
                            ax3.plot(data.S[it, :], "-.")
                            ax3.set_ylabel("S")
                            plt.savefig(os.path.join(rundir, "profile%04i.png") % it)
                            plt.close(fig)

                self._logger.error("All nan Qlf detected for some profiles")
                return None, error_code_from_string("all_nan_Qlf_for_some_profiles")

            Qin[i, j, :] = np.nanmean(Qlf[i, j, :, :], axis=1)
            # if np.any(np.isnan(Qin[i, j, :])):
            #     for it in range(0, data.H.shape[0]):
            #         print("%03i %.2f %.2f %12.5e" % (it, np.nanmean(data.H[it, :]), np.nanmean(data.W[it, :]), np.nanmean(data.S[it, :])))
            # print(Qin[i, j, :])
            # plt.plot(Qin[i, j, :])
            # plt.axhline(priors["q0"]._scale, c="r", ls="--")
            # plt.show()
            q0[i, j] = np.mean(Qin[i, j, :])

            prior_pdf[i, j] = priors["h0"].pdf(h0[i,j]) * priors["k0"].pdf(k0[i, j]) * priors["q0"].pdf(q0[i, j])

            cost[i, j] = model.cost(Qin[i, j, :], data)

        trace = {"h0": h0,
                 "k0": k0,
                 "q0": q0,
                 "Qlf": Qlf,
                 "Qin": Qin,
                 "Hsp": Hsp,
                 "prior_pdf": prior_pdf,
                 "cost": cost}
                 
        return trace, 0

    def _write_trace_(self, fname, trace):

        trace_variables = {"h0": (["N", "M"], trace["h0"]),
                           "k0": (["N", "M"], trace["h0"]), 
                           "q0": (["N", "M"], trace["h0"]), 
                           "Qlf": (["N", "M", "nt", "nx"], trace["Qlf"]), 
                           "Qin": (["N", "M", "nt"], trace["Qin"]), 
                           "prior_df": (["N", "M"], trace["cost"]), 
                           "cost": (["N", "M"], trace["cost"])} 
        trace_coords = {"N": np.arange(0, trace["h0"].shape[0]),
                        "M": np.arange(0, trace["h0"].shape[1]),
                        "nt": np.arange(0, self._data.H.shape[0]),
                        "nx": np.arange(0, self._data.H.shape[1])}
                           
        results_dataset = xr.Dataset(data_vars=trace_variables, coords=trace_coords)
        results_dataset.to_netcdf(fname)

    def _plot_distributions_(self, trace, plot_file: str=None):
        dh = trace["h0"][1, 0] - trace["h0"][0, 0]
        dk = trace["k0"][0, 1] - trace["k0"][0, 0]
        lh = np.exp(-trace["cost"] / np.min(np.ravel(trace["cost"])))
        extent = [trace["h0"][0, 0] - 0.5 * dh, trace["h0"][-1, 0] + 0.5 * dh,
                  trace["k0"][0, 0] - 0.5 * dk, trace["k0"][0, -1] + 0.5 * dk]
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
        im1 = ax1.imshow(trace["prior_pdf"].T, extent=extent, aspect="auto")
        ax1.set_xlabel("h0")
        ax1.set_ylabel("k0")
        plt.colorbar(im1, ax=ax1, location="bottom")
        ax1.set_title("Prior")
        im2 = ax2.imshow(lh.T, extent=extent, aspect="auto")
        ax2.set_xlabel("h0")
        plt.colorbar(im2, ax=ax2, location="bottom")
        ax2.set_title("Likelihood")
        im3 = ax3.imshow(lh.T * trace["prior_pdf"].T, extent=extent, aspect="auto")
        ax3.set_xlabel("h0")
        plt.colorbar(im3, ax=ax3, location="bottom")
        ax3.set_title("Posterior")
        plt.tight_layout()
        # im4 = ax4.imshow(np.log(trace["cost"]).T, extent=extent, aspect="auto")
        # ax4.set_xlabel("h0")
        # plt.colorbar(im4, ax=ax4, location="bottom")
        if plot_file is not None:
            plt.savefig(plot_file)
        else:
            plt.show()
        plt.close(fig)

    def _plot_discharge_comparison_(self, qm_prior, qm_post, plot_file:str=None, trace=None, qm_post_ci=None):
        data = self._data
        fig = plt.figure()
        if qm_post_ci is not None:
            # print("size:dates=%i, qm_post_ci=%i" % (data.dates.size, qm_post_ci.shape[1]))
            plt.fill_between(data.dates, qm_post_ci[0, :], qm_post_ci[1, :], color="gray", alpha=0.5)
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
        if plot_file is not None:
            plt.savefig(plot_file)
        else:
            plt.show()
        plt.close(fig)

    # def _plot_validation_(self, qm_prior, qm_post, plot_file:str=None, qm_post_ci=None):
    #     data = self._data
    #     print("data is", data)
    #     print("dir(data)=", dir(data))
    #     if hasattr(data, "_insitu_data"):

    #         fig = plt.figure()
    #         if qm_post_ci is not None:
    #             plt.fill_between(data.dates, qm_post_ci[0, :], qm_post_ci[1, :], color="gray", alpha=0.5)
    #         for key in data._insitu_data:
    #             t_obs = data._insitu_data["key"]["t"]
    #             q_obs = data._insitu_data["key"]["Q"]
    #             plt.plot(t_obs, q_obs, 'r.', label="in-situ (%i)" % key)
    #         plt.axhline(data._QmeanModel, color='k', ls='--', label="QMeanModel")
    #         plt.plot(data.dates, qm_prior, 'b-', label="prior")
    #         plt.axhline(np.mean(qm_prior), color='b', ls='--')
    #         plt.plot(data.dates, qm_post, 'g-', label="posterior")
    #         plt.axhline(np.mean(qm_post), color='g', ls='--')
    #         plt.legend()
    #         # if title is not None:
    #         #     plt.title(title)
    #         plt.tight_layout()
    #         plt.savefig(fname)
    #         plt.close(fig)

    #     elif hasattr(data, "_Q"):

    #         nrows = int(np.ceil(data.nx//4))
    #         ncols = min(data.nx, 4)
    #         fig = plt.figure(figsize=(nrows*4, ncols*4))
    #         gs = fig.add_gridspec(nrows, ncols)
    #         for r in range(0, data.nx):
    #             row = r//4
    #             col = r%4

    #             # Compute metrics
    #             nrmse = compute_metric(qm_post, data._Q[:, r], "nrmse")
    #             nse = compute_metric(qm_post, data._Q[:, r], "nse")
    #             ax = fig.add_subplot(gs[row, col])
    #             if qm_post_ci is not None:
    #                 ax.fill_between(data.dates, qm_post_ci[0, :], qm_post_ci[1, :], color="gray", alpha=0.5)
    #             ax.plot(data.dates, data.Q[:, r], 'r.', label="target")
    #             ax.axhline(data._QmeanModel, color='k', ls='--', label="QMeanModel")
    #             ax.plot(data.dates, qm_prior, 'b-', label="prior")
    #             ax.axhline(np.mean(qm_prior), color='b', ls='--')
    #             ax.plot(data.dates, qm_post, 'g-', label="posterior")
    #             ax.axhline(np.mean(qm_post), color='g', ls='--')
    #             if r == 0:
    #                 ax.legend()
    #             # if title is not None:
    #             #     plt.title(title)
    #             plt.tight_layout()
    #             if plot_file is not None:
    #                 plt.savefig(plot_file)
    #             else:
    #                 plt.show()
    #             plt.close(fig)

    #     else:
    #         self._logger.warning("No target data for validation plot")

