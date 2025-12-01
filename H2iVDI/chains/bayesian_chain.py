import arviz as az
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
#from tqdm import tqdm
from tqdm.autonotebook import tqdm
import xarray as xr

from H2iVDI.core import error_code_from_string, error_string_from_code
from H2iVDI.core.distributions import new_distribution, BetaScaledDist
from H2iVDI.core.likelihoods import new_likelihood
from H2iVDI.core.metrics import compute_metric
from H2iVDI.models import new_model
from .inference_chain import InferenceChain

class BayesianChain(InferenceChain):

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

        # STEP 1: Sample h0, k0 (Low-Froude Qin method)
        self._logger.info("Bayesian calibration")
        self._logger.debug("- Sample variables space")
        self._logger.debug("- Qin method: %s" % self._parameters["q0_method"])
        trace, error_code = self._sample_(*tuple(self._parameters["sample_sizes"]), rundir=rundir, qin_method=self._parameters["q0_method"])
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
            self._write_trace_(os.path.join(rundir, "mc_trace.nc"), trace)
        if np.all(np.ravel(np.isnan(trace["cost"]))):
            self._logger.error("All NaN cost detected")
        if np.any(np.ravel(np.isnan(trace["cost"]))):
            self._logger.error("NaN cost detected")

        # Compute priors and posteriors
        self._logger.debug("- Compute prior and posterior pdfs")
        C_prior = np.sum(np.ravel(trace["prior_pdf"]))
        self._logger.debug("- Initial C_prior: %s" % str(C_prior))

        correction_method = 0
        while C_prior < 1e-15 and correction_method < 2:

            correction_method += 1
            self._logger.debug("- Correction method: %i" % correction_method)

            # if self._logger._debug_level > 0:
            #     plt.figure()
            #     plt.plot(trace["priors_margin_pdf"][:, :, 0], "b-")
            #     plt.plot(trace["priors_margin_pdf"][:, :, 1], "r-")
            #     plt.plot(trace["priors_margin_pdf"][:, :, 2], "g-")
            #     # plt.show()
            #     plt.plot(trace["q0"].flatten(), "g-")
            #     plt.axhline(0.2 * self._data._QmeanModel, color="r", ls="--")
            #     plt.axhline(5.0 * self._data._QmeanModel, color="r", ls="--")
            #     plt.show()


            if correction_method == 1:
                # STEP 1bis: Sample h0, k0 (optim Low Froude with adapted prior Qmean)
                if isinstance(self._priors["q0"], BetaScaledDist):
                    self._priors["q0"]._scale = np.mean(trace["q0"].flatten())
                else:
                    raise NotImplementedError("q0 prior correction is not implemented for prior of type: %s" % str(type(priors["q0"])))
                self._logger.debug("- Sample variables space (qin_method = Low Froude)")
                trace, error_code = self._sample_(*tuple(self._parameters["sample_sizes"]), rundir=rundir, qin_method=self._parameters["q0_method"])
            else:
                # STEP 1ter: Sample h0, k0 (optim Qin method)
                self._logger.debug("- Sample variables space (qin_method = optim)")
                trace, error_code = self._sample_(*tuple(self._parameters["sample_sizes"]), rundir=rundir, qin_method=self._parameters["q0_method"])

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
                self._write_trace_(os.path.join(rundir, "mc_trace.nc"), trace)
            if np.all(np.ravel(np.isnan(trace["cost"]))):
                self._logger.error("All NaN cost detected")
            if np.any(np.ravel(np.isnan(trace["cost"]))):
                self._logger.error("NaN cost detected")

            # Compute priors and posteriors
            self._logger.debug("- Compute prior and posterior pdfs")
            C_prior = np.sum(np.ravel(trace["prior_pdf"]))
            self._logger.debug("  - Corrected C_prior: %s" % str(C_prior))


        if C_prior < 1e-15:
            self._logger.error("C_prior is zero")
            if self._logger._debug_level > 0:
                plt.figure()
                plt.plot(trace["priors_margin_pdf"][:, :, 0], "b-")
                plt.plot(trace["priors_margin_pdf"][:, :, 1], "r-")
                plt.plot(trace["priors_margin_pdf"][:, :, 2], "g-")

                plt.figure()
                plt.plot(trace["Qin"][0, 0], "b-")
                plt.plot(trace["Qin"][-1, -1], "b-")
                plt.axhline(0.2 * self._data._QmeanModel, color="r", ls="--")
                plt.axhline(5.0 * self._data._QmeanModel, color="r", ls="--")
                plt.figure()
                plt.plot(trace["q0"].flatten(), "g-")
                plt.axhline(0.2 * self._data._QmeanModel, color="r", ls="--")
                plt.axhline(5.0 * self._data._QmeanModel, color="r", ls="--")
                plt.show()
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
        
        if self._parameters["calibrate_sigma_obs"] is True:
            l = self._calibrate_sigma_obs_(trace, rundir=rundir, qin_method=self._parameters["q0_method"])
        # else:
        #     l = self._auto_sigma_obs_(trace, rundir=rundir, qin_method=self._parameters["q0_method"])
        else:
            sigma_obs = self._auto_sigma_obs_(trace, rundir=rundir, qin_method=self._parameters["q0_method"])
            l = 0

        nt, nx = trace["Hsp"].shape[2:]
        h0_prior = np.sum(np.ravel(trace["h0"]) * np.ravel(trace["prior_pdf"])) / C_prior
        k0_prior = np.sum(np.ravel(trace["k0"]) * np.ravel(trace["prior_pdf"])) / C_prior
        qm_prior = np.zeros(self._data.H.shape[0])
        # print("COST:", np.min(np.ravel(trace["cost"]))), np.nanmin(np.ravel(trace["cost"]))
        # lh = np.exp(-2**l * (trace["cost"] / np.min(np.ravel(trace["cost"])) - 1)**2)
        if l is None:
            self._likelihood.set_sigma(sigma_obs)
            lh = self._likelihood.likelihood_from_cost(nt*nx, trace["cost"])
        else:
            lh = np.exp(-2**l * self._likelihood.loglikelihood_from_cost(nt*nx, trace["cost"]))
        if np.any(np.ravel(np.isnan(lh))):
            self._logger.error("NaN Likelihood detected")
        C_post = np.sum(np.ravel(lh * trace["prior_pdf"]))
        # print("C_post=%12.5e" % C_post)
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
        self._logger.info("- Prior:")
        self._logger.info("  - h0(0): %.2f" % h0_prior)
        self._logger.info("  - k0(0): %.2f" % k0_prior)
        self._logger.info("  - q0(0): %.2f" % np.mean(qm_prior))
        self._logger.info("- Posterior:")
        self._logger.info("  - h0(*): %.2f" % h0_post)
        self._logger.info("  - k0(*): %.2f" % k0_post)
        self._logger.info("  - q0(*): %.2f" % np.mean(qm_post))
        if self._data._A is not None or self._data._Q is not None:
            self._logger.info("- Target:")
        if self._data._A is not None:
            self._logger.info("  - h0(*): %.2f" % np.mean(np.min(self._data._A, axis=0) / np.min(self._data._W, axis=0)))
        if self._data._K is not None:
            self._logger.info("  - k0(*): %.2f" % np.mean(np.mean(self._data._K, axis=0)))
        if self._data._Q is not None:
            self._logger.info("  - q0(*): %.2f" % np.mean(np.mean(self._data._Q, axis=0)))
        if self._data._A is not None:
            h0_target = np.mean(np.min(self._data._A, axis=0) / np.min(self._data._W, axis=0))
        else:
            h0_target = None
        if self._data._K is not None:
            k0_target = np.mean(np.mean(self._data._K, axis=0))
        else:
            k0_target = None

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
            self._plot_distributions_(trace, plot_file=plot_file, target=(h0_target, k0_target))
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

        # print("PARAMETERS:", self._parameters)

        # Check sample sizes
        if not "sample_sizes" in self._parameters:
            self._parameters["sample_sizes"] = [50, 50]

        # Set q0 computation method
        if not "q0_method" in self._parameters:
            self._parameters["q0_method"] = "low-froude"

        # Set likelihood
        if not "likelihood" in self._parameters:
            self._parameters["likelihood"] = {"name": "gaussian", "sigma": 1.0}
        self._likelihood = new_likelihood(self._parameters["likelihood"]["name"], sigma=float(self._parameters["likelihood"]["sigma"]))

        # Set automatic sigma obs calibration
        if not "calibrate_sigma_obs" in self._parameters:
            self._parameters["calibrate_sigma_obs"] = True


        # Create model
        if "model" in self._parameters:
            model_id = self._parameters["model"]
        else:
            model_id = "swst3lfb"
        self._model = new_model(model_id, self._data, **self._parameters)

        # Initialise prior distributions
        if not "priors" in self._parameters:
            if np.isnan(self._data._QmeanModel):
                q0_prior = new_distribution("One", "q0")
            else:
                if self._parameters["run_mode"] == "unconstrained":
                    q0_bounds = (0.2, 5.0)
                elif self._parameters["run_mode"] == "constrained":
                    q0_bounds = (0.4, 2.5)
                else:
                    raise ValueError("Wrong run mode: %s" % self._parameters["run_mode"])
                q0_prior = new_distribution("BetaScaled", "q0", a=2.0, b=5.5, scale=self._data._QmeanModel, bounds=q0_bounds)
            self._priors = {"h0": new_distribution("Beta", "h0", a=1.01, b=1.16, bounds=(0.1, 10.0)),
                            "k0": new_distribution("Beta", "k0", a=1.25, b=1.25, bounds=(10.0, 90.0)),
                            "q0": q0_prior}
        else:
            self._priors = {}
            if "h0" in self._parameters["priors"]:
                distribution_parameters = self._parameters["priors"]["h0"]
                distribution_id = distribution_parameters["distribution"]
                distribution_kwargs = {key:distribution_parameters[key] for key in distribution_parameters if key != "distribution"}
                self._priors["h0"] = new_distribution(distribution_id, "h0", **distribution_kwargs)
            else:
                self._priors["h0"] = new_distribution("Beta", "h0", a=1.01, b=1.16, bounds=(0.1, 10.0))
            if "k0" in self._parameters["priors"]:
                distribution_parameters = self._parameters["priors"]["k0"]
                distribution_id = distribution_parameters["distribution"]
                distribution_kwargs = {key:distribution_parameters[key] for key in distribution_parameters if key != "distribution"}
                self._priors["k0"] = new_distribution(distribution_id, "k0", **distribution_kwargs)
            else:
                self._priors["k0"] = new_distribution("Beta", "k0", a=1.25, b=1.25, bounds=(10.0, 90.0))
            if "q0" in self._parameters["priors"]:
                distribution_parameters = self._parameters["priors"]["q0"]
                distribution_id = distribution_parameters["distribution"]
                distribution_kwargs = {key:distribution_parameters[key] for key in distribution_parameters if key != "distribution"}
                if "QmeanModel" in distribution_kwargs:
                    distribution_kwargs["QmeanModel"] = self._data._QmeanModel
                self._priors["q0"] = new_distribution(distribution_id, "q0", **distribution_kwargs)
            else:
                if np.isnan(self._data._QmeanModel):
                    q0_prior = new_distribution("One", "q0")
                else:
                    if self._parameters["run_mode"] == "unconstrained":
                        q0_bounds = (0.2, 5.0)
                    elif self._parameters["run_mode"] == "constrained":
                        q0_bounds = (0.5, 2.0)
                        print("HERE!!!")
                    else:
                        raise ValueError("Wrong run mode: %s" % self._parameters["run_mode"])
                    q0_prior = new_distribution("BetaScaled", "q0", a=2.0, b=5.5, scale=self._data._QmeanModel, bounds=q0_bounds)
                self._priors["q0"] = q0_prior

    def _sample_(self, N: int, M: int, rundir: str=None, qin_method: str="low-froude"):

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
        if self._logger._debug_level > 1:
            priors_margin_pdf = np.zeros((N, M, 3))

        # print("priors=", priors)
        # print("priors[k0].bounds=", priors["k0"].bounds)
        # choice = input()
        
        for ij in tqdm(range(0, N*M)):
            i = ij//M
            j = ij%M
            h0[i, j] = priors["h0"].bounds[0] + i / float(N-1) * (priors["h0"].bounds[1] - priors["h0"].bounds[0])
            k0[i, j] = priors["k0"].bounds[0] + j / float(M-1) * (priors["k0"].bounds[1] - priors["k0"].bounds[0])
            model.set_h0(h0[i, j])
            model.set_k0(k0[i, j])
            # print("qin_method=", qin_method)

            if qin_method == "low-froude":
        
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
                                ax2.plot(data.W[it, :], "-.")
                                ax2.set_ylabel("Wr")
                                ax3.plot(data.S[it, :], "-.")
                                ax3.set_ylabel("S")
                                plt.savefig(os.path.join(rundir, "profile%04i.png") % it)
                                plt.close(fig)

                    self._logger.error("All nan Qlf detected for some profiles")
                    return None, error_code_from_string("all_nan_Qlf_for_some_profiles")

                Qin[i, j, :] = np.nanmean(Qlf[i, j, :, :], axis=1)

            elif qin_method == "optim":
                # print("Q0 bounds[1]:", priors["q0"].bounds)
                # choice = input()
                q0_bounds = (0.2 * self._data._QmeanModel, 5.0 * self._data._QmeanModel)
                # print("Q0 bounds[2]:", q0_bounds, self._data._QmeanModel)
                # choice = input()
                Qin[i, j, :] = model.compute_optim_qin(q0_bounds)
                # print(i, j, h0[i, j], k0[i, j], np.mean(Qin[i, j, :]))


            # if np.any(np.isnan(Qin[i, j, :])):
            #     for it in range(0, data.H.shape[0]):
            #         print("%03i %.2f %.2f %12.5e" % (it, np.nanmean(data.H[it, :]), np.nanmean(data.W[it, :]), np.nanmean(data.S[it, :])))
            # print(Qin[i, j, :])
            # plt.plot(Qin[i, j, :])
            # plt.axhline(priors["q0"]._scale, c="r", ls="--")
            # plt.show()
            q0[i, j] = np.mean(Qin[i, j, :])

            prior_pdf[i, j] = priors["h0"].pdf(h0[i,j], data) * priors["k0"].pdf(k0[i, j], data) * priors["q0"].pdf(q0[i, j], data, k0=k0[i, j])
            if self._logger._debug_level > 1:
                priors_margin_pdf[i, j, 0] = priors["h0"].pdf(h0[i,j], data)
                priors_margin_pdf[i, j, 1] = priors["k0"].pdf(k0[i, j], data)
                priors_margin_pdf[i, j, 2] = priors["q0"].pdf(q0[i, j], data, k0=k0[i, j])
                
            # print(i, j, h0[i, j], k0[i, j], q0[i, j], prior_pdf[i, j])
            pm0 = priors["h0"].pdf(h0[i,j], data)
            pm1 = priors["k0"].pdf(k0[i, j], data)
            pm2 = priors["q0"].pdf(q0[i, j], data, k0=k0[i, j])
            if np.isnan(pm0) or np.isnan(pm1) or np.isnan(pm2):
                print(priors["q0"])
                print("pms:", pm0, pm1, pm2)
                choice = input()

            cost[i, j] = model.cost(Qin[i, j, :], data)

        trace = {"h0": h0,
                 "k0": k0,
                 "q0": q0,
                 "Qlf": Qlf,
                 "Qin": Qin,
                 "Hsp": Hsp,
                 "prior_pdf": prior_pdf,
                 "cost": cost}
        if self._logger._debug_level > 1:
            trace["priors_margin_pdf"] = priors_margin_pdf
                 
        return trace, 0

    def _calibrate_sigma_obs_(self, trace: dict, rundir: str=None, qin_method: str="low-froude"):

        data = self._data
        model = self._model
        priors = self._priors
        nt, nx = data.H.shape

        C_prior = np.sum(np.ravel(trace["prior_pdf"]))
        h0_prior = np.sum(np.ravel(trace["h0"]) * np.ravel(trace["prior_pdf"])) / C_prior
        k0_prior = np.sum(np.ravel(trace["k0"]) * np.ravel(trace["prior_pdf"])) / C_prior
        qm_prior = np.zeros(self._data.H.shape[0])
        for it in range(0, self._data.H.shape[0]):
            qm_prior[it] = np.sum(np.ravel(trace["Qin"][:, :, it]) * np.ravel(trace["prior_pdf"]))
        qm_prior /= C_prior

        cost_min = np.min(np.ravel(trace["cost"]))

        ll = []
        lx = []
        ly = []
        for l in range(-15, 15):

            # lh = np.exp(-2**l * (trace["cost"] / cost_min - 1)**2)
            lh = np.exp(-2**l * self._likelihood.loglikelihood_from_cost(nt*nx, trace["cost"]))
            # print("lh=", lh)
            if np.any(np.ravel(np.isnan(lh))):
                self._logger.error("NaN Likelihood detected")
            C_post = np.sum(np.ravel(lh * trace["prior_pdf"]))
            if C_post < 1e-36:
                print(l, "STOP:C_post=%12.5e" % C_post)
                break
            h0_post = np.sum(np.ravel(trace["h0"]) * np.ravel(lh * trace["prior_pdf"])) / C_post
            k0_post = np.sum(np.ravel(trace["k0"]) * np.ravel(lh * trace["prior_pdf"])) / C_post
            qm_post = np.zeros(self._data.H.shape[0])
            varq = np.zeros(self._data.H.shape[0])
            for it in range(0, self._data.H.shape[0]):
                qm_post[it] = np.sum(np.ravel(trace["Qin"][:, :, it]) * np.ravel(lh * trace["prior_pdf"]))
                varq[it] = np.sum(np.ravel(((trace["Qin"][:, :, it]) - qm_post[it])**2) * np.ravel(lh * trace["prior_pdf"]))
                #qm[it] = np.sum(100.0 * np.ravel(tracek["prior_pdf"]))
            qm_post /= C_post
            varq /= C_post

            model.set_h0(h0_post)
            model.set_k0(k0_post)

            if qin_method == "low-froude":
        
                Qlf = model.compute_lowfroude_discharge()

                # Check that Qlf are not all nan for some profiles
                if np.any(np.all(np.isnan(Qlf), axis=1)):
                    if rundir is not None:
                        for it in range(0, Qlf.shape[0]):
                            if np.all(np.isnan(Qlf[it, :])):
                                print("all nan Qlf for profile it=%i" % it)
                                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
                                ax1.plot(data.H[it, :], "-.")
                                ax1.set_ylabel("H")
                                ax2.plot(data.W[it, :], "-.")
                                ax2.set_ylabel("Wr")
                                ax3.plot(data.S[it, :], "-.")
                                ax3.set_ylabel("S")
                                plt.savefig(os.path.join(rundir, "profile%04i.png") % it)
                                plt.close(fig)

                    self._logger.error("All nan Qlf detected for some profiles")
                    return None, error_code_from_string("all_nan_Qlf_for_some_profiles")

                qin = np.nanmean(Qlf, axis=1)

            elif qin_method == "optim":

                qin = model.compute_optim_qin(priors["q0"].bounds)

            cost = model.cost(qin, data)
            ll.append(l)
            lx.append((cost / cost_min - 1)**2)
            ly.append(np.sum((qm_post - qm_prior)**2 / np.maximum(1e-8, varq)))

            nrmse = compute_metric(qm_post, data._Q[:, 0], "nrmse")
            # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
            # ax1.plot(qm_prior, 'k.')
            # ax1.plot(qm_post, 'b-')
            # ax2.plot((qm_post - qm_prior)**2 / np.maximum(1e-8, varq))
            # ax3.plot(data._Q[:, 0], 'r.')
            # ax3.plot(qm_post, 'b-')
            # ax4.plot(trace["cost"], lh, '.')
            # fig.suptitle("nrmse=%f" % nrmse)
            # plt.show()
            # plt.close(fig)
            print(l, lx[-1], ly[-1], nrmse)

        if self._logger._debug_level > 0:
            plt.plot(lx, ly, "b-")
            plt.plot(lx, ly, "b+")
            for i, l in enumerate(ll):
                if l%5 == 0:
                    plt.text(lx[i], ly[i], "%i" % l)
            plt.show()
        l = None
        while l is None:
            try:
                l = int(input("l="))
            except:
                print("Wrong value !")
                l = None

        return l

    # def _auto_sigma_obs_(self, trace: dict, rundir: str=None, qin_method: str="low-froude"):

    #     data = self._data
    #     model = self._model
    #     priors = self._priors
    #     nt, nx = data.H.shape

    #     l = 0
    #     # print(np, type(np))
    #     llh = self._likelihood.loglikelihood_from_cost(nt*nx, trace["cost"])
    #     # print(llh, type(llh), l)
    #     lh = np.exp(-2**l * self._likelihood.loglikelihood_from_cost(nt*nx, trace["cost"]))
    #     C_post = np.sum(np.ravel(lh * trace["prior_pdf"]))
    #     # print("l=%i, C_post=%12.5e" % (l, C_post))
    #     while C_post < 1e-15:
    #         l = l - 1
    #         llh = self._likelihood.loglikelihood_from_cost(nt*nx, trace["cost"])
    #         # print(llh, type(llh), l)
    #         lh = np.exp(-2**l * self._likelihood.loglikelihood_from_cost(nt*nx, trace["cost"]))
    #         C_post = np.sum(np.ravel(lh * trace["prior_pdf"]))
    #         # print("l=%i, C_post=%12.5e" % (l, C_post))
    #     # print("auto_sigma_obs: l=%i" % l)
    #     return l

    def _auto_sigma_obs_(self, trace: dict, rundir: str=None, qin_method: str="low-froude"):

        data = self._data
        model = self._model
        priors = self._priors
        nt, nx = data.H.shape

        sigma_obs = 0.25
        self._likelihood.set_sigma(sigma_obs)
        # print(np, type(np))
        # llh = self._likelihood.loglikelihood_from_cost(nt*nx, trace["cost"])
        # print(llh, type(llh), l)
        # print("llh=", -self._likelihood.loglikelihood_from_cost(nt*nx, trace["cost"]))
        # lh = np.exp(-self._likelihood.loglikelihood_from_cost(nt*nx, trace["cost"]))
        lh = self._likelihood.likelihood_from_cost(nt*nx, trace["cost"])
        C_post = np.sum(np.ravel(lh * trace["prior_pdf"]))
        # print("sigma_obs=%f, C_post=%12.5e, cost=%12.5e" % (sigma_obs, C_post, np.mean(trace["cost"].flatten())))
        best_sigma_obs = sigma_obs
        best_C_post = C_post
        while C_post < 1e-15 and sigma_obs < 1.5:
            sigma_obs += 0.05
            # l = l - 1
            self._likelihood.set_sigma(sigma_obs)
            # llh = self._likelihood.loglikelihood_from_cost(nt*nx, trace["cost"])
            # print(llh, type(llh), l)
            # print("llh=", -self._likelihood.loglikelihood_from_cost(nt*nx, trace["cost"]))
            # lh = np.exp(-self._likelihood.loglikelihood_from_cost(nt*nx, trace["cost"]))
            lh = self._likelihood.likelihood_from_cost(nt*nx, trace["cost"])
            # plt.imshow(lh)
            # plt.show()
            C_post = np.sum(np.ravel(lh * trace["prior_pdf"]))
            if C_post > best_C_post:
                best_sigma_obs = sigma_obs
                best_C_post = C_post
            # print("sigma_obs=%f, C_post=%12.5e" % (sigma_obs, C_post))
        
        # print("auto_sigma_obs: l=%i" % l)
        self._logger.debug("C_post=%s (sigma_obs=%.3f)" % (str(best_C_post), best_sigma_obs))
        return best_sigma_obs

    def _write_trace_(self, fname, trace):

        trace_variables = {"h0": (["N", "M"], trace["h0"]),
                           "k0": (["N", "M"], trace["k0"]), 
                           "q0": (["N", "M"], trace["q0"]), 
                           "Qlf": (["N", "M", "nt", "nx"], trace["Qlf"]), 
                           "Qin": (["N", "M", "nt"], trace["Qin"]), 
                           "prior_df": (["N", "M"], trace["prior_pdf"]), 
                           "cost": (["N", "M"], trace["cost"])} 
        trace_coords = {"N": np.arange(0, trace["h0"].shape[0]),
                        "M": np.arange(0, trace["h0"].shape[1]),
                        "nt": np.arange(0, self._data.H.shape[0]),
                        "nx": np.arange(0, self._data.H.shape[1])}
                           
        results_dataset = xr.Dataset(data_vars=trace_variables, coords=trace_coords)
        results_dataset.to_netcdf(fname)

    def _plot_distributions_(self, trace, plot_file: str=None, target: tuple=(None, None)):
        nt, nx = trace["Hsp"].shape[2:]
        dh = trace["h0"][1, 0] - trace["h0"][0, 0]
        dk = trace["k0"][0, 1] - trace["k0"][0, 0]
        # lh = np.exp(-trace["cost"] / np.min(np.ravel(trace["cost"])))
        lh = trace["lh"]
        extent = [trace["h0"][0, 0] - 0.5 * dh, trace["h0"][-1, 0] + 0.5 * dh,
                  trace["k0"][0, 0] - 0.5 * dk, trace["k0"][0, -1] + 0.5 * dk]
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
        im1 = ax1.imshow(trace["prior_pdf"].T, extent=extent, aspect="auto")
        if target[0] is not None and target[1] is not None:
            ax1.plot(target[0], target[1], "r+")
        elif target[0] is not None:
            ax1.axvline(target[0], color="r", ls="--")
        ax1.set_xlabel("h0")
        ax1.set_ylabel("k0")
        plt.colorbar(im1, ax=ax1, location="bottom")
        ax1.set_title("Prior")
        im2 = ax2.imshow(lh.T, extent=extent, aspect="auto")
        if target[0] is not None and target[1] is not None:
            ax2.plot(target[0], target[1], "r+")
        elif target[0] is not None:
            ax2.axvline(target[0], color="r", ls="--")
        ax2.set_xlabel("h0")
        plt.colorbar(im2, ax=ax2, location="bottom")
        ax2.set_title("Likelihood")
        im3 = ax3.imshow(lh.T * trace["prior_pdf"].T, extent=extent, aspect="auto")
        if target[0] is not None and target[1] is not None:
            ax3.plot(target[0], target[1], "r+")
        elif target[0] is not None:
            ax3.axvline(target[0], color="r", ls="--")
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
            # print(data._valid, type(data._valid))
            plt.fill_between(data.dates[data._valid], qm_post_ci[0, :], qm_post_ci[1, :], color="gray", alpha=0.5)
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
        plt.plot(data.dates[data._valid], qm_prior, 'b-', label="prior")
        plt.axhline(np.mean(qm_prior), color='b', ls='--')
        plt.plot(data.dates[data._valid], qm_post, 'g-', label="posterior")
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

