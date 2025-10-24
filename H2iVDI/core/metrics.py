import numpy as np
from scipy.stats import spearmanr 


def compute_metric(y, yobs, name):
    y = y[np.isfinite(yobs)]
    yobs = yobs[np.isfinite(yobs)]
    residuals = y - yobs
    if name == "rmse":
        return np.sqrt(np.sum(np.ravel(residuals)**2) / np.ravel(yobs).size)
    elif name == "nrmse":
        if np.abs(np.mean(np.ravel(yobs))) < 1e-16:
            return np.nan
        return np.sqrt(np.sum(np.ravel(residuals)**2) / np.ravel(yobs).size) / np.mean(np.ravel(yobs))
    elif name == "nse":
        return 1.0 - np.sum(np.ravel(residuals)**2) / np.sum((np.ravel(yobs) - np.mean(np.ravel(yobs)))**2)
    elif name == "r":
        return spearmanr(y, yobs).statistic
    elif name == "nbias":
        return np.mean(y - yobs) / np.mean(yobs)
    elif name == "sige":
        return np.std(y - yobs) / np.mean(yobs)
    else:
        raise ValueError("Wrong metric name: %s" % name)
