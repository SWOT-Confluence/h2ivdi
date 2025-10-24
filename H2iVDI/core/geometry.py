import logging
import numpy as np
import piecewise_regression as pwr
import scipy.optimize as spo

def curve_fit_lin2_pwr(x, y, tries=10, alpha_default=0.5):

    valid = np.logical_and(np.isfinite(x), np.isfinite(y))
    x = x[valid]
    y = y[valid]

    if x.size == 1:
        xi = np.array([x[0], x[0]+0.1, x[0]+0.2])
        yi = np.array([y[0], y[0], y[0]])
    elif x.size == 2:
        xi = np.array([x[0], 0.5*(x[0]+x[1]), x[1]])
        yi = np.array([y[0], 0.5*(y[0]+y[1]), y[1]])
    elif x.size == 3:
        xi = x.copy()
        yi = y.copy()
    else:
        try:
            pw_fit = pwr.Fit(x, y, n_breakpoints=1)
        except Exception as err:
            logger = logging.getLogger("H2iVDI")
            logger.error("curve_fit_lin2_pwr failed: %s" % repr(err))
            logger.error("- x=%s" % repr(x))
            logger.error("- y=%s" % repr(y))
            raise
        pw_results = pw_fit.get_results()
        if pw_results["converged"] == True:
            estimates = pw_results["estimates"]
            bp1 = estimates["breakpoint1"]["estimate"]
            xi = np.array([x[0], bp1, x[-1]])
            yi = pw_fit.predict(xi)

        else:
            xi = None
            yi = None

    return xi, yi

def curve_fit_lin2(x, y, tries=10, alpha_default=0.5):

    def modelfunc(x, x0, y0, k1, k2):
        condlist = [x < x0,
                    x >= x0]
        funclist = [lambda x:k1*x + y0-k1*x0, 
                    lambda x:k2*x + y0-k2*x0]
        return np.piecewise(x, condlist, funclist)

    class ModelForDefault:
        def __init__(self, x1):
            self._x1 = x1
        def __call__(self, x, y0, k1, k2): 
            condlist = [x < self._x1,
                        x >= self._x1]
            funclist = [lambda x:k1*x + y0-k1*self._x1, 
                        lambda x:k2*x + y0-k2*self._x1]
            return np.piecewise(x, condlist, funclist)

    valid = np.logical_and(np.isfinite(x), np.isfinite(y))
    x = x[valid]
    y = y[valid]

    if x.size == 1:
        xi = np.array([x[0], x[0]+0.1, x[0]+0.2])
        yi = np.array([y[0], y[0], y[0]])
        err_min = np.inf
    elif x.size == 2:
        xi = np.array([x[0], 0.5*(x[0]+x[1]), x[1]])
        yi = np.array([y[0], 0.5*(y[0]+y[1]), y[1]])
        err_min = np.inf
    elif x.size == 3:
        xi = x.copy()
        yi = y.copy()
        err_min = np.inf
    else:
        err_min = np.inf
        p_min = None
        for t in range(tries):
            a0 = 0.1 + np.random.rand() * 0.7
            prior_x0 = x[0] + a0 * (x[-1] - x[0])
            prior_y0 = y[x.size//2]
            prior_k1 = (prior_y0 - y[0]) / (prior_x0 - x[0])
            prior_k2 = (y[-1] - prior_y0) / (x[-1] - prior_x0)
            prior = [prior_x0, prior_y0, prior_k1, prior_k2]
            #print("prior=", prior)
            try:
                params, e = spo.curve_fit(modelfunc, x, y, p0=prior)
                err = np.sum(np.abs(y - modelfunc(x, *params)))
                diff = np.diff(modelfunc(x, *params))
            except:
                err = np.inf
            #print(err, e)

            if err < err_min and np.all(diff >= 0.0):
                p_min = params
                err_min = err
        if p_min is None:
            if alpha_default is None:
                raise RuntimeError("Unable to find suitable regression")
            else:

                # Use default model (xi[1] is fixed)
                xi = np.array([x[0], x[int(x.size * alpha_default)], x[-1]])
                default_model = ModelForDefault(xi[1])
                for t in range(tries):
                    a0 = 0.1 + np.random.rand() * 0.7
                    prior_y0 = y[x.size//2]
                    prior_k1 = (prior_y0 - y[0]) / (prior_x0 - x[0])
                    prior_k2 = (y[-1] - prior_y0) / (x[-1] - prior_x0)
                    prior = [prior_y0, prior_k1, prior_k2]
                    #print("prior=", prior)
                    params, e = spo.curve_fit(default_model, x, y, p0=prior)
                    err = np.sum(np.abs(y - default_model(x, *params)))
                    diff = np.diff(default_model(x, *params))
                    if err < err_min and np.all(diff >= 0.0):
                        p_min = params
                        err_min = err

                if p_min is None:

                    print("Default model has not converged.")
                    yi = np.array([y[0], y[int(x.size * alpha_default)], y[-1]])
                else:
                    yi = default_model(xi, *p_min)
                    
        else:
            xi = np.array([x[0], p_min[0], x[-1]])
            yi = modelfunc(xi, *p_min)

    return xi, yi, err_min
