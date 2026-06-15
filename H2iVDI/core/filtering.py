import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as spi
import scipy.stats as sps

def spline_reg(x, H, method=1, plot=False, ax=None):
    valid = np.logical_and(np.isfinite(x), np.isfinite(H))

    xi = x[valid]
    Hi = H[valid]

    regU = sps.linregress(xi[-5:], Hi[-5:])
    regD = sps.linregress(xi[:5], Hi[:5])
    regA = sps.linregress(xi, Hi)
    xiU = np.linspace(xi[-1], xi[-1] + 10000.0, 11)[1:]
    xiD = np.linspace(xi[0] - 10000.0, xi[0], 11, endpoint=True)[:-1]
    HiU = regA.intercept + xiU * regA.slope
    HiD = regA.intercept + xiD * regA.slope
    xspl = np.concatenate((xiD, xi, xiU))
    Hspl = np.concatenate((HiD, Hi, HiU))
    plt.plot(xiU, HiU, "r.")
    plt.plot(xiD, HiD, "g.")
    plt.plot(xi, Hi, "b+")
    
    if method == 1:
        spl = spi.make_smoothing_spline(xspl, Hspl)
        hf = spl(x[np.isfinite(x)])
        plt.plot(x[np.isfinite(x)], hf, "k--")
    else:
        spl = spi.splrep(xspl, Hspl, k=3, s=20)
        hf = spi.splev(x[np.isfinite(x)], spl)
        if plot:
            if ax:
                ax.plot(x[np.isfinite(x)], hf, "k--")
            else:
                plt.plot(x[np.isfinite(x)], hf, "k--")
    #plt.show()
    
    
    return hf


def filtering_m1(x, H, plot=True):

    x = x[np.isfinite(x)]
    H = H[np.isfinite(x)]

    # Compute regression
    valid = np.logical_and(np.isfinite(x), np.isfinite(H))
    xi = x[valid]
    Hi = H[valid]
    regA = sps.linregress(xi, Hi)
    Hr = regA.intercept + x * regA.slope
    dH = H - Hr
    sigma = np.nanstd(dH)
    valid1 = np.logical_and(np.abs(dH) < 2.0 * sigma, np.abs(dH) < 5.0)

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=False, figsize=(12, 4))
        ax1.plot(x, Hr, "k--")
        ax1.plot(x[valid1], H[valid1], "g.")
        ax1.plot(x[~valid1], H[~valid1], "r.")

    H[~valid1] = np.nan

    if plot:
        spline_ax = ax2
        spline_plot = True
    else:
        spline_ax = None
        spline_plot = False

    Hf = spline_reg(x, H, method=2, plot=spline_plot, ax=spline_ax)
    dH = H - Hf
    valid2 = np.abs(dH) < 0.5
    H[~valid2] = np.nan
    if plot:
        ax2.plot(x, H, "g.")
    

    if plot:
        plt.show()

    return H