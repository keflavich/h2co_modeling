"""
Given a grid of parameter values and observed values, determine best fit (ish)
"""
import numpy as np
import warnings

def grid_getmatch(parval, parerr, pargrid, chi2_thresh=1):
    if parerr < 0:
        raise ValueError("Error must be positive")
    if not (np.isscalar(parval) and np.isscalar(parerr)):
        raise TypeError("parval and parerr must be scalar")

    chi2 = ((pargrid-parval)/parerr)**2
    match = chi2 < chi2_thresh
    indbestfit = np.argmin(chi2)

    return match, chi2, indbestfit

def grid_fitter(parval, parerr, pargrid, parvals, chi2_thresh=1,
                max_deviation_sigma=2):
    """
    Find the closest match and all matches within 1-sigma and
    max_deviation_sigma
    """
    if not pargrid.shape == parvals.shape:
        raise TypeError("parvals must be the 'x-axis' of pargrid and must have same shape")

    match, chi2, indbestfit = grid_getmatch(parval, parerr, pargrid,
                                            chi2_thresh=chi2_thresh)

    sigma1mean = parvals[match].mean()
    sigma1err = parvals[match].std()

    if match.sum() == 0:
        match = chi2 < max_deviation_sigma**2
        sigmaMmean = parvals[match].mean()
        sigmaMerr = parvals[match].std()

    return parvals[indbestfit],sigma1mean,sigma1err,sigmaMmean,sigmaMerr

def grid_2p_fitter(par1, epar1, pargrid1, par2, epar2, pargrid2,
                   parvals, chi2_thresh=1):

    if not pargrid1.shape == pargrid2.shape == parvals.shape:
        raise TypeError("parameter grids must have same shape")

    match, indbest, chi2 = grid_2p_getmatch(par1, epar1, pargrid1, par2,
                                                epar2, pargrid2, chi2_thresh=1)

    best = parvals[indbest]
    parmean = parvals[match].mean()
    parerr = parvals[match].std()
    chi2best = chi2.flat[indbest]

    return best, parmean, parerr, chi2best

def grid_2p_getmatch(par1, epar1, pargrid1, par2, epar2, pargrid2,
                     chi2_thresh=1):
    if (epar1 < 0) or (epar2 < 0):
        raise ValueError("Error must be positive")
    if not all([np.isscalar(p) for p in par1, epar1, par2, epar2]):
        raise TypeError("parval and parerr must be scalar")
    if not pargrid1.shape == pargrid2.shape:
        raise TypeError("parameter grids must have same shape")

    chi2_1 = ((pargrid1-par1)/epar1)**2
    chi2_2 = ((pargrid2-par2)/epar2)**2
    chi2b = chi2_1+chi2_2

    match = (chi2_1 < chi2_thresh) & (chi2_2 < chi2_thresh)

    if match.sum() > 0:
        indbest = np.argmin(np.ma.masked_where(True-match, chi2b))
        chi2best = chi2b.flat[indbest]
    else:
        warnings.warn("No direct matches were found.  Returning closest match.")
        indbest = np.argmin(chi2b)
        chi2best = chi2b.flat[indbest]
        delta_chi2b = chi2b-chi2b.min()
        match = delta_chi2b < chi2_thresh

    return match,indbest,chi2b
