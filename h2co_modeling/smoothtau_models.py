"""
Smooth the LVG models with distributions to get tau, then fit it with
optimization procedures.
"""

import numpy as np
import hopkins_pdf
import turbulent_pdfs
from turbulent_pdfs import lognormal_massweighted

try:
    from agpy import readcol
    tablereader = lambda x: readcol(x, asRecArray=True)
except ImportError:
    import astropy.table
    tablereader = lambda x: astropy.table.read(x, format='ascii')

import os
path = __file__
pwd = os.path.split(path)[0]

# plotting stuff
import pylab as pl
pl.rc('font',size=20)



class SmoothtauModels(object):

    def __init__(self, datafile=os.path.join(pwd,'radex_data/1-1_2-2_XH2CO=1e-9_troscompt.dat')):
        self.datafile = datafile
        self._datacache = {}
        self.radtab = tablereader(self.datafile)

    def select_data(self, abundance=-8.5, opr=1, temperature=20, tolerance=0.1):
        """
        Load the data table and select only a subset of the density and column
        data.  The tolerance is required because the abundance is not exact; it
        must be re-computed from column / density*length and includes some rounding
        errors.

        Parameters
        ----------
        abundance: float
            Logarithmic abundance of ortho-Formaldehyde with respect to H2 per velocity
            gradient
        opr: float or None
            Ortho/para ratio of molecular hydrogen.  Must be in the range 1e-3 to 3
            (0 is physically acceptable, but not supported by the RADEX code unless you change
            the format of the input file)
        temperature: float
            Kinetic temperature of the gas
        tolerance: float
            A parameter governing the range around the abundance to accept.  0.1 is usually fine.
        datafile: string
            The full path to a data file
        """
        key = (abundance, opr, temperature, tolerance)
        if key in self._datacache:
            return self._datacache[key]
        else:
            radtab = self.radtab # shorthand
            #tolerance = {-10:0.1, -9.5: 0.3, -9: 0.1, -8:0.1, -8.5: 0.3}[abundance]
            OKtem = radtab['Temperature'] == temperature
            if opr is not None:
                OKopr = radtab['opr'] == opr
            else:
                OKopr = True
            OKabund = np.abs((radtab['log10col'] - radtab['log10dens'] - np.log10(3.08e18)) - abundance) < tolerance
            OK = OKtem * OKopr * OKabund

            tex1x = radtab['Tex_low'][OK]
            tex2x = radtab['Tex_hi'][OK]
            trot1x = radtab['TrotLow'][OK]
            trot2x = radtab['TrotUpp'][OK]
            tau1x = radtab['TauLow'][OK]
            tau2x = radtab['TauUpp'][OK]
            dens = radtab['log10dens'][OK]
            col = radtab['log10col'][OK]

            self._datacache[key] = trot1x,trot2x,tex1x,tex2x,tau1x,tau2x,dens,col

            return trot1x,trot2x,tex1x,tex2x,tau1x,tau2x,dens,col


    def generate_tau_functions(self, **kwargs):
        """
        Generate functions to compute the optical depth as a function of density
        given different distribution shapes.
        """
        trot1x,trot2x,tex1x,tex2x,tau1x,tau2x,dens,col = self.select_data(**kwargs)

        # ddens = (10**dens[1]-10**dens[0])
        #dlogdens = (dens[1]-dens[0])
        #dlndens = dlogdens * np.log(10)

        def tau(meandens, line=tau1x, sigma=1.0, hightail=False,
                hopkins=False, powertail=False, lowtail=False,
                compressive=False, divide_by_col=False, **kwargs):
            if compressive:
                distr = turbulent_pdfs.compressive_distr(meandens,sigma,**kwargs)
            elif lowtail:
                distr = turbulent_pdfs.lowtail_distr(meandens,sigma,**kwargs)
            elif powertail or hightail:
                distr = turbulent_pdfs.hightail_distr(meandens,sigma,**kwargs)
            elif hopkins:
                T = hopkins_pdf.T_of_sigma(sigma, logform=True)
                #distr = 10**dens * hopkins_pdf.hopkins(10**(dens), meanrho=10**(meandens), sigma=sigma, T=T) # T~0.05 M_C
                distr = hopkins_pdf.hopkins_masspdf_ofmeandens(10**dens, 10**meandens, sigma_volume=sigma, T=T, normalize=True)
                # Hopkins is integral-normalized, not sum-normalized
                #distr /= distr.sum()
            else:
                #distr = lognormal(10**dens, 10**meandens, sigma) * dlndens
                distr = lognormal_massweighted(10**dens, 10**meandens, sigma, normalize=True)
            if divide_by_col:
                return (distr*line/(10**col)).sum()
            else:
                return (distr*line).sum()

        def vtau(meandens,**kwargs):
            """ vectorized tau """
            if hasattr(meandens,'size') and meandens.size == 1:
                return tau(meandens, **kwargs)
            taumean = np.array([tau(x,**kwargs) for x in meandens])
            return taumean

        def vtau_ratio(meandens, line1=tau1x, line2=tau2x, **kwargs):
            t1 = vtau(meandens, line=line1, **kwargs)
            t2 = vtau(meandens, line=line2, **kwargs)
            return t1/t2

        return tau,vtau,vtau_ratio

    def generate_simpletools(self,**kwargs):
        tau,vtau,vtau_ratio = self.generate_tau_functions(**kwargs)

        def tauratio(meandens, sigma, **kwargs):
            return vtau_ratio(np.log10(meandens), sigma=sigma, **kwargs)

        def tauratio_hopkins(meandens, sigma, **kwargs):
            return vtau_ratio(np.log10(meandens), sigma=sigma, hopkins=True, **kwargs)

        def tau(meandens, sigma, line, **kwargs):
            return vtau(np.log10(meandens), sigma=sigma, line=line, **kwargs)

        def tau_hopkins(meandens, sigma, line, **kwargs):
            return vtau(np.log10(meandens), sigma=sigma, hopkins=True, line=line, **kwargs)

        return tauratio,tauratio_hopkins,tau,tau_hopkins


    def plot_x_vs_y(self, x='dens', y='tauratio', axis=None, abundance=-8.5,
                    sigma=1.0, temperature=20, opr=1, **kwargs):
        trot1x,trot2x,tex1x,tex2x,tau1x,tau2x,dens,col = self.select_data(abundance=abundance,temperature=temperature,opr=opr)
        tau,vtau,vtau_ratio = self.generate_tau_functions(abundance=abundance,temperature=temperature,opr=opr)

        if axis is None:
            axis = pl.gca()

        if sigma == 0:
            tau11 = tau1 = tauA = tau1x
            tau22 = tau2 = tauB = tau2x
            tauratio = tau11/tau22
        else:
            tauratio = vtau_ratio(dens, line1=tau1x, line2=tau2x, sigma=sigma)
            tau11 = tau1 = tauA = vtau(dens, line=tau1x, sigma=sigma)
            tau22 = tau2 = tauB = vtau(dens, line=tau2x, sigma=sigma)
        
        # couldn't think of anything better, I guess?
        oitaruat = 1/tauratio

        xvals = eval(x)
        yvals = eval(y)

        inds = np.argsort(xvals)

        return axis.plot(xvals[inds], yvals[inds], **kwargs)

    def generate_trot_functions(self, **kwargs):
        """
        Generate functions to compute the optical depth as a function of density
        given different distribution shapes.
        """
        trot1x,trot2x,tex1x,tex2x,tau1x,tau2x,dens,col = self.select_data(**kwargs)

        # ddens = (10**dens[1]-10**dens[0])
        #dlogdens = (dens[1]-dens[0])
        #dlndens = dlogdens * np.log(10)

        def trot(meandens, line=trot1x, sigma=1.0, hightail=False,
                 hopkins=False, powertail=False, lowtail=False,
                 compressive=False, divide_by_col=False, **kwargs):
            if compressive:
                distr = turbulent_pdfs.compressive_distr(meandens,sigma,**kwargs)
            elif lowtail:
                distr = turbulent_pdfs.lowtail_distr(meandens,sigma,**kwargs)
            elif powertail or hightail:
                distr = turbulent_pdfs.hightail_distr(meandens,sigma,**kwargs)
            elif hopkins:
                T = hopkins_pdf.T_of_sigma(sigma, logform=True)
                #distr = 10**dens * hopkins_pdf.hopkins(10**(dens), meanrho=10**(meandens), sigma=sigma, T=T) # T~0.05 M_C
                distr = hopkins_pdf.hopkins_masspdf_ofmeandens(10**dens, 10**meandens, sigma_volume=sigma, T=T, normalize=True)
                # Hopkins is integral-normalized, not sum-normalized
                #distr /= distr.sum()
            else:
                #distr = lognormal(10**dens, 10**meandens, sigma) * dlndens
                distr = lognormal_massweighted(10**dens, 10**meandens, sigma, normalize=True)
            if divide_by_col:
                return (distr*line/(10**col)).sum()
            else:
                return (distr*line).sum()

        def vtrot(meandens,**kwargs):
            """ vectorized trot """
            if hasattr(meandens,'size') and meandens.size == 1:
                return trot(meandens, **kwargs)
            trotmean = np.array([trot(x,**kwargs) for x in meandens])
            return trotmean

        def vtrot_ratio(meandens, line1=trot1x, line2=trot2x, **kwargs):
            t1 = vtrot(meandens, line=line1, **kwargs)
            t2 = vtrot(meandens, line=line2, **kwargs)
            return t1/t2

        return trot,vtrot,vtrot_ratio

    def generate_simpletools_trot(self,**kwargs):
        trot,vtrot,vtrot_ratio = self.generate_trot_functions(**kwargs)

        def trotratio(meandens, sigma, **kwargs):
            return vtrot_ratio(np.log10(meandens), sigma=sigma, **kwargs)

        def trotratio_hopkins(meandens, sigma, **kwargs):
            return vtrot_ratio(np.log10(meandens), sigma=sigma, hopkins=True, **kwargs)

        def trot(meandens, sigma, line, **kwargs):
            return vtrot(np.log10(meandens), sigma=sigma, line=line, **kwargs)

        def trot_hopkins(meandens, sigma, line, **kwargs):
            return vtrot(np.log10(meandens), sigma=sigma, hopkins=True, line=line, **kwargs)

        return trotratio,trotratio_hopkins,trot,trot_hopkins
