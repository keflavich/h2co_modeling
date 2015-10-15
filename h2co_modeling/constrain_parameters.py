"""
Functions for fitting temperature (and density and column) from the line ratio
plus whatever other constraints are available
"""
import inspect
import time
import os

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from astropy import units as u
from astropy import log
from scipy import stats
import pylab as pl
from astropy.io import fits
from astropy.utils.console import ProgressBar

from h2co_modeling import grid_fitter

short_mapping = {'dens': 'density',
                 'col': 'column',
                 'tem': 'temperature'}

def gpath(fn, gridpath='/Users/adam/work/h2co/radex/thermom/'):
    return os.path.join(gridpath, fn)

class paraH2COmodel(object):

    def __init__(self, tbackground=2.73, gridsize=[250.,101.,100.], gpath=gpath):
        """
        Initialize the model fitter.

        Parameters
        ----------
        tbackground : float
            The background temperature
        gridsize : tuple or list
            A 3-tuple containing the lengths of each dimension to interpolate
            onto.  If your underlying grid is 20x10x5, you might use a 40x20x10
            grid for higher precision using interpolation.
        gpath : function
            The path to the model files.  You need to construct these as FITS
            cubes with an appropriate header.
        """
        t0 = time.time()

        for line in ProgressBar(['303','321','322','404','422','423']):
            if os.path.exists(gpath('fjdu_pH2CO_{0}_tex_5kms.fits'.format(line))):
                self.__dict__['texgrid{0}'.format(line)] = fits.getdata(gpath('fjdu_pH2CO_{0}_tex_5kms.fits'.format(line)))
                self.__dict__['taugrid{0}'.format(line)] = fits.getdata(gpath('fjdu_pH2CO_{0}_tau_5kms.fits'.format(line)))
            else:
                print("Did not find {0}".format(gpath('fjdu_pH2CO_{0}_tex_5kms.fits'.format(line))))


        #self.texgrid303 = texgrid303 = fits.getdata(gpath('fjdu_pH2CO_303_tex_5kms.fits'))
        #self.taugrid303 = taugrid303 = fits.getdata(gpath('fjdu_pH2CO_303_tau_5kms.fits'))
        #self.texgrid321 = texgrid321 = fits.getdata(gpath('fjdu_pH2CO_321_tex_5kms.fits'))
        #self.taugrid321 = taugrid321 = fits.getdata(gpath('fjdu_pH2CO_321_tau_5kms.fits'))
        #self.texgrid322 = texgrid322 = fits.getdata(gpath('fjdu_pH2CO_322_tex_5kms.fits'))
        #self.taugrid322 = taugrid322 = fits.getdata(gpath('fjdu_pH2CO_322_tau_5kms.fits'))
        #self.texgrid404 = texgrid404 = fits.getdata(gpath('fjdu_pH2CO_404_tex_5kms.fits'))
        #self.taugrid404 = taugrid404 = fits.getdata(gpath('fjdu_pH2CO_404_tau_5kms.fits'))
        #self.texgrid422 = texgrid422 = fits.getdata(gpath('fjdu_pH2CO_422_tex_5kms.fits'))
        #self.taugrid422 = taugrid422 = fits.getdata(gpath('fjdu_pH2CO_422_tau_5kms.fits'))
        #self.texgrid423 = texgrid423 = fits.getdata(gpath('fjdu_pH2CO_423_tex_5kms.fits'))
        #self.taugrid423 = taugrid423 = fits.getdata(gpath('fjdu_pH2CO_423_tau_5kms.fits'))
        self.hdr = fits.getheader(gpath('fjdu_pH2CO_303_tex_5kms.fits'))

        t1 = time.time()
        log.debug("Loading grids took {0:0.1f} seconds".format(t1-t0))

        self.Tbackground = tbackground
        self.tline303a = ((1.0-np.exp(-np.array(self.taugrid303))) *
                          (self.texgrid303-self.Tbackground))
        self.tline321a = ((1.0-np.exp(-np.array(self.taugrid321))) *
                          (self.texgrid321-self.Tbackground))
        self.tline322a = ((1.0-np.exp(-np.array(self.taugrid322))) *
                          (self.texgrid322-self.Tbackground))
        self.tline404a = ((1.0-np.exp(-np.array(self.taugrid404))) *
                          (self.texgrid404-self.Tbackground))
        self.tline423a = ((1.0-np.exp(-np.array(self.taugrid423))) *
                          (self.texgrid423-self.Tbackground))
        self.tline422a = ((1.0-np.exp(-np.array(self.taugrid422))) *
                          (self.texgrid422-self.Tbackground))

        zinds,yinds,xinds = np.indices(self.tline303a.shape)
        upsample_factor = np.array([gridsize[0]/self.tline303a.shape[0], # temperature
                                    gridsize[1]/self.tline303a.shape[1], # density
                                    gridsize[2]/self.tline303a.shape[2]], # column
                                   dtype='float')
        uzinds,uyinds,uxinds = upsinds = np.indices([x*us
                                                     for x,us in zip(self.tline303a.shape,
                                                                     upsample_factor)],
                                                   dtype='float')
        self.tline303 = map_coordinates(self.tline303a,
                                   upsinds/upsample_factor[:,None,None,None],
                                   mode='nearest')
        self.tline321 = map_coordinates(self.tline321a,
                                   upsinds/upsample_factor[:,None,None,None],
                                   mode='nearest')
        self.tline322 = map_coordinates(self.tline322a,
                                   upsinds/upsample_factor[:,None,None,None],
                                   mode='nearest')
        self.tline404 = map_coordinates(self.tline404a,
                                   upsinds/upsample_factor[:,None,None,None],
                                   mode='nearest')
        self.tline422 = map_coordinates(self.tline422a,
                                   upsinds/upsample_factor[:,None,None,None],
                                   mode='nearest')
        self.tline423 = map_coordinates(self.tline423a,
                                   upsinds/upsample_factor[:,None,None,None],
                                   mode='nearest')

    
        self.tline = {303: self.tline303,
                      321: self.tline321,
                      322: self.tline322,
                      422: self.tline422,
                      423: self.tline423,
                      404: self.tline404,
                     }

        assert self.hdr['CTYPE2'].strip() == 'LOG-DENS'
        assert self.hdr['CTYPE1'].strip() == 'LOG-COLU'

        self.columnarr = ((uxinds + self.hdr['CRPIX1']-1)*self.hdr['CDELT1'] /
                      float(upsample_factor[2])+self.hdr['CRVAL1']) # log column
        self.densityarr  = ((uyinds + self.hdr['CRPIX2']-1)*self.hdr['CDELT2'] /
                      float(upsample_factor[1])+self.hdr['CRVAL2']) # log density
        self.temparr    = ((uzinds + self.hdr['CRPIX3']-1)*self.hdr['CDELT3'] /
                      float(upsample_factor[0])+self.hdr['CRVAL3']) # lin temperature
        self.drange = [self.densityarr.min(), self.densityarr.max()]
        self.crange = [self.columnarr.min(),  self.columnarr.max()]
        self.trange = [self.temparr.min(),    self.temparr.max()]
        self.darr = self.densityarr[0,:,0]
        self.carr = self.columnarr[0,0,:]
        self.tarr = self.temparr[:,0,0]
        self.axes = {'dens': self.darr,
                     'col': self.carr,
                     'tem': self.tarr}
        self.labels = {'dens': 'Density $n(\mathrm{H}_2)$ [log cm$^{-3}$]',
                       'col': 'p-H$_2$CO [log cm$^{-2}$/(km s$^{-1}$ pc)]',
                       'tem': 'Temperature (K)'}

        # While the individual lines are subject to filling factor uncertainties, the
        # ratio is not.
        self.modelratio1 = self.tline321/self.tline303
        self.modelratio2 = self.tline322/self.tline321
        self.modelratio_423_404 = self.tline423/self.tline404
        self.modelratio_422_404 = self.tline422/self.tline404
        self.modelratio_404_303 = self.tline404/self.tline303

        self.model_logabundance = np.log10(10**self.columnarr / u.pc.to(u.cm) /
                                           10**self.densityarr)

        t2 = time.time()
        log.debug("Grid initialization took {0:0.1f} seconds total,"
                  " {1:0.1f} since loading grids.".format(t2-t0,t2-t1))

    def grid_getmatch_321to303(self, ratio, eratio):
            match,indbest,chi2r = grid_fitter.grid_getmatch(ratio, eratio,
                                                            self.modelratio1)
            return chi2r

    def grid_getmatch_404to303(self, ratio, eratio):
            match,indbest,chi2r = grid_fitter.grid_getmatch(ratio, eratio,
                                                            self.modelratio_404_303)
            return chi2r

    def grid_getmatch_422to404(self, ratio, eratio):
            match,indbest,chi2r = grid_fitter.grid_getmatch(ratio, eratio,
                                                            self.modelratio_422_404)
            return chi2r

    def grid_getmatch_423to404(self, ratio, eratio):
            match,indbest,chi2r = grid_fitter.grid_getmatch(ratio, eratio,
                                                            self.modelratio_423_404)
            return chi2r

    def grid_getmatch_322to321(self, ratio, eratio):
            match,indbest,chi2r = grid_fitter.grid_getmatch(ratio, eratio,
                                                            self.modelratio2)
            return chi2r

    @property
    def chi2(self):
        return self._chi2

    @chi2.setter
    def chi2(self, value):
        self._chi2 = value
        self._likelihood = np.exp(-value/2)

    @property
    def likelihood(self):
        return self._likelihood

    def chi2_fillingfactor(self, tline, etline, lineid):
        """
        Return a chi^2 value for each model parameter treating the specified
        line brightness as a lower limit

        Parameters
        ----------
        tline : float
            The line brightness temperature
        lineid : int
            The line id, one of 303,321,322
        """
        chi2 = ((self.tline[lineid] - tline)/etline)**2 * (self.tline[lineid] < tline)
        return chi2

    def chi2_column(self, logh2column, elogh2column, h2coabundance, linewidth):

        h2fromh2co = self.columnarr + np.log10(np.sqrt(np.pi) * linewidth) - h2coabundance
        chi2_h2 = ((h2fromh2co-logh2column)/elogh2column)**2

        return chi2_h2

    def chi2_abundance(self, logabundance, elogabundance):
        # Should there be a factor of np.log10(np.sqrt(np.pi) * linewidth) here?
        # Should linewidth be treated as FWHM instead of sigma, as it presently is?
        model_logabundance = self.columnarr - np.log10(u.pc.to(u.cm)) - self.densityarr
        chi2X = ((model_logabundance-logabundance)/elogabundance)**2
        return chi2X

    def list_parameters():
        return ['taline303',  'etaline303', 'taline321',  'etaline321',
                'taline322',  'etaline322', 'logabundance',  'elogabundance',
                'logh2column',  'elogh2column', 'ratio321303',  'eratio321303',
                'ratio321322',  'eratio321322', 'linewidth']

    def set_constraints_fromrow(self, row, **kwargs):

        mapping = {'e321':'etaline321',
                   'Smean321':'taline321',
                   'Smean303':'taline303',
                   'er321303':'eratio321303',
                   'eratio321303':'eratio321303',
                   'e303':'etaline303',
                   'r321303':'ratio321303',
                   'ratio321303':'ratio321303',
                   'r321303':'ratio321303',
                   'er321303':'eratio321303',
                   'logabundance':'logabundance',
                   'elogabundance':'elogabundance',
                   'logh2column':'logh2column',
                   'elogh2column':'elogh2column',
                   'dustmindens':'linmindens',
                   'v_rms':'linewidth',
                  }
        pars = {mapping[k]: row[k] for k in row.colnames if k in mapping}
        pars.update(**kwargs)

        self.set_constraints(**pars)

    def set_constraints(self,
                        taline303=None, etaline303=None,
                        taline321=None, etaline321=None,
                        taline322=None, etaline322=None,
                        taline404=None, etaline404=None,
                        taline422=None, etaline422=None,
                        taline423=None, etaline423=None,
                        logabundance=None, elogabundance=None,
                        logh2column=None, elogh2column=None,
                        ratio321303=None, eratio321303=None,
                        ratio321322=None, eratio321322=None,
                        ratio404303=None, eratio404303=None,
                        ratio422404=None, eratio422404=None,
                        ratio423404=None, eratio423404=None,
                        linmindens=None,
                        mindens=None, emindens=0.2,
                        linewidth=None):
        """
        Set parameter constraints from a variety of inputs.  This will fill in
        a variety of .chi2_[x] values.

        All errors are 1-sigma Gaussian errors.

        The ``taline`` parameters are only used as lower limits.  

        Logabundance and logh2column are both log_10 values, so the errorbars
        are effectively lognormal 1-sigma errors.

        The ratios are generally the most important constraints.

        A minimum volume density, with 1-sigma lognormal one-sided error
        ``emindens``, can be included.  ``mindens`` is logarithmic, but you can
        use ``linmindens`` instead.  ``linewidth`` also needs to be specified
        in km/s.
        """

        argspec=inspect.getargvalues(inspect.currentframe())
        for arg in argspec.args:
            if argspec.locals[arg] is not None:
                setattr(self, arg, argspec.locals[arg])

        self.chi2_X = (self.chi2_abundance(logabundance, elogabundance) 
                       if not any(arg is None for arg in (logabundance,
                                                          elogabundance))
                       else 0)

        self.chi2_h2 = (self.chi2_column(logh2column, elogh2column,
                                         logabundance, linewidth) 
                        if not
                        any(arg is None for arg in (logabundance, logh2column,
                                                      elogh2column, linewidth))
                        else 0)

        self.chi2_ff1 = (self.chi2_fillingfactor(taline303, etaline303, 303)
                         if not any(arg is None for arg in (taline303,
                                                            etaline303))
                         else 0)


        self.chi2_ff2 = (self.chi2_fillingfactor(taline321, etaline321, 321)
                         if not any(arg is None for arg in (taline321,
                                                            etaline321))
                         else 0)


        self.chi2_r321303 = (self.grid_getmatch_321to303(ratio321303,
                                                         eratio321303)
                             if not any(arg is None for arg in (ratio321303,
                                                                eratio321303))
                             else 0)
        if np.all(~np.isfinite(self.chi2_r321303)):
            self.chi2_r321303 = 0

        self.chi2_r423404 = (self.grid_getmatch_423to404(ratio423404,
                                                         eratio423404)
                             if not any(arg is None for arg in (ratio423404,
                                                                eratio423404))
                             else 0)
        if np.all(~np.isfinite(self.chi2_r423404)):
            self.chi2_r423404 = 0

        self.chi2_r422404 = (self.grid_getmatch_422to404(ratio422404,
                                                         eratio422404)
                             if not any(arg is None for arg in (ratio422404,
                                                                eratio422404))
                             else 0)
        if np.all(~np.isfinite(self.chi2_r422404)):
            self.chi2_r422404 = 0

        self.chi2_r404303 = (self.grid_getmatch_404to303(ratio404303,
                                                         eratio404303)
                             if not any(arg is None for arg in (ratio404303,
                                                                eratio404303))
                             else 0)
        if np.all(~np.isfinite(self.chi2_r404303)):
            self.chi2_r404303 = 0

        self.chi2_r321322 = (self.grid_getmatch_322to321(ratio321322,
                                                         eratio321322)
                             if not any(arg is None for arg in (ratio321322,
                                                                eratio321322))
                             else 0)
        if np.all(~np.isfinite(self.chi2_r321322)):
            self.chi2_r321322 = 0

        if linmindens is not None:
            if mindens is not None:
                raise ValueError("Both linmindens and logmindens were set.")
            mindens = np.log10(linmindens)

        if mindens is not None:
            self.chi2_dens = (((self.densityarr - mindens)/emindens)**2
                              * (self.densityarr < (mindens)))
        else:
            self.chi2_dens = 0

        self.compute_chi2_fromcomponents()

    def compute_chi2_fromcomponents(self):
        """
        Compute the total chi2 from the individual chi2 components
        """
        self._parconstraints = None # not determined until get_parconstraints run
        self.chi2 = (self.chi2_X + self.chi2_h2 + self.chi2_ff1 + self.chi2_ff2
                     + self.chi2_r321322 + self.chi2_r321303 + self.chi2_dens +
                     self.chi2_r404303 + self.chi2_r423404 + self.chi2_r422404)

    def get_parconstraints(self,
                           chi2level=stats.chi2.ppf(stats.norm.cdf(1)-stats.norm.cdf(-1), 3)):
        """
        If parameter constraints have been set with set_constraints or
        set_constraints_fromrow

        Parameters
        ----------
        chi2level : float
            The maximum Delta-chi^2 value to include: this will be used to
            determine the min/max 1-sigma errorbars
        """
        if not hasattr(self, 'chi2'):
            raise AttributeError("Run set_constraints first")

        row = {}

        indbest = np.argmin(self.chi2)
        deltachi2b = (self.chi2-self.chi2.min())
        for parname,pararr in zip(('temperature','column','density'),
                                  (self.temparr,self.columnarr,self.densityarr)):
            row['{0}_chi2'.format(parname)] = pararr.flat[indbest]
            row['E({0})'.format(parname)] = (pararr*self.likelihood).sum() / self.likelihood.sum()
            OK = deltachi2b<chi2level
            if np.count_nonzero(OK) > 0:
                row['{0:1.1s}min1sig_chi2'.format(parname)] = pararr[OK].min()
                row['{0:1.1s}max1sig_chi2'.format(parname)] = pararr[OK].max()
            else:
                row['{0:1.1s}min1sig_chi2'.format(parname)] = np.nan
                row['{0:1.1s}max1sig_chi2'.format(parname)] = np.nan

        for parname in ('logh2column', 'elogh2column', 'logabundance',
                        'elogabundance'):
            row[parname] = getattr(self, parname)

        self._parconstraints = row

        return row

    @property
    def parconstraints(self):
        if not hasattr(self,'_parconstraints') or self._parconstraints is None:
            return self.get_parconstraints()
        else:
            return self._parconstraints

    def parplot(self, par1='dens', par2='col', nlevs=5, levels=None):

        xax = self.axes[par1]
        yax = self.axes[par2]
        xlabel = self.labels[par1]
        ylabel = self.labels[par2]
        axis = {('col','dens'): 0,
                ('dens','tem'): 2,
                ('col','tem'): 1}[(par1,par2)]

        if levels is None:
            levels = np.arange(nlevs)

        xmaxlike = self.parconstraints['{0}_chi2'.format(short_mapping[par1])]
        ymaxlike = self.parconstraints['{0}_chi2'.format(short_mapping[par2])]
        xexpect = self.parconstraints['E({0})'.format(short_mapping[par1])]
        yexpect = self.parconstraints['E({0})'.format(short_mapping[par2])]

        fig = pl.gcf()
        fig.clf()
        if self.chi2_r321303 is not 0:
            ax1 = pl.subplot(2,3,1)
            pl.contourf(xax, yax, self.chi2_r321303.min(axis=axis),
                        levels=self.chi2_r321303.min()+levels, alpha=0.5)
            pl.contour(xax, yax, self.chi2.min(axis=axis),
                       levels=self.chi2.min()+levels)
            pl.plot(xmaxlike, ymaxlike, 'o', markerfacecolor='none', markeredgecolor='k')
            pl.plot(xexpect, yexpect, 'x', markerfacecolor='none', markeredgecolor='k')
            if self.chi2_r321322:
                pl.contour(xax, yax, self.chi2_r321322.min(axis=axis),
                           levels=self.chi2_r321322.min()+levels,
                           cmap=pl.cm.bone)
            pl.title("Ratio $3_{2,1}-2_{2,0}/3_{0,3}-2_{0,2}$")

        if self.chi2_r404303 is not 0:
            ax5 = pl.subplot(2,3,5)
            pl.contourf(xax, yax, self.chi2_r404303.min(axis=axis),
                        levels=self.chi2_r404303.min()+levels, alpha=0.5)
            pl.contour(xax, yax, self.chi2.min(axis=axis),
                       levels=self.chi2.min()+levels)
            pl.plot(xmaxlike, ymaxlike, 'o', markerfacecolor='none', markeredgecolor='k')
            pl.plot(xexpect, yexpect, 'x', markerfacecolor='none', markeredgecolor='k')
            pl.title("Ratio $4_{0,4}-3_{2,2}/3_{0,3}-2_{0,2}$")

        if self.chi2_r422404 is not 0:
            ax6 = pl.subplot(2,3,6)
            pl.contourf(xax, yax, self.chi2_r422404.min(axis=axis),
                        levels=self.chi2_r422404.min()+levels, alpha=0.5)
            pl.contour(xax, yax, self.chi2.min(axis=axis),
                       levels=self.chi2.min()+levels)
            pl.plot(xmaxlike, ymaxlike, 'o', markerfacecolor='none', markeredgecolor='k')
            pl.plot(xexpect, yexpect, 'x', markerfacecolor='none', markeredgecolor='k')
            if self.chi2_r423404 is not 0:
                pl.contour(xax, yax, self.chi2_r423404.min(axis=axis),
                           levels=self.chi2_r423404.min()+levels,
                           cmap=pl.cm.bone)
            pl.title("Ratio $4_{2,2}-3_{2,1}/4_{0,4}-3_{2,2}$")

        ax4 = pl.subplot(2,3,2)
        if hasattr(self.chi2_X, 'size'):
            pl.contourf(xax, yax, self.chi2_X.min(axis=axis),
                        levels=self.chi2_X.min()+levels, alpha=0.5)
        pl.contour(xax, yax, self.chi2.min(axis=axis),
                   levels=self.chi2.min()+levels)
        pl.plot(xmaxlike, ymaxlike, 'o', markerfacecolor='none', markeredgecolor='k')
        pl.plot(xexpect, yexpect, 'x', markerfacecolor='none', markeredgecolor='k')
        pl.title("log(p-H$_2$CO/H$_2$) "
                 "$= {0:0.1f}\pm{1:0.1f}$".format(self.logabundance,
                                                  self.elogabundance))

        ax3 = pl.subplot(2,3,3)
        if hasattr(self.chi2_h2, 'size'):
            pl.contourf(xax, yax, self.chi2_h2.min(axis=axis),
                        levels=self.chi2_h2.min()+levels, alpha=0.5)
        pl.contour(xax, yax, self.chi2.min(axis=axis),
                   levels=self.chi2.min()+levels)
        pl.plot(xmaxlike, ymaxlike, 'o', markerfacecolor='none', markeredgecolor='k')
        pl.plot(xexpect, yexpect, 'x', markerfacecolor='none', markeredgecolor='k')
        pl.title("Total log$(N(\\mathrm{{H}}_2))$ ")
        #         "= {0:0.1f}\pm{1:0.1f}$".format(self.logh2column,
        #                                         self.elogh2column))
        ax5 = pl.subplot(2,3,4)
        #if hasattr(self.chi2_ff1, 'size'):
        #    pl.contourf(xax, yax, (self.chi2_ff1.min(axis=axis)),
        #                levels=self.chi2_ff1.min()+levels, alpha=0.5)
        if hasattr(self.chi2_dens, 'size'):
            pl.contourf(xax, yax, (self.chi2_dens.min(axis=axis)),
                        levels=self.chi2_dens.min()+levels, alpha=0.5)
        pl.contour(xax, yax, self.chi2.min(axis=axis),
                   levels=self.chi2.min()+levels)
        if hasattr(self, 'taline303'):
            pl.contour(xax, yax, (self.tline303 < 10*self.taline303).max(axis=axis),
                       levels=[0.5], colors='k')
        pl.plot(xmaxlike, ymaxlike, 'o', markerfacecolor='none', markeredgecolor='k')
        pl.plot(xexpect, yexpect, 'x', markerfacecolor='none', markeredgecolor='k')
        #pl.contour(xax, yax, (tline303 < 100*par1).max(axis=axis), levels=[0.5], colors='k')
        #pl.contour(xax, yax, (tline321 < 10*par2).max(axis=axis), levels=[0.5], colors='k', linestyles='--')
        #pl.contour(xax, yax, (tline321 < 100*par2).max(axis=axis), levels=[0.5], colors='k', linestyles='--')
        #pl.title("Line Brightness + $ff\leq1$")
        pl.title("Minimum Density")
        fig.text(0.05, 0.5, ylabel, horizontalalignment='center',
                verticalalignment='center',
                rotation='vertical', transform=fig.transFigure)
        fig.text(0.5, 0.02, xlabel, horizontalalignment='center', transform=fig.transFigure)


        if par1 == 'col':
            for ss in range(1,5):
                ax = pl.subplot(2,3,ss)
                ax.xaxis.set_ticks(np.arange(self.carr.min(), self.carr.max()))

        pl.subplots_adjust(wspace=0.25, hspace=0.45)

    def denstemplot(self):
        self.parplot('dens','tem')

    def denscolplot(self):
        self.parplot('col','dens')

    def coltemplot(self):
        self.parplot('col','tem')

