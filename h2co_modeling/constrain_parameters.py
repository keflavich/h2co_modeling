"""
Functions for fitting temperature (and density and column) from the line ratio
plus whatever other constraints are available
"""
import inspect
import time
import collections
import warnings
import os

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy import stats
from astropy import units as u
from astropy import log
import pylab as pl
import matplotlib
from astropy.io import fits
from astropy.utils.console import ProgressBar

from . import grid_fitter
from .paraH2COmodel import generic_paraH2COmodel

short_mapping = {'dens': 'density',
                 'col': 'column',
                 'tem': 'temperature'}

chi2_mapping = {'X': 'Abundance',
                'ff1': "Filling Factor $3_{0,3}-2_{0,2}$",
                'ff2': "Filling Factor $3_{2,1}-2_{2,0}$",
                'r321303': "Ratio $3_{0,3}-2_{0,2}$ / $3_{2,1}-2_{2,0}$",
                'dens': "Density $n(H_2)$ cm$^{-3}$",
                'h2': "Column $N(H_2)$ cm$^{-2}$",
               }

def gpath(fn, gridpath='/Users/adam/work/h2co/radex/thermom/'):
    return os.path.join(gridpath, fn)

class paraH2COmodel(generic_paraH2COmodel):

    def __init__(self, tbackground=2.73, gridsize=[250.,101.,100.],
                 gpath=gpath, lines=['303','321','322','404','422','423'],
                 grid_linewidth=5.0,
                 fname_template='fjdu_pH2CO_{line}_{textau}_5kms.fits',
                 gridpath='/Users/adam/work/h2co/radex/thermom/'):
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
        gridpath : str
            The path that will be passed to the 'gpath' function
        lines : list
            List of string names of the lines to be put into the fname_template
        grid_linewidth : float
            The line width used to compute the model grid
        fname_template : str
            Filename template.   Must have input locations for the line name
            {line} and tex or tau {textau}
        """
        t0 = time.time()

        print("Loading grids...")
        for line in ProgressBar(lines):
            if os.path.exists(gpath(fname_template.format(line=line, textau='tex'))):
                setattr(self, 'texgrid{0}'.format(line),
                        fits.getdata(gpath(fname_template.format(line=line,
                                                                 textau='tex'))))
                setattr(self, 'taugrid{0}'.format(line),
                        fits.getdata(gpath(fname_template.format(line=line,
                                                                 textau='tau'))))
                #print("Found and loaded file {0}".format(gpath('fjdu_pH2CO_{0}_tex_5kms.fits'.format(line))))
            else:
                print("Did not find {0}".format(gpath('fjdu_pH2CO_{0}_tex_5kms.fits'.format(line))))

        # example old-style loading
        #self.texgrid303 = texgrid303 = fits.getdata(gpath('fjdu_pH2CO_303_tex_5kms.fits'))

        self.hdr = fits.getheader(gpath('fjdu_pH2CO_303_tex_5kms.fits'))
        self.grid_linewidth = grid_linewidth

        t1 = time.time()
        log.debug("Loading grids took {0:0.1f} seconds".format(t1-t0))

        self.Tbackground = tbackground
        self.tline303a = ((1.0-np.exp(-np.array(self.taugrid303))) *
                          (self.texgrid303-self.Tbackground))
        self.tline321a = ((1.0-np.exp(-np.array(self.taugrid321))) *
                          (self.texgrid321-self.Tbackground))
        self.tline322a = ((1.0-np.exp(-np.array(self.taugrid322))) *
                          (self.texgrid322-self.Tbackground))
        if hasattr(self, 'taugrid404'):
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
        print("Upsampling grids...")
        for line in ProgressBar(lines):
            if hasattr(self, 'tline{0}a'.format(line)):
                setattr(self, 'tline{0}'.format(line),
                        map_coordinates(getattr(self, 'tline{0}a'.format(line)),
                                        upsinds/upsample_factor[:,None,None,None],
                                        mode='nearest'))

    
        self.tline = {303: self.tline303,
                      321: self.tline321,
                      322: self.tline322,
                     }
        if hasattr(self, 'tline404a'):
            self.tline.update({
                      422: self.tline422,
                      423: self.tline423,
                      404: self.tline404,
            })

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
        if hasattr(self, 'tline404'):
            self.modelratio_423_404 = self.tline423/self.tline404
            self.modelratio_422_404 = self.tline422/self.tline404
            self.modelratio_404_303 = self.tline404/self.tline303

        self.model_logabundance = np.log10(10**self.columnarr / u.pc.to(u.cm) /
                                           10**self.densityarr)

        t2 = time.time()
        log.debug("Grid initialization took {0:0.1f} seconds total,"
                  " {1:0.1f} since loading grids.".format(t2-t0,t2-t1))

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


    def parplot_J32(self, par1='col', par2='dens', nlevs=5, levels=None,
                    colors=[(0.5,0,0), (0.75,0,0), (1.0,0,0), (1.0,0.25,0), (0.75,0.5,0)],
                    colorsf=[0.0, 0.33, 0.66, 1.0, 'w']):

        cdict = {x:   [(0.0, 0.0, 0.0),
                       (1.0, 1.0, 1.0)]
                 for x in ('red','green','blue')}
        cdict['blue'] = [(0.0, 1., 1.), (1.0, 1.0, 1.0)]
        cm = matplotlib.colors.LinearSegmentedColormap('mycm', cdict)
        colorsf = [cm(float(ii)) if isinstance(ii, (float,int))
                   else ii
                   for ii in colorsf]

        xax = self.axes[par1]
        yax = self.axes[par2]
        xlabel = self.labels[par1]
        ylabel = self.labels[par2]
        amapping = {('col','dens'): 0,
                    ('dens','tem'): 2,
                    ('col','tem'): 1}
        if (par1,par2) in amapping:
            axis = amapping[(par1,par2)]
            swaps = (0,0)
        elif (par2,par1) in amapping:
            axis = amapping[(par2,par1)]
            swaps = (0,1)

        if levels is None:
            levels = ([0]+[(stats.norm.cdf(ii)-stats.norm.cdf(-ii))
                           for ii in range(1,nlevs)]+[1])

        xmaxlike = self.parconstraints['{0}_chi2'.format(short_mapping[par1])]
        ymaxlike = self.parconstraints['{0}_chi2'.format(short_mapping[par2])]
        xexpect = self.parconstraints['expected_{0}'.format(short_mapping[par1])]
        yexpect = self.parconstraints['expected_{0}'.format(short_mapping[par2])]

        fig = pl.gcf()
        fig.clf()
        ax1 = pl.subplot(2,2,1)
        if 'chi2_r321303' in self.individual_likelihoods:
            like = (self.individual_likelihoods['chi2_r321303'])
            pl.contourf(xax, yax, cdf_of_like(like.sum(axis=axis)).swapaxes(*swaps),
                        levels=levels, alpha=0.5, zorder=-5, colors=colorsf)
        pl.contour(xax, yax,
                   cdf_of_like(self.likelihood.sum(axis=axis)).swapaxes(*swaps),
                   levels=levels, colors=colors, zorder=10)
        pl.plot(xmaxlike, ymaxlike, 'o', markerfacecolor='none', markeredgecolor='k')
        pl.plot(xexpect, yexpect, 'x', markerfacecolor='none', markeredgecolor='k')
        if self.chi2_r321322 is not 0:
            like = cdf_of_like(self.individual_likelihoods['chi2_r321322'])
            pl.contour(xax, yax, like.sum(axis=axis).swapaxes(*swaps),
                       levels=levels,
                       cmap=pl.cm.bone)
        pl.title("Ratio $3_{0,3}-2_{0,2}/3_{2,1}-2_{2,0}$")

        ax4 = pl.subplot(2,2,2)
        if hasattr(self.chi2_X, 'size'):
            like = self.individual_likelihoods['chi2_X']
            pl.contourf(xax, yax, cdf_of_like(like.sum(axis=axis)).swapaxes(*swaps),
                        levels=levels, alpha=0.5, zorder=-5, colors=colorsf)
        pl.contour(xax, yax,
                   cdf_of_like(self.likelihood.sum(axis=axis)).swapaxes(*swaps),
                   levels=levels, colors=colors, zorder=10)
        pl.plot(xmaxlike, ymaxlike, 'o', markerfacecolor='none', markeredgecolor='k')
        pl.plot(xexpect, yexpect, 'x', markerfacecolor='none', markeredgecolor='k')
        pl.title("log(p-H$_2$CO/H$_2$) "
                 "$= {0:0.1f}\pm{1:0.1f}$".format(self.logabundance,
                                                  self.elogabundance))

        ax3 = pl.subplot(2,2,3)
        if hasattr(self.chi2_h2, 'size'):
            like = (self.individual_likelihoods['chi2_h2'])
            pl.contourf(xax, yax, cdf_of_like(like.sum(axis=axis)).swapaxes(*swaps),
                        levels=levels, alpha=0.5, zorder=-5, colors=colorsf)
        pl.contour(xax, yax,
                   cdf_of_like(self.likelihood.sum(axis=axis)).swapaxes(*swaps),
                   levels=levels, colors=colors, zorder=10)
        pl.plot(xmaxlike, ymaxlike, 'o', markerfacecolor='none', markeredgecolor='k')
        pl.plot(xexpect, yexpect, 'x', markerfacecolor='none', markeredgecolor='k')
        pl.title("Total log$(N(\\mathrm{{H}}_2))$ ")
        #         "= {0:0.1f}\pm{1:0.1f}$".format(self.logh2column,
        #                                         self.elogh2column))
        ax5 = pl.subplot(2,2,4)
        if hasattr(self.chi2_ff1, 'size'):
            cdict = {x:   [(0.0, 0.5, 0.5),
                           (1.0, 0.0, 0.0)]
                     for x in ('red','green','blue')}
            cdict['green'] = [(0, 0.5, 0.5), (1,1,1)]
            cdict['red'] = [(0, 0.5, 0.5), (1,0.7,0.7)]
            cdict['blue'] = [(0, 0.0, 0.0), (1,0,0)]
            #cdict['alpha'] = [(0.0, 0.0, 0.0), (1.0, 0.3, 0.3)]
            darker = matplotlib.colors.LinearSegmentedColormap('darker', cdict)
            like = (self.individual_likelihoods['chi2_ff1'])
            plim = cdf_of_like(like.sum(axis=axis)).swapaxes(*swaps)
            pl.contour(xax, yax, plim, levels=levels,
                        cmap=darker, zorder=5)
        if hasattr(self.chi2_dens, 'size'):
            like = (self.individual_likelihoods['chi2_dens'])
            pl.contourf(xax, yax, cdf_of_like(like.sum(axis=axis)).swapaxes(*swaps),
                        levels=levels, alpha=0.5, zorder=-5, colors=colorsf)
        pl.contour(xax, yax,
                   cdf_of_like(self.likelihood.sum(axis=axis)).swapaxes(*swaps),
                   levels=levels, colors=colors, zorder=10)
        #if hasattr(self, 'taline303'):
        #    ff1_mask = (self.tline303 < 10*self.taline303)
        #    pl.contour(xax, yax, ff1_mask.max(axis=axis).swapaxes(*swaps),
        #               levels=[0.5], colors='k')
        pl.plot(xmaxlike, ymaxlike, 'o', markerfacecolor='none', markeredgecolor='k')
        pl.plot(xexpect, yexpect, 'x', markerfacecolor='none', markeredgecolor='k')
        #pl.contour(xax, yax, (tline303 < 100*par1).max(axis=axis).swapaxes(*swaps), levels=[0.5], colors='k')
        #pl.contour(xax, yax, (tline321 < 10*par2).max(axis=axis).swapaxes(*swaps), levels=[0.5], colors='k', linestyles='--')
        #pl.contour(xax, yax, (tline321 < 100*par2).max(axis=axis).swapaxes(*swaps), levels=[0.5], colors='k', linestyles='--')
        #pl.title("Line Brightness + $ff\leq1$")
        pl.title("Minimum Density & $ff$")
        fig.text(0.05, 0.5, ylabel, horizontalalignment='center',
                verticalalignment='center',
                rotation='vertical', transform=fig.transFigure)
        fig.text(0.5, 0.02, xlabel, horizontalalignment='center', transform=fig.transFigure)


        if par1 == 'col':
            for ss in range(1,5):
                ax = pl.subplot(2,2,ss)
                ax.xaxis.set_ticks(np.arange(self.carr.min(), self.carr.max()))

        pl.subplots_adjust(wspace=0.25, hspace=0.45)


    def parplot_J43(self, par1='col', par2='dens', nlevs=5, levels=None,
                colors=[(0.5,0,0), (0.75,0,0), (1.0,0,0), (1.0,0.25,0), (0.75,0.5,0)],
                colorsf=[0.0, 0.33, 0.66, 1.0, 'w']):

        cdict = {x:   [(0.0, 0.0, 0.0),
                       (1.0, 1.0, 1.0)]
                 for x in ('red','green','blue')}
        cdict['blue'] = [(0.0, 1., 1.), (1.0, 1.0, 1.0)]
        cm = matplotlib.colors.LinearSegmentedColormap('mycm', cdict)
        colorsf = [cm(float(ii)) if isinstance(ii, (float,int))
                   else ii
                   for ii in colorsf]

        xax = self.axes[par1]
        yax = self.axes[par2]
        xlabel = self.labels[par1]
        ylabel = self.labels[par2]
        amapping = {('col','dens'): 0,
                    ('dens','tem'): 2,
                    ('col','tem'): 1}
        if (par1,par2) in amapping:
            axis = amapping[(par1,par2)]
            swaps = (0,0)
        elif (par2,par1) in amapping:
            axis = amapping[(par2,par1)]
            swaps = (0,1)

        if levels is None:
            levels = ([0]+[(stats.norm.cdf(ii)-stats.norm.cdf(-ii))
                           for ii in range(1,nlevs)]+[1])

        xmaxlike = self.parconstraints['{0}_chi2'.format(short_mapping[par1])]
        ymaxlike = self.parconstraints['{0}_chi2'.format(short_mapping[par2])]
        xexpect = self.parconstraints['expected_{0}'.format(short_mapping[par1])]
        yexpect = self.parconstraints['expected_{0}'.format(short_mapping[par2])]


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


    def parplot1d(self, par='col', levels=None, clf=True,
                  legend=True, legendfontsize=14):

        xax = self.axes[par]
        xlabel = self.labels[par]
        amapping = {'col':(2,(0,1)),
                    'dens':(1,(0,2)),
                    'tem':(0,(1,2))}
        axis,axes = amapping[par]


        xmaxlike = self.parconstraints['{0}_chi2'.format(short_mapping[par])]
        xexpect = self.parconstraints['expected_{0}'.format(short_mapping[par])]

        like = self.likelihood.sum(axis=axes)
        like /= like.sum()
        inds_cdf = np.argsort(like)
        cdf = like[inds_cdf]

        fig = pl.gcf()
        if clf:
            fig.clf()
        ax = fig.gca()
        ax.plot(xax, like, 'k-', label='Posterior')

        for key in self.individual_likelihoods:
            if key in ('chi2','_chi2'):
                continue # already done
            ilike = self.individual_likelihoods[key].sum(axis=axes)
            ilike /= ilike.sum()
            ax.plot(xax, ilike, label=chi2_mapping[key.replace("chi2_","")])

        ax.vlines((xmaxlike,), 0, like.max(), linestyle='--', color='r',
                  label='Maximum Likelihood')

        ax.vlines((xexpect,), 0, like.max(), linestyle='--', color='b',
                  label='E[{0}]'.format(xlabel))
        xexpect_v2 = (like*xax).sum()/like.sum()
        ax.vlines((xexpect_v2,), 0, like.max(), linestyle='--', color='c',
                  zorder=-1)
        print("par:{4} xmaxlike: {0}, xexpect: {1}, xexpect_v2: {2},"
              "maxlike: {3}, diff:{5}"
              .format(xmaxlike, xexpect, xexpect_v2, like.max(), par,
                      xexpect-xmaxlike))

        if levels is not None:
            if not isinstance(levels, collections.Iterable):
                levels = [levels]
            cdf_inds = np.argsort(like)
            ppf = 1-like[cdf_inds].cumsum()
            cutoff_likes = [like[cdf_inds[np.argmin(np.abs(ppf-lev))]]
                            for lev in levels]

            for fillind,cutoff in enumerate(sorted(cutoff_likes)):
                selection = like > cutoff
                ax.fill_between(xax[selection], like[selection]*0,
                                like[selection], alpha=0.1, zorder=fillind-20)
                if np.abs(like[selection].sum() - levels[0]) > 0.05:
                    # we want the sum of the likelihood to be right!
                    #import ipdb; ipdb.set_trace()
                    warnings.warn("Likelihood is not self-consistent.")



        if legend:
            ax.legend(loc='best', fontsize=legendfontsize)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('$P(${0}$)$'.format(xlabel))

    def parplot1d_all(self, legendfontsize=14, **kwargs):

        fig = pl.gcf()
        if not all(fig.get_size_inches() == [12,16]):
            num = fig.number
            pl.close(fig)
            fig = pl.figure(num, figsize=(12,16))

        for axindex,par in enumerate(('col','dens','tem')):
            ax = fig.add_subplot(3,1,axindex+1)
            self.parplot1d(par=par, clf=False, legend=False, **kwargs)

            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            if axindex == 1:
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                          fontsize=legendfontsize)

        pl.subplots_adjust(hspace=0.45)

    @property
    def individual_likelihoods(self):
        if hasattr(self, '_likelihoods') and self._likelihoods is not None:
            return self._likelihoods
        else:
            self._likelihoods = {}
            for key in self.__dict__:
                if 'chi2' in key and getattr(self,key) is not 0:
                    self._likelihoods[key] = np.exp(-getattr(self,key)/2.)
                    self._likelihoods[key] /= self._likelihoods[key].sum()
            return self._likelihoods

def cdf_of_like(like):
    """
    There is probably an easier way to do this, BUT it works:
    Turn a likelihood image into a CDF image
    """
    like = like/like.sum()
    order = np.argsort(like.flat)[::-1]
    cdf = like.flat[order].cumsum()[np.argsort(order)].reshape(like.shape)
    cdf[like == like.max()] = 0
    return cdf

def ppf_of_like(like):
    return 1-cdf_of_like(like)
