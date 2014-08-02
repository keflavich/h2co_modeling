import numpy as np
import pylab as pl
from matplotlib.ticker import MaxNLocator

from h2co_modeling.smoothtau_models import SmoothtauModels

modelpath = '/Users/adam/work/h2co/radex/troscompt_April2013_linearXH2CO/'
stm = SmoothtauModels(modelpath+'1-1_2-2_XH2CO=1e-9_troscompt.dat')

abund = -9
temperature = 20
opr = 0.1
sigma = 1.0
#tau1,tau2,dens,col = stm.select_data(abund)
trot1,trot2,tex1,tex2,tau1,tau2,dens,col = stm.select_data(abundance=abund,
                                                           opr=opr,
                                                           temperature=temperature)

tau,vtau,vtau_ratio = stm.generate_tau_functions(abundance=abund, opr=opr,
                                                 temperature=temperature)
tline,vtline,vtline_ratio,tline1,tline2 = stm.generate_tline_functions(abundance=abund,
                                                                       opr=opr,
                                                                       temperature=temperature)

tbg1 = 2.73
tbg2 = 2.73
mindens = 1

for fignum in (1,2,3):
    pl.figure(fignum)
    pl.clf()

pl.rcParams['lines.linewidth'] = 2
pl.rcParams['axes.color_cycle'] = ["#"+x for x in "348ABD, 7A68A6, A60628, 467821, CF4457, 188487, E24A33".split(", ")]
pl.rcParams['xtick.major.pad'] = 8
pl.rcParams['ytick.major.pad'] = 8

def get_inflection_point(x, w=2, threshold=0.01):
    from astropy.convolution import convolve, Gaussian1DKernel
    sm = convolve(x, Gaussian1DKernel(w), boundary='extend')
    pos = np.diff(sm) > threshold
    return np.argmax(pos)
    return np.argmin(np.abs(np.diff(x)))

def get_inflection_points(x, w=2, threshold=0.01):
    """
    Return the valid range in pixels
    """
    from astropy.convolution import convolve, Gaussian1DKernel
    sm = convolve(x, Gaussian1DKernel(w), boundary='extend')
    pos = np.diff(sm) > threshold
    # Find the first point at which the slope is positive
    # (this is very near the 0/0 point)
    pospt = np.argmax(pos)
    if np.count_nonzero(pos) < 5:
        # implies there were no inflection points found
        return 0,x.size
    elif pospt == 0:
        # find first *negative* inflection point
        negpt = np.argmin(pos)
        pos[:negpt] = False
        pospt = np.argmax(pos)
        return negpt,pospt
    else:
        return 0,pospt

for sigma in [0.01,0.5,1.0]:
    for ii,(tbg1, tbg2) in enumerate(((2.73,2.73),(75,9))):
        pl.figure(1)
        pl.subplot(2,1,ii+1)
        pl.title("$T_{{BG}} = {0}$".format(tbg1))
        tl1 = vtline(dens, lvg_tau=tau1, tex=tex1, sigma=sigma, tbg=tbg1,
                     obs_tau=True)
        ok = (tl1 > 0) & (dens > mindens)
        pl.plot(dens[ok], tl1[ok],
                label='$\\sigma={s}$'.format(s=sigma))
        pl.xlabel("log($n(H_2) [\mathrm{cm}^{-3}]$")
        pl.ylabel("$\\tau_{obs}(1-1)$")
        pl.legend(loc='best')

        pl.figure(2)
        pl.subplot(2,1,ii+1)
        pl.title("$T_{{BG}} = {0}$".format(tbg2))
        tl2 = vtline(dens, lvg_tau=tau2, tex=tex2, sigma=sigma, tbg=tbg2,
                     obs_tau=True)
        ok = (tl2 > 0) & (dens > mindens)
        pl.plot(dens[ok], tl2[ok],
                label='$\\sigma={s}$'.format(s=sigma))
        pl.xlabel("log($n(H_2) [\mathrm{cm}^{-3}]$")
        pl.ylabel("$\\tau_{obs}(2-2)$")
        pl.legend(loc='best')

        fig = pl.figure(3)
        pl.subplot(2,1,ii+1)
        pl.title("$T_{{BG}} = {0},{1}$".format(tbg1,tbg2))
        tlr = vtline_ratio(dens, sigma=sigma, tbg1=tbg1, tbg2=tbg2,
                           obs_tau_ratio=True)
        ok = (tl1 > 0.00001) & (tl2 > 0.00001) & (dens>3)
        ##ok[ok] &= (np.arange(np.count_nonzero(ok)) < get_inflection_point(tlr[ok]))
        negpt,pospt = get_inflection_points(tlr[ok])
        # therefore we modify the OK part by deselecting everything, then selecting
        # the parts that are good
        first_ind = np.argmax(ok)
        ok = np.zeros(dens.size, dtype='bool')
        ok[:first_ind+pospt] = True
        if ok.sum() < 5:
            import pdb; pdb.set_trace()

        ok &= (dens > mindens) 
        l, = pl.plot(dens[ok], tlr[ok], label='$\\sigma={s}$'.format(s=sigma))
        pl.plot(dens[ok], tl1[ok]/tl2[ok], color=l.get_color(), linestyle='--')
        #pl.ylim(0,25)
        pl.xlim(1,6)
        pl.xlabel("log($n(H_2) [\mathrm{cm}^{-3}]$")
        #pl.ylabel("Ratio $\\tau_{obs}(1-1) / \\tau_{obs}(2-2)$")
        fig.text(0.5, 0.04, "Ratio $\\tau_{obs}(1-1) / \\tau_{obs}(2-2)$",
                 ha='center', va='center')
        
        pl.legend(loc='best')


for fignum in (1,2,3):
    fig = pl.figure(fignum)
    fig.subplots_adjust(hspace=0)
    ax = pl.subplot(2,1,1)
    ax.xaxis.set_ticklabels([])
    ax = pl.subplot(2,1,2)
    title = ax.get_title()
    ax.set_title("")
    ax.yaxis.set_major_locator(MaxNLocator(6,prune='upper'))
    ax.text(0.5, 0.90, title,
         horizontalalignment='center',
         transform=ax.transAxes)

fig = pl.figure(3)
fig.savefig("tau_ratio_vs_density_varybackground_vary_sigma.pdf")


# pl.figure(4)
# pl.clf()
# for sigma in (0.1,1.0):
#     for tbg1 in (2.73, 10):
#         pl.plot(vtline(dens, lvg_tau=tau1, tex=tex1, sigma=sigma, tbg=tbg1, obs_tau=True),
#                 vtau(dens, lvg_tau=tau1, tex=tex1, sigma=sigma, tbg=tbg1),
#                 label='$T_{{BG}} = {0}$, $\sigma = {1}$'.format(tbg1,sigma))
# pl.axis([0,1,0,1])
# pl.legend(loc='best')
# pl.xlabel("Observed optical depth, $-\log(T_{MB}/T_{BG})$")
# pl.ylabel("True optical depth, $\\tau$")
# pl.savefig("observed_vs_true_lowTbg.pdf")
# 
# # Purely for the hilarity
# pl.figure(5)
# pl.clf()
# for sigma in (0.01,0.1,1.0):
#     for tbg1 in (100,):
#         pl.plot(vtline(dens, lvg_tau=tau1, tex=tex1, sigma=sigma, tbg=tbg1, obs_tau=True),
#                 vtau(dens, lvg_tau=tau1, tex=tex1, sigma=sigma, tbg=tbg1),
#                 label='$T_{{BG}} = {0}$, $\sigma = {1}$'.format(tbg1,sigma))
# pl.axis([0,1.6,0,1.6])
# pl.legend(loc='best')
# pl.xlabel("Observed optical depth, $-\log(T_{MB}/T_{BG})$")
# pl.ylabel("True optical depth, $\\tau$")
# pl.savefig("observed_vs_true_highTbg.pdf")

pl.show()
