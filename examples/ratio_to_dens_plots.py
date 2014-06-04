import sys
sys.path.append("/Users/adam/work/h2co/lowdens/code/")
from h2co_modeling.smoothtau_models import SmoothtauModels
import pylab as pl
import itertools

modelpath = '/Users/adam/work/h2co/radex/troscompt_April2013_linearXH2CO/'
stm = SmoothtauModels(modelpath+'1-1_2-2_XH2CO=1e-9_troscompt.dat')

pl.rcParams['font.size'] = 20

pl.figure(1)
pl.clf()

linestyles = itertools.cycle(['-','--','-.',':'])

temperature = 20
opr = 1.0

for abund in (-8.5,-9,-9.5,-10):
    trot1,trot2,tex1,tex2,tau1,tau2,dens,col = stm.select_data(abundance=abund,
                                                               opr=opr,
                                                               temperature=temperature)
    tau,vtau,vtau_ratio = stm.generate_tau_functions(abundance=abund, opr=opr,
                                                     temperature=temperature)

    tauratio = vtau_ratio(dens, line1=tau1, line2=tau2, tex1=tex1, tex2=tex2,
                          sigma=1.0, opr=opr, temperature=temperature)

    #ok = np.arange(tauratio.size) > np.argmax(tauratio)

    #def ratio_to_dens(ratio):
    #    inds = np.argsort(tauratio[ok])
    #    return np.interp(ratio, tauratio[ok][inds], dens[ok][inds], np.nan, np.nan)

    pl.plot(dens,tauratio,label='$X($H$_2$CO$)=10^{%0.1f}$' % abund, linewidth=2, alpha=0.7,
            linestyle=linestyles.next())

pl.axis([0,6,0,13])
pl.legend(loc='best')
pl.xlabel(r'Volume-averaged density $\log(n(H_2))$')
pl.ylabel(r'Ratio $\tau_{1-1}/\tau_{2-2}$')

pl.savefig('/Users/adam/work/h2co/maps/paper/figures/tau_ratio_vs_density_thinlimit_t20_sigma1.pdf',bbox_inches='tight')
pl.show()
