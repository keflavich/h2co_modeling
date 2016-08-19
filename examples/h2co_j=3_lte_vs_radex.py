import matplotlib
import numpy as np
import pylab as pl

from astropy import log
from astropy import units as u

import pyradex
from h2co_modeling.temperature_mapper import ph2cogrid, TemperatureMapper

from h2co_modeling import lte_model

"""
It seems that RADEX never converges to LTE because the optical depths of the
lines are never the same.
"""

if 'tm' not in locals():
    tm = TemperatureMapper(logdensities=[3,4,5,6,7,8], abundances=(1.2e-9,))
    tm2 = TemperatureMapper(logdensities=[4], deltav=20.0)
    tm3 = TemperatureMapper(logdensities=[4], deltav=1.0)
    tm4 = TemperatureMapper(logdensities=[4], abundances=(1e-8,1e-10))
    tm5 = TemperatureMapper(logdensities=[3,4,5,6,7,8], abundances=(1e-13,))

if not hasattr(tm, 'temperatures'):
    tm.init()
if not hasattr(tm2, 'temperatures'):
    tm2.init()
if not hasattr(tm3, 'temperatures'):
    tm3.init()
if not hasattr(tm4, 'temperatures'):
    tm4.init()
if not hasattr(tm5, 'temperatures'):
    tm5.init()

fig2 = pl.figure(2)
fig2.clf()
ax2 = pl.gca()
fig3 = pl.figure(3)
fig3.clf()
ax3 = pl.gca()

ax2.set_xlabel("Ratio $S(3_{2,1}-2_{2,0})/S(3_{0,3}-2_{0,2})$")
ax2.set_ylabel("Temperature (K)")
ax3.set_xlabel("Ratio $S(3_{2,1}-2_{2,0})/S(3_{0,3}-2_{0,2})$")
ax3.set_ylabel("Temperature (K)")
ax2.set_title("$X=1.2\\times10^{9}$")
ax3.set_title("Optically Thin")

R = pyradex.Radex(species='ph2co-h2', temperature=50, density=1e6, abundance=1e-9)
if 'radex_lte' not in locals():
    radex_lte = [R(column=1e10, temperature=T, density=1e9)[np.array([2,9,12])] for T in tm.temperatures]
    ratio1_lte = [x['T_B'][2]/x['T_B'][0] for x in radex_lte]


L1, = ax2.plot(tm.Xarr[1.2e-9]['ratio1'][1e3], tm.temperatures, 'k-.', label=r'$n(H_2)=10^3$ cm$^{-3}$', zorder=-5)
L2, = ax2.plot(tm.Xarr[1.2e-9]['ratio1'][1e4], tm.temperatures, 'k--', label=r'$n(H_2)=10^4$ cm$^{-3}$', zorder=-5)
L3, = ax2.plot(tm.Xarr[1.2e-9]['ratio1'][1e5], tm.temperatures, 'k:',  label=r'$n(H_2)=10^5$ cm$^{-3}$', zorder=-5)
L11, = ax2.plot(tm.Xarr[1.2e-9]['ratio1'][1e6], tm.temperatures, 'b:',  label=r'$n(H_2)=10^6$ cm$^{-3}$', zorder=-5)
L12, = ax2.plot(tm.Xarr[1.2e-9]['ratio1'][1e7], tm.temperatures, 'b--',  label=r'$n(H_2)=10^7$ cm$^{-3}$', zorder=-5)
L9, = ax2.plot(tm.Xarr[1.2e-9]['ratio1'][1e8], tm.temperatures, '--', color='#7A4446',  label=r'$n(H_2)=10^8$ cm$^{-3}$', zorder=-5, alpha=0.5, linewidth=4)
#L10, = ax2.plot(ratio1_lte, tm.temperatures, '-', color='#7A44DD',  label=r'LTE', zorder=-6, linewidth=5, alpha=0.5)


L1, = ax3.plot(tm5.Xarr[1e-13]['ratio1'][1e3], tm.temperatures, 'k-.', label=r'$n(H_2)=10^3$ cm$^{-3}$', zorder=-5)
L2, = ax3.plot(tm5.Xarr[1e-13]['ratio1'][1e4], tm.temperatures, 'k--', label=r'$n(H_2)=10^4$ cm$^{-3}$', zorder=-5)
L3, = ax3.plot(tm5.Xarr[1e-13]['ratio1'][1e5], tm.temperatures, 'k:',  label=r'$n(H_2)=10^5$ cm$^{-3}$', zorder=-5)
L11, = ax3.plot(tm5.Xarr[1e-13]['ratio1'][1e6], tm.temperatures, 'b:',  label=r'$n(H_2)=10^6$ cm$^{-3}$', zorder=-5)
L12, = ax3.plot(tm5.Xarr[1e-13]['ratio1'][1e7], tm.temperatures, 'b--',  label=r'$n(H_2)=10^7$ cm$^{-3}$', zorder=-5)
L9, = ax3.plot(tm5.Xarr[1e-13]['ratio1'][1e8], tm5.temperatures, '--', color='#7A4446',  label=r'$n(H_2)=10^8$ cm$^{-3}$', zorder=-5, alpha=0.5, linewidth=4)
#L10, = ax3.plot(ratio1_lte, tm.temperatures, '-', color='#7A44DD',  label=r'LTE', zorder=-6, linewidth=5, alpha=0.5)


L8, = ax2.plot(lte_model.T_321/lte_model.T_303, lte_model.tem,
               color='orange', linewidth=2,
               label=r'LTE', zorder=-4)
L8, = ax3.plot(lte_model.T_321/lte_model.T_303, lte_model.tem,
               color='orange', linewidth=2,
               label=r'LTE', zorder=-4)
leg2 = ax2.legend(loc='upper left', fontsize=14)
leg3 = ax3.legend(loc='upper left', fontsize=14)
ax2.axis([0,1,10,300])
ax3.axis([0,1,10,300])

fig2.savefig("models_temperature_vs_ratio_LTE_and_RADEX_X1.2e-9.png")
fig3.savefig("models_temperature_vs_ratio_LTE_and_RADEX_thin.png")

#with open("lte_model.py") as f:
#    code = compile(f.read(), "lte_model.py", 'exec')
#    exec(code, globals(), locals())
#
#fig1 = pl.figure(1)
#ax2 = pl.subplot(3,1,2)
#fivekms_nu = (5*u.km/u.s / constants.c) * ph2co_303['frequency']
#ax2.plot(tm.temperatures, [x['tau'][0]*fivekms_nu.to(u.Hz).value for x in radex_lte], color='b', linestyle='--')
#ax2.plot(tm.temperatures, [x['tau'][2]*fivekms_nu.to(u.Hz).value for x in radex_lte], color='r', linestyle='--')
#
#for sp in fig1.get_axes():
#    lines = sp.get_lines()
#    ymin = np.inf
#    ymax = -np.inf
#    for L in lines:
#        x,y = L.get_data()
#        ymin = np.min([ymin, y[x>50].min()])
#        ymax = np.max([ymax, y[x>50].max()])
#    sp.axis([50,200,ymin,ymax])
#
#fig1.savefig("LTEvsRADEX_50to200.png")
#
#for sp in fig1.get_axes():
#    lines = sp.get_lines()
#    ymin = np.inf
#    ymax = -np.inf
#    for L in lines:
#        x,y = L.get_data()
#        ymin = np.min([ymin, y[x<50].min()])
#        ymax = np.max([ymax, y[x<50].max()])
#    sp.axis([0,50,ymin,ymax])
#
#fig1.savefig("LTEvsRADEX_upto50.png")


# analytic LTE solution
# computed by taking ratio of absorption coefficients, 12.17 in Wilson+2009
# (if B_nu is expressed in K and we're in the RT regime, the source
# function is independent
def ratio_lte_analytic(temperature,
                       nu1=218.222192*u.GHz, nu2=218.760066*u.GHz,
                       eu1=21.0*u.K, eu2=68*u.K,
                       Aul1=2.818e-04*u.Hz, Aul2=1.577e-04*u.Hz):
    from astropy import constants
    hoverkt = constants.h/(constants.k_B*temperature)
    term1 = (nu2/nu1)**2
    term2 = np.exp(-eu1/temperature)/np.exp(-eu2/temperature)
    term3 = Aul1/Aul2
    term4 = (1-np.exp(-hoverkt*nu1))/(1-np.exp(-hoverkt*nu2))
    return term1 * term2 * term3 * term4
