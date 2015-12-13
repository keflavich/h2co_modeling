from __future__ import print_function
import os
import numpy as np
from astropy.io import fits
from astropy import log

def ph2cogrid(ntemp=50, trange=[10,200], abundances=(1.2e-9,),
              logdensities=(4,5), opr=3, deltav=5.0, # km/s
             ):
    """
    Create a model grid with some ``ntemp`` of temperatures in the range
    ``trange``.
    """
    import pyradex

    temperatures=np.linspace(trange[0],trange[1],ntemp)

    # initial density; will be modified later
    density = 1e4


    fortho = opr/(1.+opr)

    R = pyradex.Radex(species='ph2co-h2',
                      abundance=abundances[0],
                      collider_densities={'H2':density},
                      deltav=deltav,
                      column=None,
                      temperature=temperatures[0],
                      )

    Xarr = {}
    for abundance in abundances:
        Xarr[abundance] = {}

        densities = [10**x for x in logdensities]
        ratio1 = {d:[] for d in densities}
        ratio2 = {d:[] for d in densities}
        f1 = {d:[] for d in densities}
        f2 = {d:[] for d in densities}
        f3 = {d:[] for d in densities}


        for density in densities:
            #R.density = {'H2': density}
            R.density = {'oH2':density*fortho,'pH2':density*(1.-fortho)}
            for temperature in temperatures:
                R.temperature = temperature
                R.abundance = abundance
                log.info("niter={0} col={1:0.1f} dens={2:0.1f} tem={3:0.1f} abund={4:0.2g}"
                         .format(R.run_radex(validate_colliders=False,
                                          reload_molfile=False),
                              np.log10(R.column.value),
                              np.log10(R.density['oH2'].value),
                              R.temperature,
                              R.abundance))

                #import ipdb; ipdb.set_trace()

                F1 = R.T_B[2]  # 218.222192 3_0_3
                F2 = R.T_B[12] # 218.760066 3_2_1
                F3 = R.T_B[9]  # 218.475632 3_2_2

                ratio1[density].append(F2/F1)
                ratio2[density].append(F3/F1)
                f3[density].append(F3)
                f2[density].append(F2)
                f1[density].append(F1)
            print()

        f1 = {d:np.array([x.value for x in f1[d]]) for d in densities}
        f2 = {d:np.array([x.value for x in f2[d]]) for d in densities}
        f3 = {d:np.array([x.value for x in f3[d]]) for d in densities}
        ratio1 = {d:np.array(ratio1[d]) for d in densities}
        ratio2 = {d:np.array(ratio2[d]) for d in densities}

        Xarr[abundance] = {'flux1':f1,
                           'flux2':f2,
                           'flux3':f3,
                           'ratio1':ratio1,
                           'ratio2':ratio2}

    return Xarr

class TemperatureMapper(object):
    """
    For lazier evaluation of temperature mapping function
    """
    def __init__(self, trange=[10,300], ntemp=100, logdensities=(4,4.5,5),
                 abundances=(1e-8, 1.2e-9, 1e-10), **kwargs):
        self.trange = trange
        self.ntemp = ntemp
        self.kwargs = kwargs
        self.kwargs['logdensities'] = logdensities
        self.kwargs['abundances'] = abundances

    def init(self):
        self.Xarr = ph2cogrid(trange=self.trange, ntemp=self.ntemp,
                              **self.kwargs)
        self.temperatures = np.linspace(self.trange[0], self.trange[1],
                                        self.ntemp)


    def get_mapper(self, lineid, tmin=np.nan, tmax=np.nan,
                   density=1e4, abundance=1.2e-9):
        if not hasattr(self,'temperatures'):
            self.init()

        rationame = {'321220': 'ratio1',
                     '322221': 'ratio2'}[lineid]

        # ugly hack because ph2co is indexed with floats
        # Use FIXED abundance, FIXED column, FIXED density
        ratios = self.Xarr[abundance][rationame][density]

        def ratio_to_tem(r):
            inds = np.argsort(ratios)
            return np.interp(r, ratios[inds], self.temperatures[inds], tmin,
                             tmax)

        return ratio_to_tem

    def __call__(self, x, lineid='321220', **kwargs):
        return self.get_mapper(lineid, **kwargs)(x)

if 'tm' not in locals():
    tm = TemperatureMapper(trange=[10,300],ntemp=100)
