from pyspeckit.spectrum.models.lte_molecule import line_tau, line_brightness
import numpy as np
from astropy import units as u
from astropy import constants

kb_cgs = constants.k_B.cgs
h_cgs = constants.h.cgs
eightpicubed = 8 * np.pi**3
threehc = 3 * constants.h.cgs * constants.c.cgs

# o-h2co 2_12-2_11
J = 2
gI = 3/4.# (2 I + 1) (I + 1) / (2 I + 1)^sigma; sigma=2; I=1/2
gJ = 2 * J + 1
gK = 1 # K-level doubling removes the degeneracy in (slightly) asymmetric tops

oh2co = {'tex':18.75*u.K,
         'total_column': 1e12*u.cm**-2,
         'partition_function': 44.6812, # splatalogue's 18.75
         'degeneracy': gI*gJ*gK,
         #'dipole_moment': 2.331e-18*u.esu*u.cm, #2.331*u.debye,
        }

oh2co_212 = {
         'frequency': 14.48849*u.GHz,
         'energy_upper': kb_cgs*22.61771*u.K,
}
oh2co_212.update(oh2co)
oh2co_212['dnu'] = (1*u.km/u.s/constants.c * oh2co_212['frequency'])


# 303
J = 3
gI = 0.25 # according to Mangum 2015, p24, this is 0.75? only for ortho
gJ = 2*J+1
gK_303 = 1 # supposedly, k!=0 has gK=2, but K=0 has gK=1?
gK_321 = 1 # however, LAMDA has identical degeneracies=7 for all 3
gK_322 = 1

# 321 has same parameters for g
# except maybe that gK=2

ph2co = {'tex':18.75*u.K,
         'total_column': 1e12*u.cm**-2,
         'partition_function': 44.6812, # splatalogue's 18.75
         #'dipole_moment': 2.331e-18*u.esu*u.cm, #2.331*u.debye,
        }

ph2co_303 = {'frequency': 218.22219*u.GHz,
             'energy_upper': kb_cgs*20.95582*u.K,
             'einstein_A': 10**-3.55007/u.s,
             'degeneracy': gI*gJ*gK_303,
}
ph2co_303.update(ph2co)
ph2co_303['dnu'] = (1*u.km/u.s/constants.c * ph2co_303['frequency'])

ph2co_321 = {'frequency': 218.76007*u.GHz,
             'energy_upper': kb_cgs*68.11081*u.K,
             'einstein_A': 10**-3.80235/u.s,
             'degeneracy': gI*gJ*gK_321,
}
ph2co_321.update(ph2co)
ph2co_321['dnu'] = (1*u.km/u.s/constants.c * ph2co_321['frequency'])

ph2co_322 = {'frequency': 218.47563*u.GHz,
             'energy_upper': kb_cgs*68.0937*u.K,
             'einstein_A': 10**-3.80373/u.s,
             'degeneracy': gI*gJ*gK_322,
}
ph2co_322.update(ph2co)
ph2co_322['dnu'] = (1*u.km/u.s/constants.c * ph2co_322['frequency'])

# CDMS Q
import requests
import bs4
url = 'http://cdms.ph1.uni-koeln.de/cdms/tap/'
rslt = requests.post(url+"/sync", data={'REQUEST':"doQuery", 'LANG': 'VSS2', 'FORMAT':'XSAMS', 'QUERY':"SELECT SPECIES WHERE MoleculeStoichiometricFormula='CH2O'"})
bb = bs4.BeautifulSoup(rslt.content, 'html5lib')
h = [x for x in bb.findAll('molecule') if x.ordinarystructuralformula.value.text=='H2CO'][0]
tem_, Q_ = h.partitionfunction.findAll('datalist')
tem = [float(x) for x in tem_.text.split()]
Q = [float(x) for x in Q_.text.split()]

del ph2co_303['tex']
del ph2co_303['partition_function']
T_303 = np.array([line_brightness(tex=tex*u.K, partition_function=pf,
                                  **ph2co_303).value for tex,pf in
                  zip(tem,Q)])

del ph2co_321['tex']
del ph2co_321['partition_function']
T_321 = np.array([line_brightness(tex=tex*u.K, partition_function=pf,
                                  **ph2co_321).value for tex,pf in
                  zip(tem,Q)])

del ph2co_322['tex']
del ph2co_322['partition_function']
T_322 = np.array([line_brightness(tex=tex*u.K, partition_function=pf,
                                  **ph2co_322).value for tex,pf in
                  zip(tem,Q)])

if __name__ == "__main__":
    import pylab as pl

    #del ph2co_303['dnu']
    #del ph2co_321['dnu']
    #del ph2co_322['dnu']

    #print("tau303 = {0}".format(line_tau(**ph2co_303)))
    #print("tau321 = {0}".format(line_tau(**ph2co_321)))
    #print("tau322 = {0}".format(line_tau(**ph2co_322)))
    #print("r303/r321 = {0}".format(line_brightness(**ph2co_321)/line_brightness(**ph2co_303)))
    #print("r303/r322 = {0}".format(line_brightness(**ph2co_322)/line_brightness(**ph2co_303)))


    pl.clf()
    pl.subplot(2,1,1)
    pl.plot(tem, T_321, label='$3_{2,1}-2_{2,0}$')
    pl.plot(tem, T_322, label='$3_{2,2}-2_{2,1}$')
    pl.plot(tem, T_303, label='$3_{0,3}-2_{0,2}$')
    pl.xlim(0,200)
    pl.subplot(2,1,2)
    pl.plot(tem, T_321/T_303, label='321/303')
    pl.plot(tem, T_322/T_303, label='322/303')
    pl.xlim(0,200)

    pl.draw(); pl.show()

    pl.figure()
    pl.plot(T_321/T_303, tem, label='321/303')
