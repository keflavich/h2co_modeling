"""
Given physical parameters and a background, compute the expected line
brightness
"""

def brightness(background, tex, tau):
    """
    Given a background brightness in Kelvin, compute radiative transfer...
    """

    bgterm = background * np.exp(-tau) 
    fgterm = (1-np.exp(-tau))*tex

    return bgterm + fgterm
