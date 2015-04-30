A complete workflow for modeling & fitting H2CO mm data (the J=3-2 or J=4-3 lines)

 1. Install `pyradex <https://github.com/keflavich/pyradex>`_
 2. Install `this package <https://github.com/keflavich/h2co_modeling>`_
 3. Create a grid of RADEX models using the `pyradex_h2comm_grid <examples/pyradex_h2comm_grid.py>`_ code 
    in this package.  
 4. Load the appropriate `constrain_parameters script (J=4-3) <https://github.com/keflavich/APEX_CMZ_H2CO/blob/master/analysis/constrain_parameters_4to3.py>`_
    or `(J=3-2) <https://github.com/keflavich/APEX_CMZ_H2CO/blob/master/analysis/constrain_parameters.py>`_.
    In the future these may be replaced with a more general script to be hosted in this repository, but for now we just have two sets of models
 5. Create a ``paraH2COmodel`` object with the grids from above loaded (you might have to hard-code in the paths)
 6. Set the observational constraints (e.g., the line brightness, the line
    ratio, the density, abundance, line width) using the `set_constraints
    <https://github.com/keflavich/APEX_CMZ_H2CO/blob/master/analysis/constrain_parameters_4to3.py#L212>`_
    method.
 7. Derive the parameter constraints (on temperature, density, H2CO column)
    from the `get_parconstraints
    <https://github.com/keflavich/APEX_CMZ_H2CO/blob/master/analysis/constrain_parameters_4to3.py#L321>`_
    method
 8. (optional) Plot the relevant parameter spaces using the `parplot
    <https://github.com/keflavich/APEX_CMZ_H2CO/blob/master/analysis/constrain_parameters_4to3.py#L350>`_
    method
