# c51_corr_fitter

This fitter requires Peter Lepage's libraries:
- gvar
- lsqfit

both of which can be `pip install`'ed'.

Two example fit_param files are included
- a094m400mL6p0trMc_params.py
- a12m220XL_params.py

The first is from the OpenLat Stabilised Wilson Fermion runs and the latter is from a CalLat project.  Each file indicates the name of the data file to be used in the analysis.

The idea for building up a fit is one first would make effective mass plots to estimate the priors for the ground state values
```
python3 fit_twopt.py a094m400mL6p0trMc_params.py --eff
```
which will create `m_eff` and `z_eff` plots assuming a factorized model of the correlation function
```
C(t) = sum_n zsnk_n zsrc_n exp(-E_n t)
```
There are many desirable features that will be added as we add more complexity to the fit models and more types of correlation functions.

To perform the correlated fit, one runs
```
python3 fit_twopt.py a094m400mL6p0trMc_params.py --eff --fit [--states proton omega ...]
```
where the `states` are specified in the input file but can be overridden in the command line.  A sweep over `t_min` and `n_states` can be performed also
```
python3 fit_twopt.py a094m400mL6p0trMc_params.py --eff --fit --sweep proton pion
```
If the data has a bad condition number in the covariance matrix, one can specify an SVD cut either in the command line or in the input file.



A more extensive readme will be updated as users debug the interface and more features are added.
