# BANDIT (Bayesian ANalysis of Data in Imaginary Time)

This fitter is designed to analyze correlation functions generated from lattice QFT calculations.  This fitting package is currently undergoing rapid development, and there is no promise of backwards compatibility yet.  Version numbers will be used to support reproducibility.

- [Installation](#installation)
- [Example Usage](#example-usage)
- [Input File Description](#input-file-info)
- [Contributors](#contributors)
- [Copyright Notice](#copyright-notice)

## Installation
This package requires the [lsqfit](https://github.com/gplepage/lsqfit) and [gvar](https://github.com/gplepage/gvar) libraries developed by Peter Lepage.

This package is now locally pip-installable in `editable` mode (which means, you can actively develop the code and re-install quickly).  Tested locally with a clean anaconda installation, the following works for installing on a mac OS at least.

### Make a clean environment
skip these two steps unless you want to test a "clean" install - ALSO - they require an [Anaconda Python](https://www.anaconda.com) installation to have the `conda` util.  You can use the package installer - or, they provide bash script installers.  Follow the [install instructions](https://docs.anaconda.com/anaconda/install/mac-os/)
```
conda create -n bare3.8 python=3.8
conda activate bare3.8
```
When you are finished, the environment can be removed
```
conda remove -n bare3.8 --all
```

### Main Code
```
git clone https://github.com/callat-qcd/c51_corr_analysis
pushd c51_corr_analysis
pip install [--user] -e .
popd
```
This will result in an installation that claims there are pip install errors, but in practice, the installation works.  To test success, type
```
which fit_corr
```
and if this returns a binary, the installation has worked.

### Updating
If you want to pull updates, follow the simple 2-steps
```
cd <path_to_repo>/c51_corr_analysis
git pull
pip install -e .
```

### Optional LSQFIT-GUI
One can install the OPTIONAL `lsqfit-gui` utility that is very convenient for estimating priors.
```
cd <some_dir_to_store_src_code>
[optional LSQFIT-GUI interface]
git clone https://github.com/ckoerber/lsqfit-gui
pushd lsqfit-gui
pip install [--user] -e .
popd
```
Ignore the "ERROR" raised by the install - it works, but still throws an error.  We are working to understand how to fix that message.

## Example Usage

To build up a fit, one usually looks at effective mass plots etc., and starts to guess the input ground state energy and overlap factors.  With a working installation, you should be able (from the source directory) do
```
cd c51_corr_analysis/tests
fit_corr input_file/a09m310_test.py --eff --fit
```
which will generate effective mass plots and perform the fit of the states specified in the input file which is included in the `tests/data` directory.

If you have installed `lsqfit-gui`, you can add
```
fit_corr input_file/a09m310_test.py --eff --fit --gui
```


## Input File Info


## Contributors

- André Walker-Loud ([walkloud](https://github.com/walkloud))
- Grant Bradely ([gerbradl](https://github.com/gerbradl))

# Copyright Notice


Bayesian ANalysis of Data in Imaginary Time (BANDIT) Suite
Copyright (c) 2022, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit others to do so.
