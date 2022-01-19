# Metalibm

Metalibm is available under MIT Licence (see LICENSE file) from https://github.com/metalibm/metalibm/

## INTEGRATION STATUS
master branch: [![pipeline status](https://gitlab.com/nibrunie/metalibm_github/badges/master/pipeline.svg)](https://gitlab.com/nibrunie/metalibm_github/commits/master)

meta-function generation report (master branch): https://nibrunie.gitlab.io/metalibm/perfs/report.main.html

## DOCUMENTATION

Documentation (for master branch) is available on gitlab's pages: https://nibrunie.gitlab.io/metalibm/doc/index.html


## INSTALL

Metalibm is a framework written in Python.


Python version compatibility: as of version 1.0, metalibm works with python3 (tested with 3.8).

- Whatever the install you select, sollya (binary and headers) must be installed on your machine first.
   On a recent version of ubuntu/debian you can install sollya binary and headers by executing:

```sudo apt install sollya libsollya-dev```

- Some features of Metalibm require Gappa (http://gappa.gforge.inria.fr/) (gappa install is recommended)
- To run performance benchmark from metalibm, you will need a compiler setup and papi headers (https://icl.utk.edu/papi/)

```sudo apt install libpapi-dev```



### Quick Install (without intent to modify metalibm sources)
Quick install from git sources (assuming sollya binary and header are installed):
```
pip install git+https://github.com/metalibm/metalibm
```

### Install for Development
For development purpose, this repo should be cloned, and metalibm dependencies installed manually.
Once sollya and gappa have been installed, you can install metalibm's python depdendies by running 

```pip install -r requirements.txt```
    

## USAGE
Before running metalibm, you will need to add its top directory to your PYTHONPATH env var:

```export PYTHONPATH=<path to metalibm/metalibm_core>:$PYTHONPATH```

You will also need to set the `ML_SRC_DIR` env var to point to metalibm support library:

```export ML_SRC_DIR=<path to metalibm/>```

Example of metafunctions can be found under the **metalibm_functions/** directory.

* Example to generate a faithful (default) approximation of the exponential function for single precision on a x86 AVX2 target:
```python3 metalibm_functions/ml_exp.py --precision binary32 --target x86_avx2 --output x86_avx2_exp_fp32.c ```

* Explore the other functions of this directory, e.g. :
``` python3 metalibm_functions/ml_log.py --help  ```

A more comprehensive user documentation can be found in [doc/USERGUIDE.md](https://github.com/metalibm/metalibm/blob/master/doc/USERGUIDE.md)


## TEST
* Unit-testing (software code generation):
  ``` python3 valid/soft_unit_test.py ```
* Unit-testing (hardware code generation):
  ``` python3 valid/rtl_unit_test.py ```

* Non-regression tests (software code generation):
  ``` python3 valid/non_regression.py ```
* Non-regression tests (hardware code generation):
  ``` python3 valid/hw_non_regression.py ```

* Functionnal coverage (generate a report on meta-functions' generation/build/valid status):
  ``` python3 valid/soft_coverage_test.py --report-only --output report.html ```

## DOCUMENTATION

* Metalibm Description Language documentation:  [doc/MDL.md](https://github.com/metalibm/metalibm/blob/master/doc/MDL.md)
* User interface documentation: [doc/USERGUIDE.md](https://github.com/metalibm/metalibm/blob/master/doc/USERGUIDE.md)
* Custom Meta-function documentation: [doc/METAFUNCTION.md](https://github.com/metalibm/metalibm/blob/master/doc/METAFUNCTION.md)
* Metalibm engine optimization pass documentation: [doc/PASSES.md](https://github.com/metalibm/metalibm/blob/master/doc/PASSES.md)
* Metalibm unit testing framework: [doc/UNITTESTS.md](https://github.com/metalibm/metalibm/blob/master/doc/UNITTESTS.md)

## Version History

- Version **1.0.alpha**: Released March 12th, 2018: First alpha for first official version
- Version **1.0.beta**:  Released March 31th, 2018: First beta for first official version



## AUTHOR(S)

    Nicolas Brunie, Hugues de Lassus Saint-Geniès,
    Marc Mezzarobba, Guillaume Gonnachon, Florent de Dinechin, Julien Le Maire,
    Julien Villette, Guillaume Revy, Guillaume Melquiond

    This work has been supported by Kalray (kalrayinc.com) and other entities
    (to be listed)
