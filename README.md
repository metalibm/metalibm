# Metalibm

Metalibm is available under MIT Licence (see LICENSE file) from https://github.com/kalray/metalibm/

## INTEGRATION STATUS
master branch: [![pipeline status](https://gitlab.com/nibrunie/metalibm_github/badges/master/pipeline.svg)](https://gitlab.com/nibrunie/metalibm_github/commits/master)

meta-function generation report (master branch): https://nibrunie.gitlab.io/metalibm/perfs/report.master.html

## DOCUMENTATION

Documentation (for master branch) is available on gitlab's pages: https://nibrunie.gitlab.io/metalibm/doc/index.html


## INSTALL

* Dependencies:
    - metalibm depends on master branch Pythonsollya (python wrapper to Sollya library).
        easy install (if sollya is already installed): `pip install git+https://gitlab.com/metalibm-dev/pythonsollya`
        or Pythonsollya can be downloaded from https://gitlab.com/metalibm-dev/pythonsollya
        beware that sollya master branch is required for pythonsollya master
    - Some features of Metalibm require Gappa (http://gappa.gforge.inria.fr/)

* Python version compatibility: as of version 1.0, metalibm works with python3 (tested with 3.4). A legacy support for python2 (>= 2.7) is maintained but will be dropped soon.


* Installation procedure for standard users
    - install pythonsollya (and optionally gappa)
    - make sure pythonsollya is available in your PYTHONPATH
    - make sure metalibm's top directory is in your PYTHONPATH

## USAGE
Example of metafunctions can be found under the **metalibm_functions/** directory.

* Example to generate a faithful (default) approximation of the exponential function for single precision on a x86 AVX2 target:
```python3 metalibm_functions/ml_exp.py --precision binary32 --target x86_avx2 --output x86_avx2_exp2d.c ```

* Explore the other functions of this directory, e.g. :
``` python3 metalibm_functions/ml_log.py --help  ```

A more comprehensive user documentation can be found in [doc/USERGUIDE.md](https://github.com/kalray/metalibm/blob/master/doc/USERGUIDE.md)


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

* Metalibm Description Language documentation:  [doc/MDL.md](https://github.com/kalray/metalibm/blob/master/doc/MDL.md)
* User interface documentation: [doc/USERGUIDE.md](https://github.com/kalray/metalibm/blob/master/doc/USERGUIDE.md)
* Custom Meta-function documentation: [doc/METAFUNCTION.md](https://github.com/kalray/metalibm/blob/master/doc/METAFUNCTION.md)
* Metalibm engine optimization pass documentation: [doc/PASSES.md](https://github.com/kalray/metalibm/blob/master/doc/PASSES.md)
* Metalibm unit testing framework: [doc/UNITTESTS.md](https://github.com/kalray/metalibm/blob/master/doc/UNITTESTS.md)

## Version History

- Version **1.0.alpha**: Released March 12th, 2018: First alpha for first official version
- Version **1.0.beta**:  Released March 31th, 2018: First beta for first official version



## AUTHOR(S)

    Nicolas Brunie (nbrunie (AT) kalray.eu), Hugues de Lassus Saint-Geniès,
    Marc Mezzarobba, Guillaume Gonnachon, Florent de Dinechin, Julien Le Maire,
    Julien Villette, Guillaume Revy

    This work has been supported by Kalray (kalrayinc.com) and other entities
    (to be listed)
