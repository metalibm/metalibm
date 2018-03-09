# Metalibm

Metalibm is available under MIT Licence (see LICENSE file) from https://github.com/kalray/metalibm/

## INTEGRATION STATUS

[![pipeline status](https://gitlab.com/nibrunie/metalibm_github/badges/dev/pipeline.svg)](https://gitlab.com/nibrunie/metalibm_github/commits/dev)

## INSTALL


* Dependencies:
    - metalibm depends on Pythonsollya (python wrapper to Sollya library). Pythonsollya can be downloaded from https://gitlab.com/metalibm-dev/pythonsollya
    - Some features of Metalibm relies on Gappa (http://gappa.gforge.inria.fr/)

* Python version compatibility: as of version 1.0, metalibm works with both python2 (>= 2.7) and python3 (tested with 3.4)


* Installation procedure for standard users
    - install pythonsollya (and optionally gappa)
    - make sure pythonsollya is available in your PYTHONPATH
    - make sure metalibm's top directory is in your PYTHONPATH

## USAGE
Example of metafunctions can be found under the metalibm_functions directory.

* Example to generate a faithful (default) approximation of the exponential function for single precision and a x86 AVX2 target:
```python2 metalibm_functions/ml_exp.py --precision binary32 --target x86_avx2 --output x86_avx2_exp2d.c ```

* Explore the other functions of this directory, e.g. :
``` python2 metalibm_functions/ml_log.py --help  ```


## TEST
* Unit-testing (software code generation):
  ``` python2 valid/soft_unit_test.py ```
* Unit-testing (hardware code generation):
  ``` python2 valid/rtl_unit_test.py ```

* Non-regression tests (software code generation):
  ``` python2 valid/non_regression.py ```
* Non-regression tests (hardware code generation):
  ``` python2 valid/hw_non_regression.py ```

## Version History

- Version **1.0.alpha**: Released March 12th, 2018: First official version



## AUTHOR(S)

    Nicolas Brunie (nbrunie (AT) kalray.eu), Hugues de Lassus Saint-Geniès,
    Marc Mezzaroba, Guillaume Gonnachon, Florent de Dinechin, Julien Le Maire,
    Julien Villette, Guillaume Revy

    This work was supported by Kalray (kalrayinc.com) and other entities
    (to be listed)
