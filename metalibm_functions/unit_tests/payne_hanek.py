# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/kalray/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2018 Kalray
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################
# last-modified:    Mar  7th, 2018
# Author(s): Nicolas Brunie <nbrunie@kalray.eu>
###############################################################################
import sys

from sollya import SollyaObject, pi

S2 = SollyaObject(2)

from metalibm_core.core.ml_function import ML_Function, ML_FunctionBasis

from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_complex_formats import *
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_Table
from metalibm_core.core.payne_hanek import generate_payne_hanek

from metalibm_core.code_generation.c_code_generator import CCodeGenerator
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.mpfr_backend import MPFRProcessor


from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator

from metalibm_core.utility.gappa_utils import execute_gappa_script_extract
from metalibm_core.utility.ml_template import *

from metalibm_core.utility.arg_utils import test_flag_option, extract_option_value

from metalibm_core.utility.debug_utils import *

class ML_UT_PayneHanek(ML_Function("ml_ut_payne_hanek")):
  def __init__(self, args):
    #precision = ArgDefault.select_value([arg_template.precision, precision])
    #io_precisions = [precision] * 2

    # initializing base class
    ML_FunctionBasis.__init__(self, args)

  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for current class,
        builtin from a default argument mapping overloaded with @p kw """
    default_args = {
        "output_file": "ut_payne_hanek.c",
        "function_name": "ut_payne_hanek",
        "precision": ML_Binary64,
        "fast_path_extract": True,
        "fuse_fma": True,
        "libm_compliant": True
    }
    default_args.update(kw)
    return DefaultArgTemplate(**default_args)


  def generate_scheme(self):
    int_precision = {ML_Binary32 : ML_Int32, ML_Binary64 : ML_Int64}[self.precision]
    vx = self.implementation.add_input_variable("x", self.precision)
    k = 4
    frac_pi = S2**k/pi
    
    red_stat, red_vx, red_int = generate_payne_hanek(vx, frac_pi, self.precision, k = k, n= 100)
    C32 = Constant(32, precision = int_precision)
    red_int_f = Conversion(Select(red_int < Constant(0, precision = int_precision), red_int + C32, red_int), precision = self.precision)

    red_add = Addition(
      red_vx, 
      red_int_f,
      precision = self.precision
    )

    scheme = Statement(
      red_stat,
      Return(red_add, precision = self.precision)
    )

    return scheme

if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(default_arg=ML_UT_PayneHanek.get_default_args())
  args = arg_template.arg_extraction()


  ml_ut_payne_hanek = ML_UT_PayneHanek(args)

  ml_ut_payne_hanek.gen_implementation(display_after_gen = False, display_after_opt = False)
