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
# Created:          Dec 16th, 2015
# last-modified:    Mar  7th, 2018
# Author(s): Nicolas Brunie <nbrunie@kalray.eu>
# Description: fast and low accuracy sine and cosine implementation
###############################################################################

import sollya

from sollya import (
    S2, Interval, ceil, floor, round, inf, sup, log2, cos, sin,
    guessdegree, dirtyinfnorm
)

from metalibm_core.core.ml_function import ML_FunctionBasis

from metalibm_core.core.ml_operations import (
    Constant, BitLogicRightShift, BitLogicAnd,
    Multiplication, Addition, Subtraction,
    TypeCast, TableLoad, Conversion,
    Return, Statement)


from metalibm_core.core.ml_formats import  (
    ML_Custom_FixedPoint_Format, ML_Binary32, ML_Int32)
from metalibm_core.core.polynomials import Polynomial
from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.core.precisions import ML_Faithful
from metalibm_core.targets.common.fixed_point_backend import FixedPointBackend

from metalibm_core.core.payne_hanek import generate_payne_hanek

from metalibm_core.utility.ml_template import DefaultArgTemplate, ML_NewArgTemplate
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import debug_fixed32, debugd

# set sollya verbosity level to 0
sollya.settings.verbosity(0)

## Fast implementation of trigonometric function sine and cosine
#  Focuses on speed rather than on accuracy. Accepts --accuracy
#  and --input-interval options
class ML_FastSinCos(ML_FunctionBasis):
    """ Implementation of cosinus function """
    function_name = "ml_fast_cos"
    def __init__(self, args=DefaultArgTemplate):
        super().__init__(args)
        self.cos_output = args.cos_output
        self.table_size_log = args.table_size_log

    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for ML_SinCos,
            builtin from a default argument mapping overloaded with @p kw """
        default_args_sincos = {
            "output_file": "my_sincos.c",
            "function_name": "new_fastsincos",
            "precision": ML_Binary32,
            "accuracy": ML_Faithful,
            "target": FixedPointBackend.get_target_instance(),
            "cos_output": True,
        }
        default_args_sincos.update(kw)
        return DefaultArgTemplate(**default_args_sincos)


    def generate_scheme(self):
        # declaring CodeFunction and retrieving input variable
        vx = self.implementation.add_input_variable("x", self.precision)

        table_size_log = self.table_size_log
        integer_size = 31
        integer_precision = ML_Int32

        max_bound = sup(abs(self.input_intervals[0]))
        max_bound_log = int(ceil(log2(max_bound)))
        Log.report(Log.Info, "max_bound_log=%s " % max_bound_log)
        scaling_power = integer_size - max_bound_log
        Log.report(Log.Info, "scaling power: %s " % scaling_power)

        storage_precision = ML_Custom_FixedPoint_Format(1, 30, signed = True)

        Log.report(Log.Info, "tabulating cosine and sine")
        # cosine and sine fused table
        fused_table = ML_NewTable(dimensions = [2**table_size_log, 2], storage_precision = storage_precision, tag = "fast_lib_shared_table") # self.uniquify_name("cossin_table"))
        # filling table
        for i in range(2**table_size_log):
          local_x = i / S2**table_size_log * S2**max_bound_log

          cos_local = cos(local_x) # nearestint(cos(local_x) * S2**storage_precision.get_frac_size())

          sin_local = sin(local_x) # nearestint(sin(local_x) * S2**storage_precision.get_frac_size())

          fused_table[i][0] = cos_local
          fused_table[i][1] = sin_local

        # argument reduction evaluation scheme
        # scaling_factor = Constant(S2**scaling_power, precision = self.precision)

        red_vx_precision = ML_Custom_FixedPoint_Format(31 - scaling_power, scaling_power, signed = True)
        Log.report(Log.Verbose, "red_vx_precision.get_c_bit_size()=%d" % red_vx_precision.get_c_bit_size())
        # red_vx = NearestInteger(vx * scaling_factor, precision = integer_precision)
        red_vx = Conversion(vx, precision = red_vx_precision, tag = "red_vx", debug = debug_fixed32)

        computation_precision = red_vx_precision # self.precision
        output_precision      = self.get_output_precision()
        Log.report(Log.Info, "computation_precision is %s" % computation_precision)
        Log.report(Log.Info, "storage_precision     is %s" % storage_precision)
        Log.report(Log.Info, "output_precision      is %s" % output_precision)

        hi_mask_value = 2**32 - 2**(32-table_size_log - 1)
        hi_mask = Constant(hi_mask_value, precision = ML_Int32)
        Log.report(Log.Info, "hi_mask=0x%x" % hi_mask_value)

        red_vx_hi_int = BitLogicAnd(TypeCast(red_vx, precision = ML_Int32), hi_mask, precision = ML_Int32, tag = "red_vx_hi_int", debug = debugd)
        red_vx_hi = TypeCast(red_vx_hi_int, precision = red_vx_precision, tag = "red_vx_hi", debug = debug_fixed32)
        red_vx_lo = red_vx - red_vx_hi
        red_vx_lo.set_attributes(precision = red_vx_precision, tag = "red_vx_lo", debug = debug_fixed32)
        table_index = BitLogicRightShift(TypeCast(red_vx, precision = ML_Int32), scaling_power - (table_size_log - max_bound_log), precision = ML_Int32, tag = "table_index", debug = debugd)

        tabulated_cos = TableLoad(fused_table, table_index, 0, tag = "tab_cos", precision = storage_precision, debug = debug_fixed32)
        tabulated_sin = TableLoad(fused_table, table_index, 1, tag = "tab_sin", precision = storage_precision, debug = debug_fixed32)

        error_function = lambda p, f, ai, mod, t: dirtyinfnorm(f - p, ai)

        Log.report(Log.Info, "building polynomial approximation for cosine")
        # cosine polynomial approximation
        poly_interval = Interval(0, S2**(max_bound_log - table_size_log))
        Log.report(Log.Info, "poly_interval=%s " % poly_interval)
        cos_poly_degree = 2 # int(sup(guessdegree(cos(x), poly_interval, accuracy_goal)))

        Log.report(Log.Verbose, "cosine polynomial approximation")
        cos_poly_object, cos_approx_error = Polynomial.build_from_approximation_with_error(cos(sollya.x), [0, 2] , [0] + [computation_precision.get_bit_size()], poly_interval, sollya.absolute, error_function = error_function)
        #cos_eval_scheme = PolynomialSchemeEvaluator.generate_horner_scheme(cos_poly_object, red_vx_lo, unified_precision = computation_precision)
        Log.report(Log.Info, "cos_approx_error=%e" % cos_approx_error)
        cos_coeff_list = cos_poly_object.get_ordered_coeff_list()
        coeff_C0 = cos_coeff_list[0][1]
        coeff_C2 = Constant(cos_coeff_list[1][1], precision = ML_Custom_FixedPoint_Format(-1, 32, signed = True))

        Log.report(Log.Info, "building polynomial approximation for sine")

        # sine polynomial approximation
        sin_poly_degree = 2 # int(sup(guessdegree(sin(x)/x, poly_interval, accuracy_goal)))
        Log.report(Log.Info, "sine poly degree: %e" % sin_poly_degree)
        Log.report(Log.Verbose, "sine polynomial approximation")
        sin_poly_object, sin_approx_error = Polynomial.build_from_approximation_with_error(sin(sollya.x)/sollya.x, [0, 2], [0] + [computation_precision.get_bit_size()] * (sin_poly_degree+1), poly_interval, sollya.absolute, error_function = error_function)
        sin_coeff_list = sin_poly_object.get_ordered_coeff_list()
        coeff_S0 = sin_coeff_list[0][1]
        coeff_S2 = Constant(sin_coeff_list[1][1], precision = ML_Custom_FixedPoint_Format(-1, 32, signed = True))

        # scheme selection between sine and cosine
        if self.cos_output:
          scheme = self.generate_cos_scheme(computation_precision, tabulated_cos, tabulated_sin, coeff_S2, coeff_C2, red_vx_lo)
        else:
          scheme = self.generate_sin_scheme(computation_precision, tabulated_cos, tabulated_sin, coeff_S2, coeff_C2, red_vx_lo)

        result = Conversion(scheme, precision=self.get_output_precision())

        Log.report(Log.Verbose, "result operation tree :\n %s " % result.get_str(display_precision = True, depth = None, memoization_map = {}))
        scheme = Statement(
          Return(result)
        )

        return scheme


    ## generate scheme for cosine approximation of cos(X = x + u)
    #  @param computation_precision ML_Format used as default precision for scheme evaluation
    #  @param tabulated_cos tabulated value of cosine(high part of vx)
    #  @param tabulated_sin tabulated value of   sine(high part of vx)
    #  @param sin_C2 polynomial coefficient of sine approximation for u^3 
    #  @param cos_C2 polynomial coefficient of cosine approximation for u^2
    #  @param red_vx_lo low part of the reduced input variable (i.e. u)
    def generate_cos_scheme(self, computation_precision, tabulated_cos, tabulated_sin, sin_C2, cos_C2, red_vx_lo):
        cos_C2 = Multiplication(
                  tabulated_cos,
                  cos_C2,
                  precision = ML_Custom_FixedPoint_Format(-1, 32, signed = True),
                  tag = "cos_C2"
                )
        u2 = Multiplication(
              red_vx_lo,
              red_vx_lo,
              precision = computation_precision, # ML_Custom_FixedPoint_Format(5, 26, signed = True)
              tag = "u2"
            )
        sin_u = Multiplication(
                  tabulated_sin,
                  red_vx_lo,
                  precision = computation_precision, # ML_Custom_FixedPoint_Format(1, 30, signed = True)
                  tag = "sin_u"
                )

        cos_C2_u2 = Multiplication(
                      cos_C2,
                      u2,
                      precision = computation_precision, # ML_Custom_FixedPoint_Format(1, 30,signed = True)
                      tag = "cos_C2_u2"
                    )

        S2_u2 = Multiplication(
                  sin_C2, 
                  u2,
                  precision = ML_Custom_FixedPoint_Format(-1, 32, signed = True),
                  tag = "S2_u2"
                )

        S2_u3_sin = Multiplication(
                      S2_u2,
                      sin_u,
                      precision = computation_precision, # ML_Custom_FixedPoint_Format(5,26, signed = True)
                      tag = "S2_u3_sin"
                    )

        cos_C2_u2_P_cos = Addition(
                            tabulated_cos,
                            cos_C2_u2,
                            precision = computation_precision, # ML_Custom_FixedPoint_Format(5, 26, signed = True)
                            tag = "cos_C2_u2_P_cos"
                          )

        cos_C2_u2_P_cos_M_sin_u = Subtraction(
                                    cos_C2_u2_P_cos,
                                    sin_u,
                                    precision = computation_precision # ML_Custom_FixedPoint_Format(5, 26, signed = True)
                                  )

        scheme = Subtraction(
                    cos_C2_u2_P_cos_M_sin_u,
                    S2_u3_sin,
                    precision = computation_precision # ML_Custom_FixedPoint_Format(5, 26, signed = True)
                  )

        return scheme


    ## generate scheme for sine approximation of sin(X = x + u)
    #  @param computation_precision ML_Format used as default precision for scheme evaluation
    #  @param tabulated_cos tabulated value of cosine(high part of vx)
    #  @param tabulated_sin tabulated value of   sine(high part of vx)
    #  @param sin_C2 polynomial coefficient of sine approximation for u^3 
    #  @param cos_C2 polynomial coefficient of cosine approximation for u^2
    #  @param red_vx_lo low part of the reduced input variable (i.e. u)
    def generate_sin_scheme(self, computation_precision, tabulated_cos, tabulated_sin, coeff_S2, coeff_C2, red_vx_lo):
        sin_C2 = Multiplication(
                  tabulated_sin,
                  coeff_C2,
                  precision = ML_Custom_FixedPoint_Format(-1, 32, signed = True),
                  tag = "sin_C2"
                )
        u2 = Multiplication(
              red_vx_lo,
              red_vx_lo,
              precision = computation_precision, # ML_Custom_FixedPoint_Format(5, 26, signed = True)
              tag = "u2"
            )
        cos_u = Multiplication(
                  tabulated_cos,
                  red_vx_lo,
                  precision = computation_precision, # ML_Custom_FixedPoint_Format(1, 30, signed = True)
                  tag = "cos_u"
                )

        S2_u2 = Multiplication(
                      coeff_S2,
                      u2,
                      precision = ML_Custom_FixedPoint_Format(-1, 32,signed = True),
                      tag = "S2_u2"
                    )

        sin_C2_u2 = Multiplication(
                  sin_C2, 
                  u2,
                  precision = computation_precision,
                  tag = "sin_C2_u2"
                )

        S2_u3_cos = Multiplication(
                      S2_u2,
                      cos_u,
                      precision = computation_precision, # ML_Custom_FixedPoint_Format(5,26, signed = True)
                      tag = "S2_u3_cos"
                    )

        sin_P_cos_u = Addition(
                            tabulated_sin,
                            cos_u,
                            precision = computation_precision, # ML_Custom_FixedPoint_Format(5, 26, signed = True)
                            tag = "sin_P_cos_u"
                          )

        sin_P_cos_u_P_C2_u2_sin = Addition(
                                    sin_P_cos_u,
                                    sin_C2_u2,
                                    precision = computation_precision, # ML_Custom_FixedPoint_Format(5, 26, signed = True)
                                    tag = "sin_P_cos_u_P_C2_u2_sin"
                                  )

        scheme = Addition(
                    sin_P_cos_u_P_C2_u2_sin,
                    S2_u3_cos,
                    precision = computation_precision # ML_Custom_FixedPoint_Format(5, 26, signed = True)
                  )

        return scheme

    def numeric_emulate(self, input_value):
        if self.cos_output:
            return cos(input_value)
        else:
            return sin(input_value)

if __name__ == "__main__":
    arg_template = ML_NewArgTemplate(default_arg=ML_FastSinCos.get_default_args())
    # argument extraction
    arg_template.get_parser().add_argument(
        "--sin", dest="cos_output", default=True, const=False,
        action="store_const", help="select sine output (default is cosine)")
    arg_template.get_parser().add_argument(
        "--table-size-log", dest="table_size_log", default=8, type=int,
        action="store", help="logarithm of the table size to be used")

    args = arg_template.arg_extraction()
    ml_fast_sincos = ML_FastSinCos(args)
    ml_fast_sincos.gen_implementation()

