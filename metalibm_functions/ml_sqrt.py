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
###############################################################################


import sollya


from metalibm_core.core.ml_operations import (
    Constant,
    FMA, FMSN,
    ML_LeafNode,
    Test, Comparison,
    LogicalOr, LogicalNot,
    ConditionBlock, Return, Statement,
    ReciprocalSquareRootSeed,
)
from metalibm_core.core.attributes import Attributes
from metalibm_core.core.ml_formats import (
    ML_Binary32, ML_Binary64,
    ML_Bool,
    ML_RoundToNearest, ML_GlobalRoundMode
)
from metalibm_core.core.ml_function import (
    ML_FunctionBasis, DefaultArgTemplate
)
from metalibm_core.core.special_values import (
    FP_PlusZero, FP_MinusZero, FP_QNaN, FP_PlusInfty
)
from metalibm_core.core.precisions import ML_Faithful
from metalibm_core.core.simple_scalar_function import ScalarUnaryFunction

from metalibm_core.code_generation.generic_processor import GenericProcessor

from metalibm_core.utility.ml_template import ML_NewArgTemplate
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import debug_multi

## Newton-Raphson iteration object
class NR_Iteration:
  def __init__(self, value, approx, half_value, c_half):
    Attributes.set_default_rounding_mode(ML_RoundToNearest)
    Attributes.set_default_silent(True)

    self.square = approx * approx
    mult = FMSN(half_value, self.square, c_half)
    self.new_approx =  FMA(approx, mult, approx)

    Attributes.unset_default_rounding_mode()
    Attributes.unset_default_silent()


  def get_new_approx(self):
    return self.new_approx

# TODO; should be factorized with Metalibm's Core process
## propagate @p precision on @p optree on all operands with
#  no precision (None) set, applied recursively
def propagate_format(optree, precision):
  if optree.get_precision() is None:
    optree.set_precision(precision)
    if not isinstance(optree, ML_LeafNode):
      for op_input in optree.get_inputs():
        propagate_format(op_input, precision)


def compute_sqrt(vx, init_approx, num_iter, debug_lftolx = None, precision = ML_Binary64):

    C_half = Constant(0.5, precision = precision)
    h = C_half * vx
    h.set_attributes(tag = "h", debug = debug_multi, silent = True, rounding_mode = ML_RoundToNearest)

    current_approx = init_approx
    # correctly-rounded inverse computation

    for i in range(num_iter):
        new_iteration = NR_Iteration(vx, current_approx, h, C_half)
        current_approx = new_iteration.get_new_approx()
        current_approx.set_attributes(tag = "iter_%d" % i, debug = debug_multi)

    final_approx = current_approx
    final_approx.set_attributes(tag = "final_approx", debug = debug_multi)

    # multiplication correction iteration
    # to get correctly rounded full square root
    Attributes.set_default_silent(True)
    Attributes.set_default_rounding_mode(ML_RoundToNearest)

    S = vx * final_approx
    t5 = final_approx * h
    H = C_half * final_approx
    d = FMSN(S, S, vx)
    t6 = FMSN(t5, final_approx, C_half)
    S1 = FMA(d, H, S)
    H1 = FMA(t6, H, H)
    d1 = FMSN(S1, S1, vx)
    pR = FMA(d1, H1, S1)
    d_last = FMSN(pR, pR, vx, silent = True, tag = "d_last")

    S.set_attributes(tag = "S")
    t5.set_attributes(tag = "t5")
    H.set_attributes(tag = "H")
    d.set_attributes(tag = "d")
    t6.set_attributes(tag = "t6")
    S1.set_attributes(tag = "S1")
    H1.set_attributes(tag = "H1")
    d1.set_attributes(tag = "d1")


    Attributes.unset_default_silent()
    Attributes.unset_default_rounding_mode()

    R = FMA(d_last, H1, pR, rounding_mode = ML_GlobalRoundMode, tag = "NR_Result", debug = debug_multi)

    # set precision
    propagate_format(R, precision)
    propagate_format(S1, precision)
    propagate_format(H1, precision)
    propagate_format(d1, precision)

    return R



class MetalibmSqrt(ScalarUnaryFunction):
    function_name = "ml_sqrt"

    def __init__(self, args=DefaultArgTemplate):
        # initializing base class
        super().__init__(args)
        self.num_iter = args.num_iter


    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for MetalibmSqrt,
            builtin from a default argument mapping overloaded with @p kw """
        default_args_sqrt = {
            "output_file": "my_sqrtf.c",
            "function_name": "my_sqrtf",
            "num_iter": 3, 
            "precision": ML_Binary32,
            "accuracy": ML_Faithful,
            "target": GenericProcessor.get_target_instance()
        }
        default_args_sqrt.update(kw)
        return DefaultArgTemplate(**default_args_sqrt)

    def generate_scalar_scheme(self, vx):
        vx.set_attributes(precision = self.precision, tag = "vx", debug =debug_multi)
        Log.set_dump_stdout(True)

        Log.report(Log.Info, "\033[33;1m Generating implementation scheme \033[0m")
        if self.debug_flag:
            Log.report(Log.Info, "\033[31;1m debug has been enabled \033[0;m")

        C0 = Constant(0, precision=self.precision)

        C0_plus = Constant(FP_PlusZero(self.precision))
        C0_minus = Constant(FP_MinusZero(self.precision))

        def local_test(specifier, tag):
            """ Local wrapper to generate Test operations """
            return Test(
                vx, specifier=specifier, likely=False, debug=debug_multi,
                tag="is_%s"%tag, precision=ML_Bool
            )

        test_NaN = local_test(Test.IsNaN,"is_NaN")
        test_inf = local_test(Test.IsInfty,"is_Inf")
        test_NaN_or_Inf = local_test(Test.IsInfOrNaN,"is_Inf_Or_Nan")

        test_negative = Comparison(
            vx, C0, specifier=Comparison.Less, debug=debug_multi,
            tag="is_Negative", precision=ML_Bool, likely=False)
        test_NaN_or_Neg = LogicalOr(test_NaN, test_negative, precision=ML_Bool)

        test_std = LogicalNot(
            LogicalOr(
                test_NaN_or_Inf, test_negative, precision=ML_Bool, likely=False
            ),
            precision=ML_Bool, likely=True
        )

        test_zero = Comparison(
            vx, C0, specifier=Comparison.Equal, likely=False, debug=debug_multi,
            tag="Is_Zero", precision=ML_Bool)

        return_NaN_or_neg = Statement(Return(FP_QNaN(self.precision)))
        return_inf = Statement(Return(FP_PlusInfty(self.precision)))

        return_PosZero = Return(C0_plus)
        return_NegZero = Return(C0_minus)

        NR_init = ReciprocalSquareRootSeed(
            vx, precision=self.precision, tag="sqrt_seed",
            debug = debug_multi)

        result = compute_sqrt(vx, NR_init, int(self.num_iter), precision=self.precision)

        return_non_std = ConditionBlock(
            test_NaN_or_Neg,
            return_NaN_or_neg,
            ConditionBlock(
              test_inf,
              return_inf,
              ConditionBlock(
                test_zero,
                return_PosZero,
                return_NegZero
            )
            )
        )
        return_std = Return(result)

        scheme = ConditionBlock(
          test_std,
          return_std,
          return_non_std
        )
        return scheme


    def numeric_emulate(self, vx):
        """ Numeric emulation for square-root """
        return sollya.sqrt(vx)


    standard_test_cases = [(1.651028399744791652636877188342623412609100341796875,)] # [sollya.parse(x)] for x in  ["+0.0", "-1*0.0", "2.0"]]

if __name__ == "__main__":
    ARG_TEMPLATE = ML_NewArgTemplate(default_arg=MetalibmSqrt.get_default_args())
    # TODO: should not be a command line argument but rather determined during generation
    ARG_TEMPLATE.parser.add_argument(
      "--num-iter", dest="num_iter", action="store", default=3,
      help="number of Newton-Raphson iterations")

    ARGS = ARG_TEMPLATE.arg_extraction()

    ML_SQRT  = MetalibmSqrt(ARGS)
    ML_SQRT.gen_implementation()

