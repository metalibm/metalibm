# -*- coding: utf-8 -*-

import sys

from pythonsollya import *

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.code_generation.c_code_generator import CCodeGenerator
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.code_object import CodeObject
from metalibm_core.code_generation.code_element import CodeFunction
from metalibm_core.code_generation.code_constant import C_Code 
from metalibm_core.core.ml_optimization_engine import OptimizationEngine
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_Table

from metalibm_core.targets.kalray.k1a_processor import K1A_Processor
from metalibm_core.targets.kalray.k1b_processor import K1B_Processor

from metalibm_core.code_generation.gappa_code_generator import GappaCodeGenerator

from metalibm_core.utility.gappa_utils import execute_gappa_script_extract
from metalibm_core.utility.log_report import Log
from metalibm_core.utility.common import test_flag_option, extract_option_value  

from metalibm_core.utility.ml_template import ML_ArgTemplate


debugf = ML_Debug(display_format = "%f")
debuglf = ML_Debug(display_format = "%lf")
debugx = ML_Debug(display_format = "%x")
debuglx = ML_Debug(display_format = "%lx")
debugd = ML_Debug(display_format = "%d")
debug_ftox_fp64 = ML_Debug(display_format = "%\"PRIx64\"", pre_process = lambda v: "double_to_64b_encoding(%s)" % v)
debug_ftox_fp32 = ML_Debug(display_format = "%\"PRIx32\"", pre_process = lambda v: "float_to_32b_encoding(%s)" % v)


class NR_Iteration: 
    def __init__(self, value, approx, half_value):
        Attributes.set_default_rounding_mode(ML_RoundToNearest)
        Attributes.set_default_silent(True)

        self.square = approx * approx
        error_mult = self.square * half_value
        self.error = 0.5 - error_mult
        approx_mult = self.error * approx
        self.new_approx = approx + approx_mult 

        Attributes.unset_default_rounding_mode()
        Attributes.unset_default_silent()


    def get_new_approx(self):
        return self.new_approx

    def get_hint_rules(self, gcg, gappa_code, exact):
        pass

def compute_sqrt(vx, init_approx, debug_lftolx, precision = ML_Binary64):
    debug_lftolx = debug_ftox_fp64 if precision == ML_Binary64 else debug_ftox_fp32
    debug_dec = debugf if precision == ML_Binary32 else debuglf

    h = 0.5 * vx
    h.set_attributes(tag = "h", debug = debug_lftolx, silent = True, rounding_mode = ML_RoundToNearest)

    current_approx = init_approx 
    # correctly-rounded inverse computation
    num_iteration = num_iter
    inv_iteration_list = []
    for i in xrange(num_iteration):
        new_iteration = NR_Iteration(vx, current_approx, h)
        inv_iteration_list.append(new_iteration)
        current_approx = new_iteration.get_new_approx()
        current_approx.set_attributes(tag = "iter_%d" % i, debug = debug_dec)

    final_approx = current_approx
    final_approx.set_attributes(tag = "final_approx", debug = debug_lftolx)

    # multiplication correction iteration
    # to get correctly rounded full square root
    Attributes.set_default_silent(True)
    Attributes.set_default_rounding_mode(ML_RoundToNearest)
    S = vx * final_approx
    S.set_attributes(tag = "S", debug = debug_lftolx)
    t5 = final_approx * h
    t5.set_attributes(tag = "t5", debug = debug_lftolx)
    H = 0.5 * final_approx
    H.set_attributes(tag = "H", debug = debug_lftolx)
    d = vx - S * S
    #d = FMSN(S, S, vx)
    d.set_attributes(tag = "d", debug = debug_lftolx)
    t6 = 0.5 - t5 * final_approx
    t6.set_attributes(tag = "t6", debug = debug_lftolx)
    S1 = S + d * H
    S1.set_attributes(tag = "S1", debug = debug_lftolx)
    H1 = H + t6 * H
    H1.set_attributes(tag = "H1", debug = debug_lftolx)
    #d1 = vx - S1 * S1
    d1 = FMSN(S1, S1, vx) #, clearprevious = True)
    d1.set_attributes(tag = "d1", debug = debug_lftolx)

    pR = FMA(d1, H1, S1)

    #t7 = 0.5 - t6 * final_approx
    #H2 = H1 + t6 * H1
    d_last = FMSN(pR, pR, vx, silent = True, tag = "d_last", debug = debug_lftolx)


    Attributes.unset_default_silent()
    Attributes.unset_default_rounding_mode()
    #R  = S1 + d1 * H1 

    #R = FMA(d1, H1, S1, rounding_mode = ML_GlobalRoundMode)
    R = FMA(d_last, H1, pR, rounding_mode = ML_GlobalRoundMode)
    return R


class ML_Sqrt:
    def __init__(self, 
                 precision = ML_Binary32, 
                 abs_accuracy = S2**-24, 
                 libm_compliant = True, 
                 debug_flag = False, 
                 fuse_fma = True, 
                 num_iter = 3,
                 fast_path_extract = True,
                 target = GenericProcessor(), 
                 output_file = "sqrtf.c", 
                 function_name = "sqrtf",
                 dot_product_enabled = False):
        # declaring target and instantiating OptimizationEngine
        processor = target
        self.dot_product_enabled = dot_product_enabled
        opt_eng = OptimizationEngine(processor, dot_product_enabled = self.dot_product_enabled)

        # declaring CodeFunction and retrieving input variable
        self.precision = precision
        self.function_name = function_name
        exp_implementation = CodeFunction(self.function_name, output_format = precision)
        pre_vx = exp_implementation.add_input_variable("x", precision) 

        debug_lftolx = debug_ftox_fp64 if self.precision == ML_Binary64 else debug_ftox_fp32

        # local overloading of RaiseReturn operation
        def SqrtRaiseReturn(*args, **kwords):
            kwords["arg_value"] = pre_vx
            kwords["function_name"] = self.function_name
            return RaiseReturn(*args, **kwords)



        debug_dec = debugf if self.precision == ML_Binary32 else debuglf
        ex = ExponentExtraction(pre_vx, tag = "ex", debug = debugd)
        equal_comp = Equal(Modulo(ex, 2), 0)
        even_ex = Select(equal_comp, ex, ex - 1, tag = "even_ex", debug = debugd)
        pre_scale_factor = ExponentInsertion(-(even_ex/2), tag = "pre_scale_factor", debug = debug_lftolx) 
        pre_scale_mult = (pre_vx * pre_scale_factor)
        pre_scale_mult.set_attributes(silent = True, rounding_mode = ML_RoundToNearest)
        slow_vx = pre_scale_mult * pre_scale_factor
        slow_vx.set_attributes(tag = "vx", silent = True, rounding_mode = ML_RoundToNearest, debug = debug_lftolx)
        slow_scale_factor = ExponentInsertion(even_ex / 2, tag = "scale_factor", debug = debug_lftolx)

        if isinstance(processor, K1B_Processor):
            vx = pre_vx
            scale_factor = 1.0
        else:
            vx = slow_vx
            scale_factor = slow_scale_factor

        # computing the inverse square root
        init_approx = None
        # forcing vx precision to make processor support test
        vx.set_precision(self.precision)
        init_approx_precision = InverseSquareRootSeed(vx, precision = self.precision, tag = "seed", debug = debugf)
        if not processor.is_supported_operation(init_approx_precision):
            if self.precision != ML_Binary32:
                px = Conversion(vx, precision = ML_Binary32, tag = "px", debug=debugf) 
                init_approx_fp32 = Conversion(InverseSquareRootSeed(px, precision = ML_Binary32, tag = "seed_fp32", debug = debugf), precision = self.precision, tag = "seed_ext", debug = debug_lftolx)
                if not processor.is_supported_operation(init_approx_fp32):
                    Log.report(Log.Error, "The target %s does not implement inverse square root seed" % processor)
                else:
                    init_approx = init_approx_fp32
            else:
                Log.report(Log.Error, "The target %s does not implement inverse square root seed" % processor)
        else:
            init_approx = init_approx_precision




        if isinstance(processor, K1B_Processor):
            result = compute_sqrt(vx, init_approx, ML_Binary64)
            slow_result = compute_sqrt(slow_vx, InverseSquareRootSeed(slow_vx, precision = self.precision, tag = "slow_seed", debug = debugf), ML_Binary64) * slow_scale_factor
        else:
            result = compute_sqrt(vx, init_approx, ML_Binary64) * scale_factor
        result.set_attributes(tag = "result", debug = debug_lftolx, clearprevious = True)

        #x_zero = Test(pre_vx, specifier = Test.IsZero, likely = False, tag = "x_zero", debug = debugd)

        def bit_match(fp_optree, bit_id, likely = False, ** kwords):
            return NotEqual(BitLogicAnd(TypeCast(fp_optree, precision = ML_Int64), 1 << bit_id), 0, likely = likely, **kwords)

        x_qnan = Test(pre_vx, specifier = Test.IsQuietNaN, likely = False, tag = "x_qnan", debug = debugd)
        if isinstance(processor, K1B_Processor):
            x_zero = bit_match(init_approx, 1)
            x_plus_inf = bit_match(init_approx, 2)
            x_nan_or_neg  = bit_match(init_approx, 3)
        else:
            x_zero = Test(pre_vx, specifier = Test.IsZero, likely = False, tag = "x_zero", debug = debugd)
            x_neg = Comparison(pre_vx, 0, specifier = Comparison.Less, likely = False)
            x_nan = Test(pre_vx, specifier = Test.IsNaN, likely = False, tag = "x_nan", debug = debugd)
            x_nan_or_neg = x_nan | x_neg
            x_plus_inf = Test(pre_vx, specifier = Test.IsPositiveInfty, likely = False, tag = "x_plus_inf", debug = debug)



        #x_inf_or_nan = Test(pre_vx, specifier = Test.IsInfOrNaN, likely = False, tag = "x_inf_or_nan", debug = debugd)
        #x_inf = Test(pre_vx, specifier = Test.IsInfty, likely = False, tag = "x_inf", debug = debugd)

        return_neg = Statement(ClearException(), SqrtRaiseReturn(ML_FPE_Invalid, return_value = FP_QNaN(self.precision)))

        specific_case_bit = bit_match(init_approx, 0, tag = "specific_case_bit")

        # x inf and y inf 
        pre_scheme = ConditionBlock(NotEqual(specific_case_bit, 0, tag = "specific_case", debug = debugd, likely = True),
            Return(result),
            ConditionBlock(x_zero,
                Statement(ClearException(), Return(pre_vx)),
                ConditionBlock(x_nan_or_neg,
                    ConditionBlock(x_qnan, 
                        Statement(ClearException(), Return(FP_QNaN(self.precision))),
                        Statement(ClearException(), SqrtRaiseReturn(ML_FPE_Invalid, return_value = FP_QNaN(self.precision)))
                    ),
                    ConditionBlock(x_plus_inf,
                        Return(pre_vx),
                        Statement(
                            #ConditionBlock(Comparison(d_last, 0, specifier = Comparison.NotEqual, likely = True),
                            #    Raise(ML_FPE_Inexact)
                            #),
                            Return(slow_result if isinstance(processor, K1B_Processor) else result)
                        )
                    )
                )
            )
        )
        scheme = None
        if isinstance(processor, K1B_Processor):
            result.set_clearprevious(False)
            scheme = Statement(result, pre_scheme)
        else:
            rnd_mode = GetRndMode()
            scheme = Statement(rnd_mode, SetRndMode(ML_RoundToNearest), S1, H1, d1, SetRndMode(rnd_mode), result, pre_scheme)


        # fusing FMA
        if fuse_fma:
            print "MDL fusing FMA"
            scheme = opt_eng.fuse_multiply_add(scheme, silence = True)

        print "MDL abstract scheme"
        opt_eng.instantiate_abstract_precision(scheme, None)

        print "MDL instantiated scheme"
        opt_eng.instantiate_precision(scheme, default_precision = self.precision)

        print "subexpression sharing"
        opt_eng.subexpression_sharing(scheme)

        print "silencing DAG"
        opt_eng.silence_fp_operations(scheme)

        # registering scheme as function implementation
        exp_implementation.set_scheme(scheme)

        #print scheme.get_str(depth = None, display_precision = True)

        # check processor support
        if not opt_eng.check_processor_support(scheme):
            Log.report(Log.Error, "unsupported operation error")

        # factorizing fast path
        #opt_eng.factorize_fast_path(scheme)
        
        cg = CCodeGenerator(processor, declare_cst = False, disable_debug = not debug_flag, libm_compliant = libm_compliant)
        self.result = exp_implementation.get_definition(cg, C_Code, static_cst = True)
        self.result.add_header("math.h")
        self.result.add_header("stdio.h")
        self.result.add_header("inttypes.h")
        self.result.add_header("support_lib/ml_special_values.h")

        output_stream = open(output_file, "w")
        output_stream.write(self.result.get(cg))
        output_stream.close()



if __name__ == "__main__":
    # auto-test
    num_iter        = int(extract_option_value("--num-iter", "3"))

    arg_template = ML_ArgTemplate(default_function_name = "sqrt", default_output_file = "sqrt.c")
    arg_template.sys_arg_extraction()


    ml_sqrt          = ML_Sqrt(arg_template.precision, 
                                  libm_compliant            = arg_template.libm_compliant, 
                                  debug_flag                = arg_template.debug_flag, 
                                  target                    = arg_template.target, 
                                  fuse_fma                  = arg_template.fuse_fma, 
                                  fast_path_extract         = arg_template.fast_path,
                                  num_iter                  = num_iter,
                                  function_name             = arg_template.function_name,
                                  output_file               = arg_template.output_file,
                                  dot_product_enabled       = arg_template.dot_product_enabled)
