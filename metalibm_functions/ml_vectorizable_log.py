# -*- coding: utf-8 -*-
# -*- vim: sw=4 sts=4 tw=79

import sys

import sollya # sollya.RN, sollya.absolute, sollya.x
from sollya import (S2, Interval, round, sup, log, log2, guessdegree)

from metalibm_core.core.ml_function import (ML_Function, ML_FunctionBasis,
                                            DefaultArgTemplate)
from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_Table
from metalibm_core.core.ml_complex_formats import ML_Mpfr_t

from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.generator_utility import (FunctionOperator,
                                                             FO_Result, FO_Arg)

from metalibm_core.utility.gappa_utils import execute_gappa_script_extract
from metalibm_core.utility.ml_template import *
from metalibm_core.utility.debug_utils import *

class ML_Log(ML_Function("ml_log")):
  def __init__(self,
               arg_template=DefaultArgTemplate,
               precision=ML_Binary32,
               abs_accuracy=S2**-24,
               libm_compliant=True,
               debug_flag=False,
               fuse_fma=True,
               fast_path_extract=True,
               target=GenericProcessor(),
               output_file="ml_log.c",
               function_name="ml_log"):
    # initializing I/O precision
    precision = ArgDefault.select_value([arg_template.precision, precision])
    io_precisions = [precision] * 2

    # initializing base class
    ML_FunctionBasis.__init__(self,
            base_name = "log",
            function_name = function_name,
            output_file = output_file,
            io_precisions = io_precisions,
            abs_accuracy = None,
            libm_compliant = libm_compliant,
            processor = target,
            fuse_fma = fuse_fma,
            fast_path_extract = fast_path_extract,
            debug_flag = debug_flag,
            arg_template = arg_template)

    self.precision = precision

  def generate_emulate(self, result_ternary, result, mpfr_x, mpfr_rnd):
    """ generate the emulation code for the function
        mpfr_x is a mpfr_t variable which should have the right precision
        mpfr_rnd is the rounding mode
    """
    emulate_func_name = "mpfr_log"
    argc = 3
    emulate_func_op = FunctionOperator(
            emulate_func_name,
            arg_map = { i: FO_Arg(i) for i in range(argc) },
            require_header = ["mpfr.h"])
    emulate_func = FunctionObject(
            emulate_func_name,
            [ ML_Mpfr_t, ML_Mpfr_t, ML_Int32 ],
            ML_Int32,
            emulate_func_op)
    mpfr_call = Statement(ReferenceAssign(
            result_ternary, emulate_func(result, mpfr_x, mpfr_rnd)))
    return mpfr_call


  def generate_scheme(self):
    """Produce an abstract scheme for the logarithm.

    This abstract scheme will be used by the code generation backend.
    """
    vx = self.implementation.add_input_variable("x", self.precision)

    precision = self.precision.sollya_object
    int_prec = self.precision.get_integer_format()
    uint_prec = self.precision.get_unsigned_integer_format()

    # The denormalized mask is defined as:
    # 2^(e_min + 2) * (1 - 2^(-precision))
    # e.g. for binary32: 2^(-126 + 2) * (1 - 2^(-24))
    denorm_mask = Constant(value = 0x017fffff if self.precision == ML_Binary32
                           else 0x002fffffffffffff, precision = int_prec)

    print "MDL table"
    table_index_size = 7 # to be abstracted somehow
    dimensions = [2**table_index_size]
    init_log2 = [
            round(log2(1 + float(i) / dimensions[0])
                - (2**(self.precision.exponent_size - 1) - 1),
                self.precision.field_size, sollya.RN)
            for i in xrange(dimensions[0])
            ]
    log2_table = ML_Table(dimensions = dimensions,
            storage_precision = self.precision,
            init_data = init_log2,
            tag = 'ml_log2_table')

    print 'MDL unified denormal handling'
    vx_as_int = TypeCast(vx, precision = int_prec, tag = 'vx_as_int')
    vx_as_uint = TypeCast(vx, precision = uint_prec, tag = 'vx_as_uint')
    # Avoid the 0.0 case
    denorm = vx_as_int - 1
    is_normal = denorm >= denorm_mask
    is_denormal = is_normal - 1
    #is_denormal = Conversion(denorm < denorm_mask, precision = int_prec)

    # NO BRANCH, INTEGER BASED DENORMAL AND LARGE EXPONENT HANDLING
    # 1. lzcnt
    lzcount = CountLeadingZeros(vx_as_uint, tag = 'lzcount',
            interval = Interval(0, self.precision.bit_size),
            precision = uint_prec)
    max8 = Max(8, Conversion(lzcount, precision = int_prec)) # Max of lzcnt and 8
    # 2. compute shift value
    shift = max8 -  8
    # 3. shift left
    res = BitLogicLeftShift(vx_as_int, shift)
    # 4. set exponent to the right value
    tmp0 = 25 - shift
    tmp1 = BitLogicAnd(tmp0, is_denormal)
    tmp2 = BitLogicLeftShift(tmp1, 23)
    exponent = tmp2 - (1 << 24) # tmp2 - 2^24

    normal_vx_as_int = res + exponent
    normal_vx = TypeCast(normal_vx_as_int, precision = self.precision,
                         tag = 'normal_vx')
    mask_to_add = BitLogicAnd(is_denormal, 25) - 2

    invx = FastReciprocal(normal_vx, tag = 'invx', precision = self.precision)
    if not self.processor.is_supported_operation(invx):
        # An approximation table could be used instead.
        invx = Division(1.0, normal_vx, tag = 'invx')

    print "MDL scheme"
    exponent = ExponentExtraction(invx, precision = self.precision,
            tag = 'exponent')
    nlog2 = Constant(round(-log(2), precision, sollya.RN),
            precision = self.precision,
            tag = 'nlog2')

    table_mantissa_half_ulp = \
            1 << (self.precision.field_size - table_index_size - 1)
    invx_round = TypeCast(invx, precision = int_prec, tag = 'invx_int') \
            + table_mantissa_half_ulp
    table_s_exp_index_mask = ~((table_mantissa_half_ulp << 1) - 1)
    invx_fast_rndn = BitLogicAnd(
            invx_round,
            table_s_exp_index_mask,
            tag = 'invx_fast_rndn'
            )
    # u should be optimized as an FMA
    u = normal_vx * TypeCast(invx_fast_rndn, precision = self.precision) - 1.0
    u.set_attributes(tag = 'u')
    unneeded_bits = self.precision.field_size - table_index_size
    invx_bits = BitLogicRightShift(invx_fast_rndn, unneeded_bits)
    table_index_mask = (1 << table_index_size) - 1
    table_index = BitLogicAnd(
            invx_bits,
            table_index_mask,
            tag = 'table_index')
    log2_1_p_rcp_x = TableLoad(log2_table, table_index, tag = 'log2_1_p_rcp_x',
            debug = debug_multi)
    exponent = BitLogicRightShift(invx_bits,
            table_index_size,
            tag = 'exponent') + mask_to_add
    expf = Conversion(exponent,
            precision = self.precision,
            tag = 'expf')

    print 'MDL polynomial approximation'
    arg_red_mag = 1.0 / dimensions[0]
    approx_interval = Interval(-arg_red_mag, arg_red_mag)
    sollya_function = log(1 + sollya.x)
    poly_degree = sup(guessdegree(
        sollya_function,
        approx_interval,
        S2**-(self.precision.get_field_size() * 2))) # prec + 1 bit
    poly_object = Polynomial.build_from_approximation(
            sollya_function,
            range(1, poly_degree + 1), # Force 1st coeff to 0
            [self.precision]*(poly_degree),
            approx_interval,
            sollya.absolute,
            0) # Force the first coefficient to 0

    print poly_object

    if is_cgpe_available():
      log1pu_poly = PolynomialSchemeEvaluator.generate_cgpe_scheme(
              poly_object,
              u,
              unified_precision = self.precision,
              )
    else:
      Log.report(Log.Warning, "CGPE not available, falling back to std poly evaluator")
      log1pu_poly = PolynomialSchemeEvaluator.generate_horner_scheme(
              poly_object,
              u,
              unified_precision = self.precision,
              )
    log1pu_poly.set_attributes(
            tag = 'log1pu_poly',
            debug = debug_lftolx)

    logx = nlog2 * (log2_1_p_rcp_x + expf) + log1pu_poly

    scheme = Return(logx, precision = self.precision)

    return scheme

  def numeric_emulate(self, input_value):
    return 1.0

if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(
          default_function_name = "LOG",
          default_output_file = "LOG.c" )
  args = arg_template.arg_extraction()

  ml_log = ML_Log(args)
  ml_log.gen_implementation()
