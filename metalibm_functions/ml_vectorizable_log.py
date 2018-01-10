# -*- coding: utf-8 -*-
# -*- vim: sw=4 sts=4 tw=79

import sys

import sollya # sollya.RN, sollya.absolute, sollya.x
from sollya import (S2, Interval, round, sup, log, log1p, guessdegree)

from metalibm_core.core.ml_function import (ML_Function, ML_FunctionBasis,
                                            DefaultArgTemplate)
from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.core.ml_complex_formats import ML_Mpfr_t
from metalibm_core.opt.ml_blocks import (generate_count_leading_zeros,
                                         generate_fasttwosum)

from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.code_generation.generator_utility import (FunctionOperator,
                                                             FO_Result, FO_Arg)

from metalibm_core.utility.gappa_utils import execute_gappa_script_extract
from metalibm_core.utility.ml_template import *
from metalibm_core.utility.debug_utils import *

class ML_Log(ML_Function("ml_log")):
  def __init__(self, args=DefaultArgTemplate):
    # initializing base class
    ML_FunctionBasis.__init__(self, args)


  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for ML_Log,
        builtin from a default argument mapping overloaded with @p kw """
    default_args_log = {
        "output_file": "LOG.c",
        "function_name": "LOG",
        "precision": ML_Binary32,
        "accuracy": ML_Faithful,
        "target": GenericProcessor()
    }
    default_args_log.update(kw)
    return DefaultArgTemplate(**default_args_log)

  def generate_scheme(self):
    """Produce an abstract scheme for the logarithm.

    This abstract scheme will be used by the code generation backend.
    """
    if self.precision not in [ML_Binary32, ML_Binary64]:
        Log.report(Log.Warning, "The demanded precision is not supported")

    vx = self.implementation.add_input_variable("x", self.precision)

    precision = self.precision.sollya_object
    int_prec = self.precision.get_integer_format()
    uint_prec = self.precision.get_unsigned_integer_format()

    print "MDL constants"
    table_index_size = 7 # to be abstracted somehow
    table_dimensions = [2**table_index_size]
    field_size = Constant(self.precision.get_field_size(),
                          precision = int_prec)
    log2_hi = Constant(
            round(log(2), precision, sollya.RN),
            precision = self.precision,
            tag = 'log2_hi'
            )
    log2_lo = Constant(
            round(log(2) - round(log(2), precision, sollya.RN),
                  precision, sollya.RN),
            precision = self.precision,
            tag = 'log2_lo'
            )
    # The subnormal mask is defined as:
    # 2^(e_min + 2) * (1 - 2^(-precision))
    # e.g. for binary32: 2^(-126 + 2) * (1 - 2^(-24))
    subnormal_mask = Constant(
            value = 0x017fffff if self.precision == ML_Binary32
            else 0x002fffffffffffff if self.precision == ML_Binary64
            else None,
            precision = int_prec,
            tag = "subnormal_mask"
            )
    fp_one = Constant(1.0, precision = self.precision, tag = 'fp_one')
    fp_one_as_uint = TypeCast(fp_one, precision = uint_prec,
                              tag = 'fp_one_as_uint')
    int_zero = Constant(0, precision = int_prec, tag = 'int_zero')
    table_mantissa_half_ulp = Constant(
            1 << (self.precision.field_size - table_index_size - 1),
            precision = int_prec
            )
    table_s_exp_index_mask = Constant(
            ~((table_mantissa_half_ulp.get_value() << 1) - 1),
            precision = uint_prec
            )

    print "MDL table"
    init_log1p_hi = [
            round(log1p(float(i) / table_dimensions[0]),
                  self.precision.get_mantissa_size(),
                  sollya.RN)
            for i in xrange(table_dimensions[0])
            ]
    log1p_table_hi = ML_NewTable(dimensions = table_dimensions,
                                 storage_precision = self.precision,
                                 init_data = init_log1p_hi,
                                 tag = 'ml_log1p_table_high')

    init_log1p_lo = [
            round(log1p(float(i) / table_dimensions[0]) - init_log1p_hi[i],
                  self.precision.get_mantissa_size(),
                  sollya.RN)
            for i in xrange(table_dimensions[0])
            ]
    log1p_table_lo = ML_NewTable(dimensions = table_dimensions,
                                 storage_precision = self.precision,
                                 init_data = init_log1p_lo,
                                 tag = 'ml_log1p_table_low')

    print 'MDL unified subnormal handling'
    vx_as_int = TypeCast(vx, precision = int_prec, tag = 'vx_as_int')
    vx_as_uint = TypeCast(vx, precision = uint_prec, tag = 'vx_as_uint')
    # Avoid the 0.0 case
    denorm = vx_as_int - 1
    is_normal = denorm >= subnormal_mask
    # hazardous conversion is_normal is a boolean
    is_subnormal = Select(
            is_normal,
            Constant(0, precision = int_prec),
            Constant(-1, precision = int_prec)
            )
    # is_subnormal = is_normal - 1
    #is_subnormal = Conversion(denorm < subnormal_mask, precision = int_prec)

    #################################################
    # Vectorizable integer based subnormal handling #
    #################################################
    # 1. lzcnt
    # custom lzcount-like for subnormal numbers using FPU (see draft article)
    Zi = BitLogicOr(vx_as_uint, fp_one_as_uint, precision = uint_prec)
    Zf = Subtraction(
            TypeCast(Zi, precision = self.precision),
            fp_one,
            precision = self.precision
            )
    # Zf exponent is -(nlz(x) - exponent_size).
    # 2. compute shift value
    # Vectorial comparison on x86+sse/avx is going to look like
    # '|0x00|0xff|0x00|0x00|' and that's why we use Negate.
    # But for a scalar code generation, comparison will rather be either 0 or
    # something different from 0 (*unspecified*, but often assumed to be 1).
    # Thus this mask below won't be correct for a scalar implementation...
    # FIXME: Can we know the backend that will be called and choose in
    # consequence? Should we make something arch-agnostic instead?
    def RawExponentExtraction(f):
        """Get the raw (biased) exponent of f.
        Only if f is a positive floating-point number.
        """
        return BitLogicRightShift(
                TypeCast(f, precision = uint_prec),
                Constant(
                    f.get_precision().get_field_size(),
                    precision = uint_prec
                    ),
                precision = int_prec
                )

    mask = BitLogicNegate(
            TypeCast(
                Comparison(
                    RawExponentExtraction(vx),
                    int_zero,
                    specifier = Comparison.NotEqual
                    ),
                precision = uint_prec
                ),
            precision = uint_prec,
            tag = 'mask'
            )
    n_value = BitLogicAnd(
            TypeCast(
                Addition(
                    RawExponentExtraction(Zf),
                    Constant(
                        self.precision.get_bias(),
                        precision = int_prec
                        ),
                    precision = int_prec
                    ),
                precision = uint_prec
                ),
            mask,
            precision = uint_prec
            )
    value = Negation(TypeCast(n_value, precision = int_prec))

    # 3. shift left
    renormalized_mantissa = BitLogicLeftShift(vx_as_int, value)
    # 4. set exponent to the right value
    # Compute the exponent to add : (p-1)-(value) + 1 = p-1-value
    # The final "+ 1" comes from the fact that once renormalized, the
    # floating-point datum has a biased exponent of 1
    tmp0 = Subtraction(
            field_size,
            value,
            precision = int_prec
            )
    # Set the value to 0 if the number is not subnormal
    tmp1 = BitLogicAnd(tmp0, is_subnormal)
    renormalized_exponent = BitLogicLeftShift(
            tmp1,
            field_size,
            )

    normal_vx_as_int = renormalized_mantissa + renormalized_exponent
    normal_vx = TypeCast(normal_vx_as_int, precision = self.precision,
                         tag = 'normal_vx')

    alpha = BitLogicAnd(field_size, is_subnormal)
    # XXX Extract the mantissa, see if this is supported in the x86 vector
    # backend or if it still uses the support_lib.
    vx_mantissa = MantissaExtraction(normal_vx, precision = self.precision)

    print "MDL scheme"
    # TODO if binary64 precision, also use FastReciprocal and not Division
    rcp_m = FastReciprocal(vx_mantissa, tag = 'rcp_m', precision = self.precision)
    if not self.processor.is_supported_operation(rcp_m):
        # FIXME An approximation table could be used instead but for vector
        # implementations another GATHER would be required.
        # However this may well be better than a division...
        # See also: using binary32 FastReciprocal for approximating 1/m when m
        # is a binary64.
        rcp_m = Division(fp_one, vx_mantissa, tag = 'rcp_m')

    # exponent is normally either 0 or -1, since m is in [1, 2). Possible
    # optimization?
    exponent = ExponentExtraction(rcp_m, precision = self.precision,
            tag = 'exponent')

    ri_round = TypeCast(
            Addition(
                TypeCast(rcp_m, precision = int_prec),
                table_mantissa_half_ulp,
                precision = int_prec
                ),
            precision = uint_prec
            )
    ri_fast_rndn = BitLogicAnd(
            ri_round,
            table_s_exp_index_mask,
            tag = 'ri_fast_rndn',
            precision = uint_prec
            )
    # u = m * ri - 1
    u = FusedMultiplyAdd(
            vx_mantissa,
            TypeCast(ri_fast_rndn, precision = self.precision),
            fp_one,
            specifier = FusedMultiplyAdd.Subtract,
            tag = 'u'
            )
    unneeded_bits = Constant(
            self.precision.field_size - table_index_size,
            precision = int_prec
            )
    ri_bits = BitLogicRightShift(
            ri_fast_rndn,
            unneeded_bits,
            precision = uint_prec
            )
    table_index_mask = Constant(
            (1 << table_index_size) - 1,
            precision = uint_prec
            )
    table_index = BitLogicAnd(
            ri_bits,
            table_index_mask,
            tag = 'table_index',
            precision = uint_prec
            )
    tbl_hi = TableLoad(log1p_table_hi, table_index, tag = 'tbl_hi',
                       debug = debug_multi)
    tbl_lo = TableLoad(log1p_table_lo, table_index, tag = 'tbl_lo',
                       debug = debug_multi)
    exponent = Addition(
            BitLogicRightShift(ri_bits,
                table_index_size, tag = 'exponent',
                interval = self.precision.get_exponent_interval()),
            alpha,
            tag = 'modified_exponent',
            #interval = self.precision.get_exponent_interval() \
            #        + alpha.get_interval()
            )
    fp_exponent = Conversion(exponent, precision = self.precision,
            tag = 'fp_exponent')

    print 'MDL polynomial approximation'
    sollya_function = log(1 + sollya.x)
    arg_red_mag = 2**(-table_index_size)
    approx_interval = Interval(-arg_red_mag, arg_red_mag)
    max_eps = 2**-(self.precision.get_field_size() + 10)
    print "max acceptable error for polynomial = {}".format(float.hex(max_eps))
    poly_degree = sup(
            guessdegree(
                sollya_function,
                approx_interval,
                max_eps,
                )
            )
    poly_object = Polynomial.build_from_approximation(
            sollya_function,
            range(1, poly_degree + 1), # Force 1st coeff to 0
            # Emulate double-self.precision coefficient formats
            [self.precision.get_mantissa_size()*2 + 1]*poly_degree,
            approx_interval,
            sollya.absolute,
            0) # Force the first coefficient to 0

    print poly_object

    constant_precision = ML_SingleSingle if self.precision == ML_Binary32 \
            else ML_DoubleDouble if self.precision == ML_Binary64 \
            else None
    if is_cgpe_available():
        log1pu_poly = PolynomialSchemeEvaluator.generate_cgpe_scheme(
                poly_object,
                u,
                unified_precision = self.precision,
                constant_precision = constant_precision
                )
    else:
        Log.report(Log.Warning,
                "CGPE not available, falling back to std poly evaluator")
        log1pu_poly = PolynomialSchemeEvaluator.generate_horner_scheme(
                poly_object,
                u,
                unified_precision = self.precision,
                constant_precision = constant_precision
                )

    # XXX Dirty implementation of double-(self.precision) poly
    def Mul211(x, y):
        zh = Multiplication(x, y)
        zl = FusedMultiplyAdd(x, y, zh, specifier = FusedMultiplyAdd.Subtract)
        return zh, zl

    def Add211(x, y):
        zh = Addition(x, y)
        t1 = Subtraction(zh, x)
        zl = Subtraction(y, t1)
        return zh, zl

    def Mul212(x, yh, yl):
        t1, t2 = Mul211(x, yh)
        t3 = Multiplication(x, yl)
        t4 = Addition(t2, t3)
        return Add211(t1, t4)

    def Mul222(xh, xl, yh, yl):
        ph = Multiplication(xh, yh)
        pl = FMS(xh, yh, ph)
        pl = FMA(xh, yl, pl)
        pl = FMA(xl, yh, pl)
        zh = Addition(ph, pl)
        zl = Subtraction(ph, zh)
        zl = Addition(zl, pl)
        return zh, zl

    def Add222(xh, xl, yh, yl):
        r = Addition(xh, yh)
        s1 = Subtraction(xh, r)
        s2 = Addition(s1, yh)
        s3 = Addition(s2, yl)
        s = Addition(s3, xl)
        zh = Addition(r, s)
        zl = Addition(Subtraction(r, zh), s)
        return zh, zl

    def Add122(xh, xl, yh, yl):
        zh, _ = Add222(xh, xl, yh, yl)
        return zh

    def dirty_poly_node_conversion(node, variable):
        if node is variable:
            return variable, None
        elif isinstance(node, Constant):
            value = node.get_value()
            value_hi = round(value, precision, sollya.RN)
            value_lo =  value - value_hi
            ch = Constant(
                    value_hi,
                    tag = node.get_tag() + "hi",
                    precision = self.precision)
            cl = Constant(
                    value_lo,
                    tag = node.get_tag() + "lo",
                    precision = self.precision
                    )
            return ch, cl

        inputs = node.get_inputs()
        if isinstance(node, Addition):
            op1h, op1l = dirty_poly_node_conversion(inputs[0], variable)
            op2h, op2l = dirty_poly_node_conversion(inputs[1], variable)
            return Add222(op1h, op1l, op2h, op2l)
        if isinstance(node, Subtraction):
            op1h, op1l = dirty_poly_node_conversion(inputs[0], variable)
            op2h, op2l = dirty_poly_node_conversion(inputs[1], variable)
            return Sub222(op1h, op1l, op2h, op2l)
        elif isinstance(node, Multiplication):
            op1h, op1l = dirty_poly_node_conversion(inputs[0], variable)
            op2h, op2l = dirty_poly_node_conversion(inputs[1], variable)
            if op1l is None:
                if op2l is None:
                    return Mul211(op1h, op2h)
                else:
                    return Mul212(op1h, op2h, op2l)
            else:
                if op2l is None:
                    return Mul212(op2h, op1h, op1l)
                else:
                    return Mul222(op1h, op1l, op2h, op2l)

    log1pu_poly_hi, log1pu_poly_lo = dirty_poly_node_conversion(log1pu_poly, u)
    log1pu_poly_hi.set_attributes(tag = 'log1pu_poly_hi')
    log1pu_poly_lo.set_attributes(tag = 'log1pu_poly_lo')

    # Compute log(2) * (e + tau - alpha)
    log2e_hi, log2e_lo = Mul212(fp_exponent, log2_hi, log2_lo)

    # Add log1p(u)
    tmp_res_hi, tmp_res_lo = Add222(log2e_hi, log2e_lo,
                                    log1pu_poly_hi, log1pu_poly_lo)


    # Add -log(2^(tau)/m) approximation retrieved by two table lookups
    logx_hi = Add122(tmp_res_hi, tmp_res_lo, tbl_hi, tbl_lo)

    scheme = Return(logx_hi, precision = self.precision)

    return scheme

  def numeric_emulate(self, input_value):
    return log(input_value)

if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(
          default_arg=ML_Log.get_default_args())
  args = arg_template.arg_extraction()

  ml_log = ML_Log(args)
  ml_log.gen_implementation()
