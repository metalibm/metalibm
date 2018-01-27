# -*- coding: utf-8 -*-
# -*- vim: sw=4 sts=4 tw=79

import sys

import sollya
from sollya import (floor, guessdegree, Interval, log, log1p, round, S2, sqrt,
                    sup, _x_)

from metalibm_core.core.ml_function import (ML_Function, ML_FunctionBasis,
                                            DefaultArgTemplate)
from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.core.ml_complex_formats import ML_Mpfr_t
from metalibm_core.opt.ml_blocks import (
    generate_count_leading_zeros, generate_fasttwosum,
    Add222, Add122, Add212, Add211, Mul212, Mul211, Mul222
)
from metalibm_core.opt.p_expand_multi_precision import (
    dirty_multi_node_expand
)

from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.targets.common.vector_backend import VectorBackend

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
        Log.report(Log.Error, "The demanded precision is not supported")

    vx = self.implementation.add_input_variable("x", self.precision)

    precision = self.precision.sollya_object
    int_prec = self.precision.get_integer_format()
    uint_prec = self.precision.get_unsigned_integer_format()

    # bool2int conversion helper functions
    def bool_convert(optree, precision, true_value, false_value):
        """ Implement conversion between boolean node (ML_Bool)
            and specific values """
        return Select(
            optree,
            Constant(true_value, precision=precision),
            Constant(false_value, precision=precision),
            precision=precision
        )

    def default_bool_convert(optree, precision=None):
        return bool_convert(optree, precision, -1, 0) \
                if isinstance(self.processor, VectorBackend) \
                else bool_convert(optree, precision, 1, 0)

    print "MDL constants"
    cgpe_scheme_idx = int(args.cgpe_index)
    table_index_size = int(args.tbl_index_size)
    #
    table_nb_elements = 2**(table_index_size)
    table_dimensions = [2*table_nb_elements]  # two values are stored for each element
    field_size = Constant(self.precision.get_field_size(),
                          precision = int_prec,
                          tag = 'field_size')
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
    # subnormal_mask aims at trapping positive subnormals except zero.
    # That's why we will subtract 1 to the integer bitstring of the input, and
    # then compare for Less (strict) the resulting integer bitstring to this
    # mask, e.g.  0x7fffff for binary32.
    subnormal_mask = Constant((1 << self.precision.get_field_size()) - 1,
                              precision = int_prec, tag = 'subnormal_mask')
    fp_one = Constant(1.0, precision = self.precision, tag = 'fp_one')
    fp_one_as_uint = TypeCast(fp_one, precision = uint_prec,
                              tag = 'fp_one_as_uint')
    int_zero = Constant(0, precision = int_prec, tag = 'int_zero')
    int_one  = Constant(1, precision = int_prec, tag = 'int_one')
    table_mantissa_half_ulp = Constant(
            1 << (self.precision.field_size - table_index_size - 1),
            precision = int_prec
            )
    table_s_exp_index_mask = Constant(
            ~((table_mantissa_half_ulp.get_value() << 1) - 1),
            precision = uint_prec
            )

    print "MDL table"
    # The table holds approximations of -log(2^tau * r_i) so we first compute
    # the index value for which tau changes from 2 to 0.
    cut = sqrt(2.)
    tau_index_limit = floor(table_nb_elements * (2./cut - 1))
    sollya_logtbl = [
            -log1p(float(i) / table_nb_elements)
            + (0 if i <= tau_index_limit else log(2.))
            for i in xrange(table_nb_elements)
            ]
    # ...
    init_logtbl_hi = [
            round(sollya_logtbl[i],
                  self.precision.get_mantissa_size(),
                  sollya.RN)
            for i in xrange(table_nb_elements)
    ]
    init_logtbl_lo = [
            round(sollya_logtbl[i] - init_logtbl_hi[i],
                  self.precision.get_mantissa_size(),
                  sollya.RN)
            for i in xrange(table_nb_elements)
    ]
    init_logtbl = [tmp[i] for i in xrange(len(init_logtbl_hi)) for tmp in [init_logtbl_hi, init_logtbl_lo]]
    log1p_table = ML_NewTable(dimensions = table_dimensions,
                              storage_precision = self.precision,
                              init_data = init_logtbl,
                              tag = 'ml_log1p_table')

    print 'MDL unified subnormal handling'
    vx_as_int = TypeCast(vx, precision = int_prec, tag = 'vx_as_int')
    vx_as_uint = TypeCast(vx, precision = uint_prec, tag = 'vx_as_uint')
    # Avoid the 0.0 case by subtracting 1 from vx_as_int
    tmp = Comparison(vx_as_int - 1, subnormal_mask,
                     specifier = Comparison.Less)
    is_subnormal = default_bool_convert(
            tmp, # Will catch negative values as well as NaNs with sign bit set
            precision = int_prec,
            )
    is_subnormal.set_attributes(tag = "is_subnormal")
    if not(isinstance(self.processor, VectorBackend)):
      is_subnormal = Subtraction(Constant(0, precision = int_prec),
                                 is_subnormal,
                                 precision = int_prec)

    #################################################
    # Vectorizable integer based subnormal handling #
    #################################################
    # 1. lzcnt
    # custom lzcount-like for subnormal numbers using FPU (see draft article)
    Zi = BitLogicOr(vx_as_uint, fp_one_as_uint, precision = uint_prec, tag="Zi")
    Zf = Subtraction(
            TypeCast(Zi, precision = self.precision),
            fp_one,
            precision = self.precision,
            tag="Zf")
    # Zf exponent is -(nlz(x) - exponent_size).
    # 2. compute shift value
    # Vectorial comparison on x86+sse/avx is going to look like
    # '|0x00|0xff|0x00|0x00|' and that's why we use Negate.
    # But for scalar code generation, comparison will rather be either 0 or 1
    # in C. Thus mask below won't be correct for a scalar implementation.
    # FIXME: Can we know the backend that will be called and choose in
    # consequence? Should we make something arch-agnostic instead?
    def DirtyExponentExtraction(f):
        """Get the raw (biased) exponent of f.
        Only if f is a positive floating-point number.
        """
        recast_f = TypeCast(f, precision = int_prec) \
                if f.get_precision() != int_prec else f
        return BitLogicRightShift(
                recast_f,
                field_size,
                precision = int_prec
                )

    n_value = BitLogicAnd(
            Addition(
                DirtyExponentExtraction(Zf),
                Constant(
                    self.precision.get_bias(),
                    precision = int_prec
                    ),
                precision = int_prec
                    ),
            is_subnormal,
            precision = int_prec,
            tag = "n_value"
            )
    alpha = Negation(n_value, tag="alpha")

    # 3. shift left
    # renormalized_mantissa = BitLogicLeftShift(vx_as_int, value)
    normal_vx_as_int = BitLogicLeftShift(vx_as_int, alpha)
    # 4. set exponent to the right value
    # Compute the exponent to add : (p-1)-(value) + 1 = p-1-value
    # The final "+ 1" comes from the fact that once renormalized, the
    # floating-point datum has a biased exponent of 1
    #tmp0 = Subtraction(
    #        field_size,
    #        value,
    #        precision = int_prec,
    #        tag="tmp0")
    # Set the value to 0 if the number is not subnormal
    #tmp1 = BitLogicAnd(tmp0, is_subnormal)
    #renormalized_exponent = BitLogicLeftShift(
    #        tmp1,
    #        field_size
    #        )

    #normal_vx_as_int = renormalized_mantissa + renormalized_exponent
    normal_vx = TypeCast(normal_vx_as_int, precision = self.precision,
                         tag = 'normal_vx')

    # alpha = BitLogicAnd(field_size, is_subnormal, tag = 'alpha')
    # XXX Extract the mantissa, see if this is supported in the x86 vector
    # backend or if it still uses the support_lib.
    vx_mantissa = MantissaExtraction(normal_vx, precision = self.precision)

    print "MDL scheme"
    rcp_m = FastReciprocal(vx_mantissa, precision = self.precision)
    if not self.processor.is_supported_operation(rcp_m):
        if self.precision == ML_Binary64:
            # Try using a binary32 FastReciprocal
            binary32_m = Conversion(vx_mantissa, precision = ML_Binary32)
            rcp_m = FastReciprocal(binary32_m, precision = ML_Binary32)
            rcp_m = Conversion(rcp_m, precision = ML_Binary64)
        if not self.processor.is_supported_operation(rcp_m):
            # FIXME An approximation table could be used instead but for vector
            # implementations another GATHER would be required.
            # However this may well be better than a division...
            rcp_m = Division(fp_one, vx_mantissa, precision = self.precision)
    rcp_m.set_attributes(tag = 'rcp_m')

    # exponent is normally either 0 or -1, since m is in [1, 2). Possible
    # optimization?
    # exponent = ExponentExtraction(rcp_m, precision = self.precision,
    #         tag = 'exponent')

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
            precision=uint_prec,
            tag="unneeded_bits"
            )
    assert self.precision.field_size - table_index_size >= 0
    ri_bits = BitLogicRightShift(
            ri_fast_rndn,
            unneeded_bits,
            precision = uint_prec,
            tag = "ri_bits"
            )
    # Retrieve mantissa's MSBs + first bit of exponent, for tau computation in case
    # exponent is 0 (i.e. biased 127, i.e. first bit of exponent is set.).
    # In this particular case, i = 0 but tau is 1
    # table_index does not need to be as long as uint_prec might be,
    # try and keep it the size of size_t.
    size_t_prec = ML_UInt32
    signed_size_t_prec = ML_Int32
    table_index_mask = Constant(
            (1 << (table_index_size + 1)) - 1,
            precision = size_t_prec
            )
    table_index = BitLogicAnd(
            Conversion(ri_bits, precision = size_t_prec),
            table_index_mask,
            tag = 'table_index',
            precision = size_t_prec
            )
    # Compute tau using the tau_index_limit value.
    tmp = default_bool_convert(
            Comparison(
                TypeCast(table_index, precision = signed_size_t_prec),
                Constant(tau_index_limit, precision = signed_size_t_prec),
                specifier = Comparison.Greater
                if isinstance(self.processor, VectorBackend)
                else Comparison.LessOrEqual
                ),
            precision = signed_size_t_prec
            )
    # A true tmp will typically be -1 for VectorBackends, but 1 for standard C.
    tau = Conversion(
        Addition(tmp, int_one, precision = signed_size_t_prec)
            if isinstance(self.processor, VectorBackend)
            else tmp,
            precision = int_prec
        )
    tau.set_attributes(tag = 'tau')
    # Update table_index: keep only table_index_size bits
    table_index_hi = BitLogicAnd(
            table_index,
            Constant((1 << table_index_size) - 1, precision = size_t_prec),
            precision = size_t_prec
            )
    # table_index_hi = table_index_hi << 1
    table_index_hi = BitLogicLeftShift(
            table_index_hi,
            Constant(1, precision = size_t_prec),
            precision = size_t_prec,
            tag = "table_index_hi"
            )
    # table_index_lo = table_index_hi + 1
    table_index_lo = Addition(
            table_index_hi,
            Constant(1, precision = size_t_prec),
            precision = size_t_prec,
            tag = "table_index_lo"
            )

    tbl_hi = TableLoad(log1p_table, table_index_hi, tag = 'tbl_hi',
                       debug = debug_multi)
    tbl_lo = TableLoad(log1p_table, table_index_lo, tag = 'tbl_lo',
                       debug = debug_multi)
    # Compute exponent e + tau - alpha, but first subtract the bias.
    tmp_eptau = Addition(
            Addition(
                BitLogicRightShift(
                    normal_vx_as_int,
                    field_size,
                    tag = 'exponent',
                    interval = self.precision.get_exponent_interval(),
                    precision = int_prec
                    ),
                Constant(
                    self.precision.get_bias(),
                    precision = int_prec
                    )
                ),
            tau,
            tag = 'tmp_eptau',
            precision = int_prec
            )
    exponent = Subtraction(tmp_eptau, alpha, precision = int_prec)
    fp_exponent = Conversion(exponent, precision = self.precision,
                             tag = 'fp_exponent')

    print 'MDL polynomial approximation'
    sollya_function = log(1 + sollya.x)
    # arg_red_mag = 2**(-table_index_size)
    # approx_interval = Interval(-arg_red_mag, arg_red_mag)
    boundrcp = 1.5 * 2**(-12)           # ... see Intel intrinsics guide
    if self.precision in [ML_Binary64]:
      if not self.processor.is_supported_operation(rcp_m):
        boundrcp = (1+boundrcp)*(1+2**(-24)) - 1
      else:
        boundrcp = 2**(-14)             # ... see Intel intrinsics guide
    arg_red_mag = boundrcp + 2**(-table_index_size-1) + boundrcp * 2**(-table_index_size-1)
    approx_interval = Interval(-arg_red_mag, arg_red_mag)
    # max_eps = 2**-(self.precision.get_field_size() + 10)
    max_eps = 2**-(2*(self.precision.get_field_size()+1))
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
            range(2, poly_degree + 1), # Force 1st 2 coeffs to 0 and 1, resp.
            # Emulate double-self.precision coefficient formats
            [self.precision.get_mantissa_size()*2 + 1]*(poly_degree - 1),
            approx_interval,
            sollya.absolute,
            0 + sollya._x_) # Force the first 2 coefficients to 0 and 1, resp.

    print poly_object

    constant_precision = ML_SingleSingle if self.precision == ML_Binary32 \
            else ML_DoubleDouble if self.precision == ML_Binary64 \
            else None
    if is_cgpe_available():
        log1pu_poly = PolynomialSchemeEvaluator.generate_cgpe_scheme(
                poly_object,
                u,
                unified_precision = self.precision,
                constant_precision = constant_precision, scheme_id = cgpe_scheme_idx
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
    def dirty_poly_node_conversion(node, variable):
        return dirty_multi_node_expand(
            node, self.precision, mem_map={variable: (variable, None)})

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
    logx_hi.set_attributes(tag = 'logx_hi')

    scheme = Return(logx_hi, precision = self.precision)

    return scheme

  def numeric_emulate(self, input_value):
    return log(input_value)

if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(
          default_arg=ML_Log.get_default_args())
  #
  arg_template.get_parser().add_argument("--table-index-size", dest = "tbl_index_size", action = "store", default = 7, help = "table index size (default: 7)")
  arg_template.get_parser().add_argument("--cgpe-scheme-index", dest = "cgpe_index", action = "store", default = 0, help = "CGPE scheme index (default: 0)")
  #
  args = arg_template.arg_extraction()

  ml_log = ML_Log(args)
  ml_log.gen_implementation()
