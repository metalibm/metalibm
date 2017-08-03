from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_hdl_format import *
from metalibm_core.core.ml_hdl_operations import *

## Wrapper for zero extension
# @param op the input operation tree
# @param s integer size of the extension
# @return the Zero extended operation node
def zext(op,s):
  s = int(s)
  assert s >= 0
  if s == 0:
    return op
  op_size = op.get_precision().get_bit_size() 
  ext_precision  = ML_StdLogicVectorFormat(op_size + s)
  return ZeroExt(op, s, precision = ext_precision)


## Wrapper for sign extension
def sext(op,s):
  s = int(s)
  assert s >= 0
  if s == 0:
    return op
  op_size = op.get_precision().get_bit_size() 
  ext_precision  = ML_StdLogicVectorFormat(op_size + s)
  return SignExt(op, s, precision = ext_precision)


## Generate the right zero extended output from @p optree
def rzext(optree, ext_size, **kw):
  assert ext_size >= 0
  if ext_size == 0:
    return optree
  ext_size = int(ext_size)
  op_size = optree.get_precision().get_bit_size()
  ext_format = ML_StdLogicVectorFormat(ext_size)
  out_format = ML_StdLogicVectorFormat(op_size + ext_size)
  return Concatenation(optree, Constant(0, precision = ext_format), precision = out_format, **kw)

###############################################################################
##                       Floating-point predicates 
###############################################################################

## Generates is infinity or NaN predicate
#  @param op input operation graph
#  @return ML_Bool operation graph implementing predicates
def fp_is_infornan(op):
  op_prec = op.get_precision().get_base_format()
  exp_prec = ML_StdLogicVectorFormat(op_prec.get_exponent_size())
  exp = ExponentExtraction(op, precision = exp_prec)
  return Equal(
    exp,
    Constant(op_prec.get_special_exponent_value(), precision = exp_prec),
    precision = ML_Bool
  )

## Generates is subnormal predicates (includes zero cases)
#  @param op input operation graph
#  @return ML_Bool operation graph implementing predicates
def fp_is_subnormal(op):
  op_prec = op.get_precision().get_base_format()
  exp_prec = ML_StdLogicVectorFormat(op_prec.get_exponent_size())
  exp = ExponentExtraction(op, precision = exp_prec)
  return Equal(
    exp,
    Constant(0, precision = exp_prec),
    precision = ML_Bool
  )

## Generates is zero predicates 
#  @param op input operation graph
#  @return ML_Bool operation graph implementing predicates
def fp_is_zero(op):
  return LogicalAnd(
    fp_is_subnormal(op),
    fp_mant_is_zero(op),
    precision = ML_Bool
  )

## Generate is positive infinity predicate
#  @param op Operation node to be tested
#  @return Operation node implementing predicate (ML_Bool precision)
def fp_is_pos_inf(op):
  return LogicalAnd(
    fp_is_inf(op),
    Equal(
      CopySign(op, precision = ML_StdLogic), 
      Constant(0, precision = ML_StdLogic),
      precision = ML_Bool
    ),
    precision = ML_Bool
  )

## Generate is negative infinity predicate
#  @param op Operation node to be tested
#  @return Operation node implementing predicate (ML_Bool precision)
def fp_is_neg_inf(op):
  return LogicalAnd(
    fp_is_inf(op),
    Equal(
      CopySign(op, precision = ML_StdLogic), 
      Constant(1, precision = ML_StdLogic),
      precision = ML_Bool
    ),
    precision = ML_Bool
  )

## Generate is mantissa field equals to zero predicate
#  @param op input operation graph
#  @return ML_Bool operation graph implementing predicates
def fp_mant_is_zero(op):
  op_prec = op.get_precision().get_base_format()
  mant_prec = ML_StdLogicVectorFormat(op_prec.get_field_size())
  mant = SubSignalSelection(
    TypeCast(op, precision = mant_prec),
    0, op_prec.get_field_size() - 1
  )
  return Equal(
    mant,
    Constant(0, precision = mant_prec),
    precision = ML_Bool
  )

## Generates is infinity predicate
#  @param op input operation graph
#  @return ML_Bool operation graph implementing predicates
def fp_is_inf(op):
  return LogicalAnd(
    fp_mant_is_zero(op),
    fp_is_infornan(op),
    precision = ML_Bool
  )

## Generates is NaN predicate
#  @param op input operation graph
#  @return ML_Bool operation graph implementing predicates
def fp_is_nan(op):
  return LogicalAnd(
    LogicalNot(fp_mant_is_zero(op), precision = ML_Bool),
    fp_is_infornan(op),
    precision = ML_Bool
  )

  
