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
