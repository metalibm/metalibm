# -*- coding: utf-8 -*-
# -*- vim: sw=4 sts=4 tw=79

import sys

from implementpoly import implementpoly

from metalibm_core.core.ml_function import (ML_Function, ML_FunctionBasis,
                                            DefaultArgTemplate)
from metalibm_core.utility.ml_template import *
from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.opt.ml_blocks import *
from metalibm_core.code_generation.generic_processor import GenericProcessor
from metalibm_core.targets.common.vector_backend import VectorBackend
from metalibm_core.utility.debug_utils import *

def get_metalibm_precision(formats, target):
  if len(formats) == 1:
    return target
  elif len(formats) == 2:
    if target == ML_Binary32:
      return ML_SingleSingle
    elif target == ML_Binary64:
      return ML_DoubleDouble
  else: # len(formats) is 3
    if target == ML_Binary64:
      return ML_TripleDouble
  return None
#

def implementpoly_multi_node_expand(tree, var, target, limbs, mem_map = {}):
  if tree in mem_map:
    return mem_map[tree]
  elif tree.is_leaf() and tree.name == 'x':
    mem_map[tree] = var
    return var
  elif tree.is_leaf() and tree.name != 'x':
    idx = int(tree.name[1:])
    cst = SollyaObject(limbs[idx][0])
    for i in range(1, len(limbs[idx])):
      cst += SollyaObject(limbs[idx][i])
    coefficient_precision = get_metalibm_precision(limbs[idx], target)
    coefficient = Constant(cst, precision=coefficient_precision, tag="a"+str(idx))
    mem_map[tree] = coefficient
    return coefficient
  else:
    subtree1 = implementpoly_multi_node_expand(tree.operand1, var, target, limbs, mem_map)
    if tree.operand1 != tree.operand2:
      subtree2 = implementpoly_multi_node_expand(tree.operand2, var, target, limbs, mem_map)
    else:
      subtree2 = subtree1
      #
    result = None
    result_precision = get_metalibm_precision(tree.formats, target)
    if tree.operator == '+':
      result = Addition(subtree1, subtree2, precision=result_precision, tag="r"+str(tree.id))
    elif tree.operator == '*':
      result = Multiplication(subtree1, subtree2, precision=result_precision, tag="r"+str(tree.id))
      #
    mem_map[tree] = result
    return result

class ML_ImplementPoly(ML_Function("ml_implementpoly")):
  def __init__(self, args=DefaultArgTemplate):
    # initializing base class
    ML_FunctionBasis.__init__(self, args)
    #
    self.function = sollya.parse(args.function)
    self.interval = sollya.parse(args.interval)
    self.epsilon = 2**(-sollya.parse(str(args.epsilon)))

  @staticmethod
  def get_default_args(**kw):
    """ Return a structure containing the arguments for ML_ImplementPoly """
    default_args_log = {
      "output_file": "POLY.c",
      "function_name": "POLY",
      "precision": ML_Binary64,
      "target": GenericProcessor(),
      "function": None,
      "interval": None,
      "epsilon": None
    }
    default_args_log.update(kw)
    return DefaultArgTemplate(**default_args_log)

  def generate_scheme(self):
    """ Produce an abstract scheme for the polynomial implementation.
        This abstract scheme will be used by the code generation backend.
    """
    x = self.implementation.add_input_variable("x", self.precision)
    #
    [a, s], [limbs, _, _] = implementpoly(self.function, self.interval, None, self.epsilon, \
                                       precision = self.precision.get_precision()+1, binary_formats = [24, 53])
    #
    p = implementpoly_multi_node_expand(s, x, self.precision, limbs, mem_map = {})
    self.implementation.set_output_format(p.precision)
    #
    return Return(p)

  def numeric_emulate(self, input_value):
    return self.function(input_value)

if __name__ == "__main__":
  # auto-test
  arg_template = ML_NewArgTemplate(
          default_arg=ML_ImplementPoly.get_default_args())
  #
  arg_template.get_parser().add_argument('--function', type=str, action='store', dest='function', default=None, required=True, help='function to be implemented')
  arg_template.get_parser().add_argument('--interval', type=str, action='store', dest='interval', default=None, required=True, help='evaluation interval')
  arg_template.get_parser().add_argument('--epsilon', type=int, action='store', dest='epsilon', default=None, required=True, help='required output accuracy (# bits)')
  #
  args = arg_template.arg_extraction()
  #  
  ml_implementpoly = ML_ImplementPoly(args)
  ml_implementpoly.gen_implementation()
