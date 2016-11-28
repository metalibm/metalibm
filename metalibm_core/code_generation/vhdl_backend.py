# -*- coding: utf-8 -*-

###############################################################################
# This file is part of New Metalibm tool
# Copyrights  Nicolas Brunie (2016)
# All rights reserved
# created:          Nov 17th, 2016
# last-modified:    Nov 17th, 2016
#
# author(s):    Nicolas Brunie (nibrunie@gmail.com)
# description:  Implement a basic VHDL backend for hardware description
#               generation
###############################################################################

from ..utility.log_report import *
from .generator_utility import *
from .code_element import *
from ..core.ml_formats import *
from ..core.ml_hdl_format import *
from ..core.ml_table import ML_ApproxTable
from ..core.ml_operations import *
from ..core.ml_hdl_operations import *
from metalibm_core.core.target import TargetRegister


from .abstract_backend import AbstractBackend

def exclude_std_logic(optree):
  return not isinstance(optree.get_precision(), ML_StdLogicVectorFormat)
def include_std_logic(optree):
  return isinstance(optree.get_precision(), ML_StdLogicVectorFormat)

vhdl_comp_symbol = {
  Comparison.Equal: "=", 
  Comparison.NotEqual: "/=",
  Comparison.Less: "<",
  Comparison.LessOrEqual: "<=",
  Comparison.GreaterOrEqual: ">=",
  Comparison.Greater: ">"
}

# class Match custom std logic vector format
MCSTDLOGICV = TCM(ML_StdLogicVectorFormat)

vhdl_code_generation_table = {
  Addition: {
    None: {
      exclude_std_logic: 
          build_simplified_operator_generation_nomap([ML_Int8, ML_UInt8, ML_Int16, ML_UInt16, ML_Int32, ML_UInt32, ML_Int64, ML_UInt64, ML_Int128,ML_UInt128], 2, SymbolOperator("+", arity = 2), cond = (lambda _: True)),
      include_std_logic:
      {
        type_custom_match(MCSTDLOGICV, MCSTDLOGICV, MCSTDLOGICV):  SymbolOperator("+", arity = 2),
      }
      
    }
  },
  Subtraction: {
    None: {
      exclude_std_logic: 
          build_simplified_operator_generation_nomap([ML_Int8, ML_UInt8, ML_Int16, ML_UInt16, ML_Int32, ML_UInt32, ML_Int64, ML_UInt64, ML_Int128,ML_UInt128], 2, SymbolOperator("-", arity = 2), cond = (lambda _: True)),
      include_std_logic:
      {
        type_custom_match(MCSTDLOGICV, MCSTDLOGICV, MCSTDLOGICV):  SymbolOperator("-", arity = 2),
      }
      
    }
  },
  LogicalAnd: {
    None: {
      lambda _: True : {
        type_strict_match(ML_Bool, ML_Bool, ML_Bool): SymbolOperator("and", arity = 2, force_folding = True),
      },
    }
  },
  Event: {
    None: {
      lambda _: True : {
        type_strict_match(ML_Bool, ML_StdLogic): SymbolOperator("\'event", lspace = "", inverse = True, arity = 1, force_folding = True), 
      },
    },
  },
  Comparison: 
      dict (
        (specifier,
          { 
              lambda _: True: {
                  type_custom_match(FSM(ML_Bool), TCM(ML_StdLogicVectorFormat), TCM(ML_StdLogicVectorFormat)): 
                    SymbolOperator(vhdl_comp_symbol[specifier], arity = 2, force_folding = True),
                  type_strict_match(ML_Bool, ML_StdLogic, ML_StdLogic):
                    SymbolOperator(vhdl_comp_symbol[specifier], arity = 2, force_folding = True),
              },
              #build_simplified_operator_generation([ML_Int32, ML_Int64, ML_UInt64, ML_UInt32, ML_Binary32, ML_Binary64], 2, SymbolOperator(">=", arity = 2), result_precision = ML_Int32),
          }) for specifier in [Comparison.Equal, Comparison.NotEqual, Comparison.Greater, Comparison.GreaterOrEqual, Comparison.Less, Comparison.LessOrEqual]
  ),
}
 

class VHDLBackend(AbstractBackend):
  """ description of MPFR's Backend """
  target_name = "vhdl_backend"
  TargetRegister.register_new_target(target_name, lambda _: VHDLBackend)


  code_generation_table = {
    VHDL_Code: vhdl_code_generation_table,
    Gappa_Code: {}
  }

  def __init__(self):
    AbstractBackend.__init__(self)
    print "initializing MPFR target"
      
