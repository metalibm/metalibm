# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2013-2015)
# All rights reserved
# created:          Jun  5th, 2015
# last-modified:    Jun  5th, 2015
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from ..utility.log_report import *
from .generator_utility import SymbolOperator, FunctionOperator, TemplateOperator, C_Code, Gappa_Code, build_simplified_operator_generation, IdentityOperator, FO_Arg, RoundOperator, type_strict_match, type_relax_match, type_result_match, type_function_match, FunctionObjectOperator, FO_Result
from .code_element import *
from ..core.ml_formats import *
from ..core.ml_complex_formats import ML_Mpfr_t
from ..core.ml_table import ML_ApproxTable
from ..core.ml_operations import *
from ..utility.common import Callable
from metalibm_core.core.target import TargetRegister


from .generic_processor import LibFunctionConstructor, GenericProcessor

Mpfr_Function = LibFunctionConstructor(["mpfr.h"])

mpfr_c_code_generation_table = {
  Abs: {
    None: {
      lambda optree: True: {
          type_strict_match(ML_Mpfr_t, ML_Mpfr_t): Mpfr_Function("mpfr_abs", arity = 1, arg_map = {0: FO_Result(0), 1: FO_Arg(0), 2: "MPFR_RNDN"}),
      }
    }
  },
  Addition: {
    None: {
      lambda optree: True: {
          type_strict_match(ML_Mpfr_t, ML_Mpfr_t, ML_Mpfr_t): Mpfr_Function("mpfr_add", arity = 2, arg_map = {0: FO_Result(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: "MPFR_RNDN"}), 
      }
    }
  },
  Conversion: {
    None: {
      lambda optree: True: {
          type_strict_match(ML_Mpfr_t, ML_Binary32): Mpfr_Function("mpfr_set_flt", arity = 1, arg_map = {0: FO_Result(0), 1: FO_Arg(0), 2: "MPFR_RNDN"}), 
          type_strict_match(ML_Mpfr_t, ML_Binary64): Mpfr_Function("mpfr_set_d", arity = 1, arg_map = {0: FO_Result(0), 1: FO_Arg(0), 2: "MPFR_RNDN"}), 
          type_strict_match(ML_Binary32, ML_Mpfr_t): Mpfr_Function("mpfr_get_flt", arity = 1, arg_map = {0: FO_Arg(0), 1: "MPFR_RNDN"}),
          type_strict_match(ML_Binary64, ML_Mpfr_t): Mpfr_Function("mpfr_get_d", arity = 1, arg_map = {0: FO_Arg(0), 1: "MPFR_RNDN"}),
      },
    },
  },
}
 

class MPFRProcessor(GenericProcessor):
  """ description of MPFR's Backend """
  target_name = "mpfr_backend"
  TargetRegister.register_new_target(target_name, lambda _: MPFRProcessor)


  code_generation_table = {
    C_Code: mpfr_c_code_generation_table,
    Gappa_Code: {}
  }

  def __init__(self):
    GenericProcessor.__init__(self)
    print "initializing MPFR target"
    tab = self.simplified_rec_op_map[C_Code][Conversion][None]
      
