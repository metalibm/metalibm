# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2016)
# All rights reserved
# created:          Feb 2nd, 2016
# last-modified:    Feb 2nd, 2016
#
# description: implement a vector backend for Metalibm
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from ..utility.log_report import *
from .generator_utility import *
from .complex_generator import *
from ..core.ml_formats import *
from ..core.ml_operations import *
from ..utility.common import Callable
from .generic_processor import GenericProcessor, LibFunctionConstructor

from metalibm_core.core.target import TargetRegister


ML_VectorLib_Function = LibFunctionConstructor(["support_lib/ml_vector_lib.h"])

vector_type = {
  ML_Binary32: {
    2: ML_Float2,
    4: ML_Float4,
    8: ML_Float8
  },
  ML_Binary64: {
    2: ML_Double2,
    4: ML_Double4,
    8: ML_Double8
  },
  ML_Int32: {
    2: ML_Int2,
    4: ML_Int4,
    8: ML_Int8
  },
  ML_UInt32: {
    2: ML_UInt2,
    4: ML_UInt4,
    8: ML_UInt8
  },
  ML_Bool: {
    2: ML_Bool2,
    4: ML_Bool4,
    8: ML_Bool8
  },
}
scalar_type_letter = {
  ML_Binary32: "f",
  ML_Binary64: "d",
  ML_UInt32:   "u",
  ML_Int32:    "i",
}

vector_c_code_generation_table = {
  Addition: {
    None: {
       lambda _: True: {
        type_strict_match(ML_Int2, ML_Int2, ML_Int2): ML_VectorLib_Function("ml_vaddi2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2),
        type_strict_match(ML_Int4, ML_Int4, ML_Int4): ML_VectorLib_Function("ml_vaddi4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Int4),
        type_strict_match(ML_Int8, ML_Int8, ML_Int8): ML_VectorLib_Function("ml_vaddi8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Int8),

        type_strict_match(ML_Float2, ML_Float2, ML_Float2): ML_VectorLib_Function("ml_vaddf2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Float2),
        type_strict_match(ML_Float4, ML_Float4, ML_Float4): ML_VectorLib_Function("ml_vaddf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Float4),
        type_strict_match(ML_Float8, ML_Float8, ML_Float8): ML_VectorLib_Function("ml_vaddf8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Float8),
      },
    },
  },
  Subtraction: {
    None: {
       lambda _: True: {
        type_strict_match(ML_Float2, ML_Float2, ML_Float2): ML_VectorLib_Function("ml_vsubf2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Float2),
        type_strict_match(ML_Float4, ML_Float4, ML_Float4): ML_VectorLib_Function("ml_vsubf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Float4),
        type_strict_match(ML_Float8, ML_Float8, ML_Float8): ML_VectorLib_Function("ml_vsubf8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Float8),
      },
    },
  },
  Multiplication: {
    None: {
       lambda _: True: {
        type_strict_match(ML_Float2, ML_Float2, ML_Float2): ML_VectorLib_Function("ml_vmulf2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Float2),
        type_strict_match(ML_Float4, ML_Float4, ML_Float4): ML_VectorLib_Function("ml_vmulf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Float4),
        type_strict_match(ML_Float8, ML_Float8, ML_Float8): ML_VectorLib_Function("ml_vmulf8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Float8),
      },
    },
  },
  FusedMultiplyAdd: {
    FusedMultiplyAdd.Standard: {
       lambda _: True: {
        type_strict_match(ML_Float2, ML_Float2, ML_Float2, ML_Float2): ML_VectorLib_Function("ml_vfmaf2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2)}, arity = 2, output_precision = ML_Float2),
        type_strict_match(ML_Float4, ML_Float4, ML_Float4, ML_Float4): ML_VectorLib_Function("ml_vfmaf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2)}, arity = 2, output_precision = ML_Float4),
        type_strict_match(ML_Float8, ML_Float8, ML_Float8, ML_Float8): ML_VectorLib_Function("ml_vfmaf8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2)}, arity = 2, output_precision = ML_Float8),
      },
    },
  },
  VectorElementSelection: 
  {
    None: {
       lambda _: True: {
        lambda rformat, opformat, indexformat, optree: True: TemplateOperator("%s._[%s]", arity = 2),
      },
    },
  },
  Comparison: 
    #specifier -> 
    dict ((comp_specifier, 
      {
        lambda _: True: 
          dict(
            ( 
              sum(
                [
                  [
                    (
                      type_strict_match(ML_Bool2, vector_type[scalar_type][vector_size], vector_type[scalar_type][vector_size]), 
                      ML_VectorLib_Function("ml_comp_%s_%s%d" % (comp_specifier.opcode, scalar_type_letter[scalar_type], vector_size), arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = vector_type[ML_Bool][vector_size])
                    )  for scalar_type in [ML_Binary32, ML_Binary64, ML_Int32, ML_UInt32]
                  ] for vector_size in [2, 4, 8]
                ], []
              )
            )
          )
      }
    ) for comp_specifier in [Comparison.Equal, Comparison.NotEqual, Comparison.Greater, Comparison.GreaterOrEqual, Comparison.Less, Comparison.LessOrEqual]
  )  ,
  Test: {
    Test.IsMaskAllZero: {
      lambda _: True {
        type_strict_match(ML_Int32, ML_Bool2): ML_VectorLib_Function("ml_is_vmask2_zero", arity = 1, output_precision = ML_Int32), 
        type_strict_match(ML_Int32, ML_Bool4): ML_VectorLib_Function("ml_is_vmask4_zero", arity = 1, output_precision = ML_Int32), 
        type_strict_match(ML_Int32, ML_Bool8): ML_VectorLib_Function("ml_is_vmask8_zero", arity = 1, output_precision = ML_Int32), 
      },
    },
    Test.IsMaskNotAllZero: {
      lambda _: True {
        type_strict_match(ML_Int32, ML_Bool2): ML_VectorLib_Function("ml_is_vmask2_non_zero", arity = 1, output_precision = ML_Int32), 
        type_strict_match(ML_Int32, ML_Bool4): ML_VectorLib_Function("ml_is_vmask4_non_zero", arity = 1, output_precision = ML_Int32), 
        type_strict_match(ML_Int32, ML_Bool8): ML_VectorLib_Function("ml_is_vmask8_non_zero", arity = 1, output_precision = ML_Int32), 
      },
    },
  },
}

vector_gappa_code_generation_table = {
}

class VectorBackend(GenericProcessor):
  target_name = "vector"
  TargetRegister.register_new_target(target_name, lambda _: VectorBackend)

  code_generation_table = {
    C_Code: vector_c_code_generation_table, 
  }
