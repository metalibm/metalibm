# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/kalray/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2018 Kalray
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################
# created:          Feb  2nd, 2016
# last-modified:    Mar  7th, 2018
#
# description: implement a vector backend for Metalibm
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################


from metalibm_core.utility.log_report import Log

from metalibm_core.core.target import UniqueTargetDecorator
from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_table import ML_TableFormat, ML_Pointer_Format
from metalibm_core.core.ml_operations import *
from metalibm_core.core.static_vectorizer import StaticVectorizer
from metalibm_core.core.legalizer import (
    min_legalizer, max_legalizer
)
from metalibm_core.core.multi_precision import (
    legalize_multi_precision_vector_element_selection
)

from metalibm_core.code_generation.generator_utility import *
from metalibm_core.code_generation.generator_utility import MatchResult, is_impl_list, ImplemList
from metalibm_core.code_generation.complex_generator import *
from metalibm_core.code_generation.abstract_backend import LOG_BACKEND_INIT
from metalibm_core.code_generation.generic_processor import GenericProcessor, LibFunctionConstructor



ML_VectorLib_Function = LibFunctionConstructor(["ml_support_lib.h"])

# OpenCL vector support library
ML_OCL_VectorLib_Function = LibFunctionConstructor(["support_lib/ml_ocl_vector_lib.h"])

OpenCL_Builtin = LibFunctionConstructor([])

scalar_type_letter = {
  ML_Binary32: "f",
  ML_Binary64: "d",
  ML_UInt32:   "u",
  ML_Int32:    "i",
  ML_UInt64:   "ul",
  ML_Int64:    "l",
}

COMPARATOR_SYMBOL_MAP = {
    Comparison.Equal: "==",
    Comparison.NotEqual: "==",
    Comparison.LessOrEqual: "<=",
    Comparison.Less: "<",
    Comparison.Greater: ">",
    Comparison.GreaterOrEqual: ">="
}

supported_vector_size = [2, 3, 4, 8]

VECTOR_BOOLEAN_FORMAT_LIST = [
    v2bool, v3bool, v4bool, v8bool,
    v2lbool, v3lbool, v4lbool, v8lbool
]


def promote_operand(op_index, precision):
    """ Promote operand operand with index <op_index>
        to precision <precision> """
    def promote(optree):
        shift_amount = optree.get_input(op_index)
        optree.set_input(
            op_index,
            Conversion(
                shift_amount,
                precision=precision
            )
        )
        return optree
    return promote

## Predicate to test if a VectorElementSelection
#  is legal for the vector_backend_target, i.e.
#  vector operand precision is a compound format
def legal_vector_element_selection(optree):
  compound_format = isinstance(
    optree.get_input(0).get_precision(),
    ML_CompoundVectorFormat
  )
  multi_precision_scalar = isinstance(
    optree.get_input(0).get_precision().get_scalar_format(),
    ML_FP_MultiElementFormat
  )
  return compound_format and not multi_precision_scalar

vector_opencl_code_generation_table = {
  BitLogicLeftShift: {
    None: {
      lambda _: True:
        dict(
          sum(
          [
            [
              (type_strict_match(
                  VECTOR_TYPE_MAP[scalar_type][vector_size],
                  VECTOR_TYPE_MAP[scalar_type][vector_size],
                  VECTOR_TYPE_MAP[scalar_type][vector_size]
                ), SymbolOperator(" << ", arity = 2)
              ) for vector_size in supported_vector_size
            ] for scalar_type in [ ML_Int32, ML_UInt32 ]
          ], [])
        )
     }
  },
  BitLogicRightShift: {
    None: {
      lambda _: True:
        dict(
          sum(
          [
                [
                  (type_strict_match(
                      VECTOR_TYPE_MAP[scalar_type][vector_size],
                      VECTOR_TYPE_MAP[scalar_type][vector_size],
                      VECTOR_TYPE_MAP[scalar_type][vector_size]
                    ), SymbolOperator(" >> ", arity = 2)
                  ) for vector_size in supported_vector_size
                ] for scalar_type in [ ML_Int32, ML_UInt32]
          ], [])
        )
     }
  },
  BitArithmeticRightShift: {
    None: {
      lambda _: True:
        dict(
          sum(
          [
            [
              (type_strict_match(
                  VECTOR_TYPE_MAP[scalar_type][vector_size],
                  VECTOR_TYPE_MAP[scalar_type][vector_size],
                  VECTOR_TYPE_MAP[scalar_type][vector_size]
                ), SymbolOperator(" >> ", arity = 2)
              ) for vector_size in supported_vector_size
            ] for scalar_type in [ ML_Int32 ]
          ], [])
        )
     }
  },
  Addition: {
    None: {
      lambda _: True:
        dict(
          sum(
          [
            [
              (type_strict_match(
                  VECTOR_TYPE_MAP[scalar_type][vector_size],
                  VECTOR_TYPE_MAP[scalar_type][vector_size],
                  VECTOR_TYPE_MAP[scalar_type][vector_size]
                ), SymbolOperator("+", arity = 2)
              ) for vector_size in supported_vector_size
            ] for scalar_type in [ML_Binary32, ML_Binary64, ML_Int32, ML_UInt32]
          ], [])
        )
     }
  },
  Subtraction: {
    None: {
      lambda _: True:
        dict(
          sum(
          [
            [
              (type_strict_match(
                  VECTOR_TYPE_MAP[scalar_type][vector_size],
                  VECTOR_TYPE_MAP[scalar_type][vector_size],
                  VECTOR_TYPE_MAP[scalar_type][vector_size]
                ), SymbolOperator("-", arity = 2)
              ) for vector_size in supported_vector_size
            ] for scalar_type in [ML_Binary32, ML_Binary64, ML_Int32, ML_UInt32]
          ], [])
        )
     }
  },
  BitLogicAnd: {
    None: {
      lambda _: True:
        dict(
          sum(
          [
            [
              (type_strict_match(
                  VECTOR_TYPE_MAP[scalar_type][vector_size],
                  VECTOR_TYPE_MAP[scalar_type][vector_size],
                  VECTOR_TYPE_MAP[scalar_type][vector_size]
                ), SymbolOperator("&", arity = 2)
              ) for vector_size in supported_vector_size
            ] for scalar_type in [ML_Binary32, ML_Binary64, ML_Int32, ML_UInt32]
          ], [])
        )
     }
  },
  BitLogicOr: {
    None: {
      lambda _: True:
        dict(
          sum(
          [
            [
              (type_strict_match(
                  VECTOR_TYPE_MAP[scalar_type][vector_size],
                  VECTOR_TYPE_MAP[scalar_type][vector_size],
                  VECTOR_TYPE_MAP[scalar_type][vector_size]
                ), SymbolOperator("|", arity = 2)
              ) for vector_size in supported_vector_size
            ] for scalar_type in [ML_Binary32, ML_Binary64, ML_Int32, ML_UInt32]
          ], [])
        )
     }
  },
  Division: {
    None: {
      lambda _: True:
        dict(
          sum(
          [
            [
              (type_strict_match(
                  VECTOR_TYPE_MAP[scalar_type][vector_size],
                  VECTOR_TYPE_MAP[scalar_type][vector_size],
                  VECTOR_TYPE_MAP[scalar_type][vector_size]
                ), SymbolOperator("/", arity = 2)
              ) for vector_size in supported_vector_size
            ] for scalar_type in [ML_Binary32, ML_Binary64,
                                  ML_Int32, ML_UInt32,
                                  ML_Int64, ML_UInt64]
          ], [])
        )
     }
  },
  Modulo: {
    None: {
      lambda _: True:
        dict(
          sum(
          [
            [
              (type_strict_match(
                  VECTOR_TYPE_MAP[scalar_type][vector_size],
                  VECTOR_TYPE_MAP[scalar_type][vector_size],
                  VECTOR_TYPE_MAP[scalar_type][vector_size]
                ), SymbolOperator("%", arity = 2)
              ) for vector_size in supported_vector_size
            ] for scalar_type in [ML_Binary32, ML_Binary64, ML_Int32, ML_UInt32]
          ], [])
        )
     }
  },
  Select: {
    None: {
      lambda _: True:
        dict(
          sum(
          [
            [
              (type_strict_match(
                  VECTOR_TYPE_MAP[scalar_type][vector_size],
                  VECTOR_TYPE_MAP[ML_Bool][scalar_type.get_bit_size()][vector_size],
                  VECTOR_TYPE_MAP[scalar_type][vector_size],
                  VECTOR_TYPE_MAP[scalar_type][vector_size]
                ), TemplateOperator("%s ? %s : %s", arity = 3)
              ) for vector_size in supported_vector_size
            ] for scalar_type in [ML_Binary32, ML_Binary64,
                                  ML_Int32, ML_UInt32,
                                  ML_Int64, ML_UInt64]
          ], [])
        )
     }
  },
  Multiplication: {
    None: {
      lambda _: True:
        dict(
          sum(
          [
            [
              (type_strict_match(
                  VECTOR_TYPE_MAP[scalar_type][vector_size],
                  VECTOR_TYPE_MAP[scalar_type][vector_size],
                  VECTOR_TYPE_MAP[scalar_type][vector_size]
                ), SymbolOperator("*", arity = 2)
              ) for vector_size in supported_vector_size
            ] for scalar_type in [ML_Binary32, ML_Binary64, ML_Int32, ML_UInt32]
          ], [])
        )
     }
  },
  Negation: {
    None: {
      lambda _: True:
        dict(
          sum(
          [
            [
              (type_strict_match(
                  VECTOR_TYPE_MAP[scalar_type][vector_size],
                  VECTOR_TYPE_MAP[scalar_type][vector_size]
                ), SymbolOperator("-", arity = 1)
              ) for vector_size in supported_vector_size
            ] for scalar_type in [ML_Binary32, ML_Binary64, ML_Int32, ML_UInt32]
          ], [])
        )
     }
  },
  FusedMultiplyAdd: {
    FusedMultiplyAdd.Standard: {
      lambda _: True:
        dict(
          sum(
          [
            [
              (type_strict_match(
                  VECTOR_TYPE_MAP[scalar_type][vector_size],
                  VECTOR_TYPE_MAP[scalar_type][vector_size],
                  VECTOR_TYPE_MAP[scalar_type][vector_size],
                  VECTOR_TYPE_MAP[scalar_type][vector_size]
                ), OpenCL_Builtin("fma", arity = 3),
              ) for vector_size in supported_vector_size
            ] for scalar_type in [ML_Binary32, ML_Binary64, ML_Int32, ML_UInt32]
          ], [])
        )
     }
  },
  NearestInteger: {
    None: {
      lambda _: True : {
        type_strict_match(v2int32, v2float32): OpenCL_Builtin("nearbyint", arity = 1, output_precision = v2int32),
        type_strict_match(v3int32, v3float32): OpenCL_Builtin("nearbyint", arity = 1, output_precision = v3int32),
        type_strict_match(v4int32, v4float32): OpenCL_Builtin("nearbyint", arity = 1, output_precision = v4int32),
        type_strict_match(v8int32, v8float32): OpenCL_Builtin("nearbyint", arity = 1, output_precision = v8int32),

        type_strict_match(v2float32, v2float32): OpenCL_Builtin("rint", arity = 1, output_precision = v2float32),
        type_strict_match(v3float32, v3float32): OpenCL_Builtin("rint", arity = 1, output_precision = v3float32),
        type_strict_match(v4float32, v4float32): OpenCL_Builtin("rint", arity = 1, output_precision = v4float32),
        type_strict_match(v8float32, v8float32): OpenCL_Builtin("rint", arity = 1, output_precision = v8float32),
      }
    }
  },
  Trunc: {
    None: {
      lambda _: True : {
        type_strict_match(ML_Binary32, ML_Binary32): OpenCL_Builtin("Trunc", arity = 1, output_precision = ML_Binary32),
        type_strict_match(ML_Binary64, ML_Binary64): OpenCL_Builtin("Trunc", arity = 1, output_precision = ML_Binary64),
      }
    }
  },
  Ceil: {
    None: {
      lambda _: True : {
        type_strict_match(ML_Binary32, ML_Binary32): OpenCL_Builtin("ceil", arity = 1, output_precision = ML_Binary32),
        type_strict_match(ML_Binary64, ML_Binary64): OpenCL_Builtin("ceil", arity = 1, output_precision = ML_Binary64),
      }
    }
  },
  Floor: {
    None: {
      lambda _: True : {
        type_strict_match(ML_Binary32, ML_Binary32): OpenCL_Builtin("floor", arity = 1, output_precision = ML_Binary32),
        type_strict_match(ML_Binary64, ML_Binary64): OpenCL_Builtin("floor", arity = 1, output_precision = ML_Binary64),
      }
    }
  },
  Test: {
    Test.IsNaN: {
        lambda optree: True: {
            type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary32]): OpenCL_Builtin("isnan", arity = 1),
            type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary64]): OpenCL_Builtin("isnan", arity = 1),
        },
    },
    Test.IsInfty: {
        lambda optree: True: {
            type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary32]): OpenCL_Builtin("isinf", arity = 1),
            type_strict_match_list([ML_Int32, ML_Bool], [ML_Binary64]): OpenCL_Builtin("isinf", arity = 1),
        },
    },
  },
  VectorElementSelection: {
    None: {
        # make sure index accessor is a Constant (or fallback to C implementation)
       lambda optree: isinstance(optree.get_input(1), Constant):  {
        lambda rformat, opformat, indexformat, optree: True: TemplateOperator("%s.s%s", arity = 2),
      },
    },
  },
  LogicalNot: {
    None: {
      lambda _: True: {
        type_strict_match(v2bool, v2bool): SymbolOperator("!", arity = 1),
        type_strict_match(v3bool, v3bool): SymbolOperator("!", arity = 1),
        type_strict_match(v4bool, v4bool): SymbolOperator("!", arity = 1),
        type_strict_match(v8bool, v8bool): SymbolOperator("!", arity = 1),
      },
    },
  },
  LogicalAnd: {
    None: {
      lambda _: True: {
        type_strict_match(v2bool, v2bool, v2bool): SymbolOperator("&&", arity = 2),
        type_strict_match(v3bool, v3bool, v3bool): SymbolOperator("&&", arity = 2),
        type_strict_match(v4bool, v4bool, v4bool): SymbolOperator("&&", arity = 2),
        type_strict_match(v8bool, v8bool, v8bool): SymbolOperator("&&", arity = 2),
      },
    },
  },
  LogicalOr: {
    None: {
      lambda _: True: dict(
        (
            type_strict_match(vbool_format, vbool_format, vbool_format),
            SymbolOperator("||", arity = 2)
        ) for vbool_format in VECTOR_BOOLEAN_FORMAT_LIST
       )
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
                      type_strict_match_list(
                        [
                          VECTOR_TYPE_MAP[ML_Bool][scalar_type.get_bit_size()][vector_size]
                        ],
                        [
                          VECTOR_TYPE_MAP[scalar_type][vector_size]
                        ],
                        [
                          VECTOR_TYPE_MAP[scalar_type][vector_size]
                        ]
                      )
                      ,
                      SymbolOperator(comp_specifier.symbol, arity = 2),
                    )  for scalar_type in [ML_Binary32, ML_Binary64, ML_Int32, ML_UInt32]
                  ] for vector_size in supported_vector_size
                ], []
              )
            )
          )
      }
    ) for comp_specifier in [Comparison.Equal, Comparison.NotEqual, Comparison.Greater, Comparison.GreaterOrEqual, Comparison.Less, Comparison.LessOrEqual]
  ),
  TypeCast: {
      None: {
          lambda optree: True: {
              type_strict_match(ML_Binary32, ML_Int32) : ML_VectorLib_Function("as_float", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = ML_Binary32),
              type_strict_match(ML_Int32, ML_Binary32) : ML_VectorLib_Function("as_int", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = ML_Int32),

              type_strict_match(v2float32, v2int32) : ML_VectorLib_Function("as_float2", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = v2float32),
              type_strict_match(v2int32, v2float32) : ML_VectorLib_Function("as_int2", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = v2int32),

              type_strict_match(v3float32, v3int32) : ML_VectorLib_Function("as_float3", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = v3float32),
              type_strict_match(v3int32, v3float32) : ML_VectorLib_Function("as_int3", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = v3int32),

              type_strict_match(v4float32, v4int32) : ML_VectorLib_Function("as_float4", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = v4float32),
              type_strict_match(v4int32, v4float32) : ML_VectorLib_Function("as_int4", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = v4int32),

              type_strict_match(v8float32, v8int32) : ML_VectorLib_Function("as_float8", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = v8float32),
              type_strict_match(v8int32, v8float32) : ML_VectorLib_Function("as_int8", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = v8int32),

              type_strict_match(v2uint32, v2float32) : ML_VectorLib_Function("as_uint2", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = v2int32),
              type_strict_match(v3uint32, v3float32) : ML_VectorLib_Function("as_uint3", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = v3int32),
              type_strict_match(v4uint32, v4float32) : ML_VectorLib_Function("as_uint4", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = v4int32),
              type_strict_match(v8uint32, v8float32) : ML_VectorLib_Function("as_uint8", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = v8int32),

  }}},
  ExponentExtraction: {
      None: {
          lambda optree: True: {
              type_strict_match(v2int32, v2float32) : ML_OCL_VectorLib_Function("ml_ocl_exp_extraction_vf2", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = v4int32),
              type_strict_match(v3int32, v3float32) : ML_OCL_VectorLib_Function("ml_ocl_exp_extraction_vf3", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = v3int32),
              type_strict_match(v4int32, v4float32) : ML_OCL_VectorLib_Function("ml_ocl_exp_extraction_vf4", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = v4int32),
              type_strict_match(v8int32, v8float32) : ML_OCL_VectorLib_Function("ml_ocl_exp_extraction_vf8", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = v8int32),
  }}},
  ExponentInsertion: {
    ExponentInsertion.Default: {
          lambda optree: True: {
              type_strict_match(v2float32, v2int32) : ML_OCL_VectorLib_Function("ml_ocl_exp_insertion_vf2", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = v2int32),
              type_strict_match(v3float32, v3int32) : ML_OCL_VectorLib_Function("ml_ocl_exp_insertion_vf3", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = v3int32),
              type_strict_match(v4float32, v4int32) : ML_OCL_VectorLib_Function("ml_ocl_exp_insertion_vf4", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = v4int32),
              type_strict_match(v8float32, v8int32) : ML_OCL_VectorLib_Function("ml_ocl_exp_insertion_vf8", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = v8int32),
  }}},
  Conversion: {
    None: {
      lambda optree: True: {
        type_strict_match(v2uint32, v2int32): ML_VectorLib_Function("convert_uint2", arity = 1, output_precision = v2uint32),
        type_strict_match(v2int32, v2uint32): ML_VectorLib_Function("convert_int2", arity = 1, output_precision = v2int32),
        type_strict_match(v3uint32, v3int32): ML_VectorLib_Function("convert_uint2", arity = 1, output_precision = v3uint32),
        type_strict_match(v3int32, v3uint32): ML_VectorLib_Function("convert_int2", arity = 1, output_precision = v3int32),
        type_strict_match(v4uint32, v4int32): ML_VectorLib_Function("convert_uint2", arity = 1, output_precision = v4uint32),
        type_strict_match(v4int32, v4uint32): ML_VectorLib_Function("convert_int2", arity = 1, output_precision = v4int32),
        type_strict_match(v8uint32, v8int32): ML_VectorLib_Function("convert_uint2", arity = 1, output_precision = v8uint32),
        type_strict_match(v8int32, v8uint32): ML_VectorLib_Function("convert_int2", arity = 1, output_precision = v8int32),
      },
    },
  },
  CountLeadingZeros: {
    None: {
      lambda optree: True: {
        type_strict_match(v2uint32, v2uint32): ML_VectorLib_Function("clz", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = v2uint32),
        type_strict_match(v2int32, v2int32): ML_VectorLib_Function("clz", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = v2int32),
        type_strict_match(v3uint32, v3uint32): ML_VectorLib_Function("clz", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = v3uint32),
        type_strict_match(v3int32, v3int32): ML_VectorLib_Function("clz", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = v3int32),
        type_strict_match(v4uint32, v4uint32): ML_VectorLib_Function("clz", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = v4uint32),
        type_strict_match(v4int32, v4int32): ML_VectorLib_Function("clz", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = v4int32),
        type_strict_match(v8uint32, v8uint32): ML_VectorLib_Function("clz", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = v8uint32),
        type_strict_match(v8int32, v8int32): ML_VectorLib_Function("clz", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = v8int32),
      },
    },
  },

  Test: {
    Test.IsMaskAllZero: {
      lambda _: True: {
        type_strict_match_list([ML_Bool, ML_Int32], [v2bool]):
            ML_OCL_VectorLib_Function("ml_ocl_is_vmask2_zero", arity = 1, output_precision = ML_Int32),
        type_strict_match_list([ML_Bool, ML_Int32], [v3bool]):
            ML_OCL_VectorLib_Function("ml_ocl_is_vmask3_zero", arity = 1, output_precision = ML_Int32),
        type_strict_match_list([ML_Bool, ML_Int32], [v4bool]):
            ML_OCL_VectorLib_Function("ml_ocl_is_vmask4_zero", arity = 1, output_precision = ML_Int32),
        type_strict_match_list([ML_Bool, ML_Int32], [v8bool]):
            ML_OCL_VectorLib_Function("ml_ocl_is_vmask8_zero", arity = 1, output_precision = ML_Int32),
      },
    },
    Test.IsMaskAnyZero: {
      lambda _: True: {
        type_strict_match_list([ML_Bool, ML_Int32], [v2bool]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask2_any_zero", arity = 1, output_precision = ML_Int32),
        type_strict_match_list([ML_Bool, ML_Int32], [v3bool]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask3_any_zero", arity = 1, output_precision = ML_Int32),
        type_strict_match_list([ML_Bool, ML_Int32], [v4bool]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask4_any_zero", arity = 1, output_precision = ML_Int32),
        type_strict_match_list([ML_Bool, ML_Int32], [v8bool]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask8_any_zero", arity = 1, output_precision = ML_Int32),
      },
    },
    Test.IsMaskNotAnyZero: {
      lambda _: True: {
        type_strict_match_list([ML_Bool, ML_Int32], [v2bool]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask2_not_any_zero", arity = 1, output_precision = ML_Int32),
        type_strict_match_list([ML_Bool, ML_Int32], [v3bool]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask3_not_any_zero", arity = 1, output_precision = ML_Int32),
        type_strict_match_list([ML_Bool, ML_Int32], [v4bool]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask4_not_any_zero", arity = 1, output_precision = ML_Int32),
        type_strict_match_list([ML_Bool, ML_Int32], [v8bool]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask8_not_any_zero", arity = 1, output_precision = ML_Int32),
      },
    },
    Test.IsMaskNotAllZero: {
      lambda _: True: {
        type_strict_match_list([ML_Bool, ML_Int32], [v2bool]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask2_not_all_zero", arity = 1, output_precision = ML_Int32),
        type_strict_match_list([ML_Bool, ML_Int32], [v3bool]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask3_not_all_zero", arity = 1, output_precision = ML_Int32),
        type_strict_match_list([ML_Bool, ML_Int32], [v4bool]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask4_not_all_zero", arity = 1, output_precision = ML_Int32),
        type_strict_match_list([ML_Bool, ML_Int32], [v8bool]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask8_not_all_zero", arity = 1, output_precision = ML_Int32),
      },
    },
    Test.IsInfOrNaN: {
      lambda _: True: {
        type_strict_match(v2bool, v2float32): ML_OCL_VectorLib_Function("ml_ocl_vtestf2_is_nan_or_inf", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v2int32),
        type_strict_match(v4bool, v3float32): ML_OCL_VectorLib_Function("ml_ocl_vtestf4_is_nan_or_inf", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v3int32),
        type_strict_match(v4bool, v4float32): ML_OCL_VectorLib_Function("ml_ocl_vtestf4_is_nan_or_inf", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v4int32),
        type_strict_match(v8bool, v8float32): ML_OCL_VectorLib_Function("ml_ocl_vtestf8_is_nan_or_inf", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v8int32),
      },
    }
  },
}

def type_uniform_op2_match(result_type, op0_type, op1_type, **kw):
    """ Type match predicates:
            a 2-operand where operands and result format must
            match
        """
    return result_type == op0_type and op0_type == op1_type

def type_uniform_op2_match_restrain(list_t):
    """ Type matching predicate which check that a 2-operand operation
        has identical type for both operand and results and that this type
        is in <list_t> """
    def local_match(result_type, op0_t, op1_t, **kw):
        return type_uniform_op2_match(result_type, op0_t, op1_t, **kw) and result_type in list_t
    return local_match


def assemble_vector(scalar_results, vector_prec, threshold=4):
    """ build a vector node from list scalar_results,
        sub-dividing in half, if vectors exceeds threshold """
    vector_size = vector_prec.get_vector_size()
    if vector_size <= threshold:
        return VectorAssembling(*tuple(scalar_results), precision=vector_prec)
    else:
        n = len(scalar_results)
        hi_l = int((n+1)/2)
        lo_l = n - hi_l
        scalar_prec = vector_prec.get_scalar_format()
        hi_format = VECTOR_TYPE_MAP[scalar_prec][hi_l]
        lo_format = VECTOR_TYPE_MAP[scalar_prec][lo_l]
        return VectorAssembling(
            VectorAssembling(*tuple(scalar_results[:lo_l]),  precision=lo_format),
            VectorAssembling(*tuple(scalar_results[lo_l:]), precision=hi_format),
            precision=vector_prec
        )

def unroll_vector_class(op_class):
    """ unroll the vector operation optree on n-element
        as n scalar operation """
    def unroll_vector_node(optree):
        vector_prec = optree.get_precision()
        vsize = vector_prec.get_vector_size()
        out_prec = vector_prec.get_scalar_format()
        scalar_ops = [
            [
                VectorElementSelection(
                    op_i, Constant(elt_j, precision=ML_Integer),
                    precision=op_i.get_precision().get_scalar_format()
                )  for op_i in optree.get_inputs()
            ] for elt_j in range(vsize)
        ]
        scalar_results = [op_class(*tuple(scalar_ops[i]), precision=out_prec) for i in range(vsize)]
        return assemble_vector(scalar_results, vector_prec)
    return unroll_vector_node

""" List of standard vector format supported by the common.vector_backend """
SUPPORTED_VECTOR_FORMATS = [
    v2float32, v3float32, v4float32, v8float32,
    v2float64, v3float64, v4float64, v8float64,
    v2int32, v3int32, v4int32, v8int32,
    v2int64, v3int64, v4int64, v8int64,
    v2uint32, v3uint32, v4uint32, v8uint32,
    v2uint64, v3uint64, v4uint64, v8uint64,
]


def get_vformat_desc(vformat):
    return {
        ML_Int32: "i",
        ML_Int64: "l",
        ML_UInt32: "u",
        ML_UInt64: "ul",
        ML_Binary32: "f",
        ML_Binary64: "d"
    }[vformat.get_scalar_format()]

BASIC_INTEGER_VFORMAT_LIST = [
    v2int32, v3int32, v4int32, v8int32,
    v2uint32, v3uint32, v4uint32, v8uint32,
    v2int64, v3int64, v4int64, v8int64,
    v2uint64, v3uint64, v4uint64, v8uint64,
]

BASIC_FLOAT_VFORMAT_LIST = [
    v2float32, v3float32, v4float32, v8float32,
    v2float64, v3float64, v4float64, v8float64,
]

BASIC_VFORMAT_LIST = BASIC_FLOAT_VFORMAT_LIST + BASIC_INTEGER_VFORMAT_LIST

def predicate_vector_broadcast(result_format, input_format, **kw):
    """ Predicate to match valid VectorBackend """
    return result_format.get_scalar_format() == input_format 

def expand_vector_broadcast(node):
    """ Expand a VectorBroadcast note into the valid VectorAssembling """
    assert isinstance(node, VectorBroadcast)
    scalar_input = node.get_input(0)
    output_vsize = node.get_precision().get_vector_size()
    return VectorAssembling(*tuple(scalar_input for _ in range(output_vsize)), precision=node.get_precision())

def vector_lib_helper(func_name, arity):
    """ Helper to describe a vector lib function taking as 1st
        argument a reference to its results followed by an ordered list
        of arguments """
    args_dict = {(i+1): FO_Arg(i) for i in range(arity)}
    args_dict[0] = FO_ResultRef(0)
    return ML_VectorLib_Function(func_name, arg_map=args_dict, arity=arity)

vector_c_code_generation_table = {
  ReciprocalSeed: {
    None: {
        lambda _: True: {
            type_strict_match(v2float32, v2float32):
                ComplexOperator(optree_modifier=unroll_vector_class(ReciprocalSeed)),
            type_strict_match(v3float32, v3float32):
                ComplexOperator(optree_modifier=unroll_vector_class(ReciprocalSeed)),
            type_strict_match(v4float32, v4float32):
                ComplexOperator(optree_modifier=unroll_vector_class(ReciprocalSeed)),
            type_strict_match(v8float32, v8float32):
                ComplexOperator(optree_modifier=unroll_vector_class(ReciprocalSeed)),
            # double precision
            type_strict_match(v4float64, v4float64):
                ComplexOperator(optree_modifier=unroll_vector_class(ReciprocalSeed)),
            type_strict_match(v8float64, v8float64):
                ComplexOperator(optree_modifier=unroll_vector_class(ReciprocalSeed)),
        }
    },
  },
  ReciprocalSquareRootSeed: {
    None: {
        lambda _: True: {
            type_strict_match(v2float32, v2float32):
                ComplexOperator(optree_modifier=unroll_vector_class(ReciprocalSquareRootSeed)),
            type_strict_match(v3float32, v3float32):
                ComplexOperator(optree_modifier=unroll_vector_class(ReciprocalSquareRootSeed)),
            type_strict_match(v4float32, v4float32):
                ComplexOperator(optree_modifier=unroll_vector_class(ReciprocalSquareRootSeed)),
            type_strict_match(v8float32, v8float32):
                ComplexOperator(optree_modifier=unroll_vector_class(ReciprocalSquareRootSeed)),
            # double precision
            type_strict_match(v4float64, v4float64):
                ComplexOperator(optree_modifier=unroll_vector_class(ReciprocalSquareRootSeed)),
            type_strict_match(v8float64, v8float64):
                ComplexOperator(optree_modifier=unroll_vector_class(ReciprocalSquareRootSeed)),
        }
    },
  },
  SubVectorExtract: {
    None: {
        lambda _: True: {
            type_strict_match(v2float32, v4float32, ML_Integer, ML_Integer):
                ML_VectorLib_Function("ml_sub_vec_4to2_extract", arity=3),
        },
    },
  },
  VectorAssembling: {
    None: {
      lambda _: True: {
        type_strict_match(v2float32, ML_Binary32, ML_Binary32): ML_VectorLib_Function("ml_vec_assembling_1_2_float", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2),
        type_strict_match(v2int32, ML_Int32, ML_Int32): ML_VectorLib_Function("ml_vec_assembling_1_2_int", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2),
        type_strict_match(v2bool, ML_Bool, ML_Bool): ML_VectorLib_Function("ml_vec_assembling_1_2_bool", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2),
        type_strict_match(v2float64, ML_Binary64, ML_Binary64):
            ML_VectorLib_Function("ml_vec_assembling_1_2_double", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity=2),

        type_strict_match(v4float32, ML_Binary32, ML_Binary32, ML_Binary32, ML_Binary32): ML_VectorLib_Function("ml_vec_assembling_1_4_float", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: FO_Arg(3)}, arity = 4),
        type_strict_match(v4int32, ML_Int32, ML_Int32, ML_Int32, ML_Int32): ML_VectorLib_Function("ml_vec_assembling_1_4_int", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: FO_Arg(3)}, arity = 4),
        type_strict_match(v4bool, ML_Bool, ML_Bool, ML_Bool, ML_Bool): ML_VectorLib_Function("ml_vec_assembling_1_4_bool", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: FO_Arg(3)}, arity = 4),

        type_strict_match(v4float32, v2float32, v2float32): ML_VectorLib_Function("ml_vec_assembling_2_4_float", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2),
        type_strict_match(v4int32, v2int32, v2int32): ML_VectorLib_Function("ml_vec_assembling_2_4_int", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2),
        type_strict_match(v4bool, v2bool, v2bool): ML_VectorLib_Function("ml_vec_assembling_2_4_bool", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2),

        type_strict_match(v8float32, v4float32, v4float32): ML_VectorLib_Function("ml_vec_assembling_4_8_float", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2),
        type_strict_match(v8int32, v4int32, v4int32): ML_VectorLib_Function("ml_vec_assembling_4_8_int", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2),
        type_strict_match(v8bool, v4bool, v4bool): ML_VectorLib_Function("ml_vec_assembling_4_8_bool", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2),

        type_strict_match(v8lbool, v4lbool, v4lbool):
            ML_VectorLib_Function("ml_vec_assembling_4_8_lbool", arg_map={0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2),

        type_strict_match(v8float32, v2float32, v2float32, v2float32, v2float32): ML_VectorLib_Function("ml_vec_assembling_2_8_float", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: FO_Arg(3)}, arity = 4),
        type_strict_match(v8int32, v2int32, v2int32, v2int32, v2int32): ML_VectorLib_Function("ml_vec_assembling_2_8_int", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: FO_Arg(3)}, arity = 4),
        type_strict_match(v8bool, v2bool, v2bool, v2bool, v2bool): ML_VectorLib_Function("ml_vec_assembling_2_8_bool", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: FO_Arg(3)}, arity = 4),

        type_strict_match(*((v8float32,) + (ML_Binary32,)*8)):
            vector_lib_helper("ml_vec_assembling_1_8_float", arity=8),
        type_strict_match(*((v8bool,) + (ML_Bool,)*8)):
            vector_lib_helper("ml_vec_assembling_1_8_bool", arity=8),

        type_strict_match(v4float64, ML_Binary64, ML_Binary64, ML_Binary64, ML_Binary64):
            vector_lib_helper("ml_vec_assembling_1_4_double", arity=4),
        type_strict_match(*((v8float64,) + (ML_Binary64,)*8)):
            vector_lib_helper("ml_vec_assembling_1_8_double", arity=8),
        type_strict_match(v8float64, v4float64, v4float64):
            ML_VectorLib_Function("ml_vec_assembling_4_8_double", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2),

        type_strict_match(v4int64, ML_Int64, ML_Int64, ML_Int64, ML_Int64):
            vector_lib_helper("ml_vec_assembling_1_4_long", arity=4),
      },
    },
  },
  VectorBroadcast: {
    None: {
        lambda _: True: {
            predicate_vector_broadcast:
                ComplexOperator(optree_modifier=expand_vector_broadcast),
        },
    },
  },
  TableLoad: {
    None: {
      lambda _: True: {
        # single precision loading (gather)
        type_custom_match(FSM(v2float32), TCM(ML_TableFormat), FSM(v2int32)):
            ML_VectorLib_Function("ML_VLOAD", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: "2"}, arity = 4),
        type_custom_match(FSM(v2float32), TCM(ML_TableFormat), FSM(v2int32), FSM(v2int32)):
            ML_VectorLib_Function("ML_VLOAD2D", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "2"}, arity = 5),
        type_custom_match(FSM(v4float32), TCM(ML_TableFormat), FSM(v4int32)):
            ML_VectorLib_Function("ML_VLOAD", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: "4"}, arity = 4),
        type_custom_match(FSM(v8float32), TCM(ML_TableFormat), FSM(v8int32)):
            ML_VectorLib_Function("ML_VLOAD", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: "8"}, arity = 4),
        # variant with unsigned index (gather)
        type_custom_match(FSM(v4float32), TCM(ML_TableFormat), FSM(v4uint32)):
            ML_VectorLib_Function("ML_VLOAD", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: "4"}, arity = 4),
        type_custom_match(FSM(v4float32), TCM(ML_TableFormat), FSM(v4int32), FSM(v4int32)):
            ML_VectorLib_Function("ML_VLOAD2D", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "4"}, arity = 5),
        type_custom_match(FSM(v8float32), TCM(ML_TableFormat), FSM(v8int32), FSM(v8int32)):
            ML_VectorLib_Function("ML_VLOAD2D", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "8"}, arity = 5),
        # double precision loading (gather)
        type_custom_match(FSM(v4float64), TCM(ML_TableFormat), FSM(v4int32)):
            ML_VectorLib_Function("ML_VLOAD", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: "4"}, arity = 4),
        type_custom_match(FSM(v4float64), TCM(ML_TableFormat), FSM(v4int64)):
            ML_VectorLib_Function("ML_VLOAD", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: "4"}, arity = 4),
        type_custom_match(FSM(v4float64), TCM(ML_TableFormat), FSM(v4int32), FSM(v4int32)):
            ML_VectorLib_Function("ML_VLOAD2D", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "4"}, arity = 5),
        type_custom_match(FSM(v2float64), TCM(ML_TableFormat), FSM(v2int32)):
            ML_VectorLib_Function("ML_VLOAD", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: "2"}, arity = 4),
        type_custom_match(FSM(v2float64), TCM(ML_TableFormat), FSM(v2int32), FSM(v2int32)):
            ML_VectorLib_Function("ML_VLOAD2D", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "2"}, arity = 5),
        # double precision loading with unsigned index (gather)
        type_custom_match(FSM(v4float64), TCM(ML_TableFormat), FSM(v4uint32)):
            ML_VectorLib_Function("ML_VLOAD", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: "4"}, arity = 4),
        type_custom_match(FSM(v4float64), TCM(ML_TableFormat), FSM(v4int64)):
            ML_VectorLib_Function("ML_VLOAD", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: "4"}, arity = 4),
        type_custom_match(FSM(v4float64), TCM(ML_TableFormat), FSM(v4uint32), FSM(v4uint32)):
            ML_VectorLib_Function("ML_VLOAD2D", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "4"}, arity = 5),
        type_custom_match(FSM(v4float64), TCM(ML_TableFormat), FSM(v4int64), FSM(v4int32)):
            ML_VectorLib_Function("ML_VLOAD2D", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "4"}, arity = 5),
        type_custom_match(FSM(v4float64), TCM(ML_TableFormat), FSM(v4int64), FSM(v4int64)):
            ML_VectorLib_Function("ML_VLOAD2D", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "4"}, arity = 5),

        # 8-element double precision vectors
        type_custom_match(FSM(v8float64), TCM(ML_TableFormat), FSM(v8int64)):
            ML_VectorLib_Function("ML_VLOAD", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: "8"}, arity = 4),

        # 8-element vectors (gather)
        type_custom_match(FSM(v8float64), TCM(ML_TableFormat), FSM(v8int64), FSM(v8int64)):
            ML_VectorLib_Function("ML_VLOAD2D", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "8"}, arity = 5),
        type_custom_match(FSM(v8float64), TCM(ML_TableFormat), FSM(v8int64), FSM(v8int32)):
            ML_VectorLib_Function("ML_VLOAD2D", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "8"}, arity = 5),
        type_custom_match(FSM(v8float32), TCM(ML_TableFormat), FSM(v8uint32)):
            ML_VectorLib_Function("ML_VLOAD", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3:  "8"}, arity = 4),
        type_custom_match(FSM(v8float64), TCM(ML_TableFormat), FSM(v8uint32)):
            ML_VectorLib_Function("ML_VLOAD", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3:  "8"}, arity = 4),
            
        # vector load (contiguous)
        type_custom_match(FSM(v2int32), TCM(ML_TableFormat), type_table_index_match):
            TemplateOperatorFormat("memcpy({0}, ((ml_int2_t*)({1}+ {2})), 8)", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity=3, require_header=["string.h"]),
        type_custom_match(FSM(v4int32), TCM(ML_TableFormat), type_table_index_match):
            TemplateOperatorFormat("memcpy({0}, ((ml_int4_t*)({1}+ {2})), 16)", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity=3, require_header=["string.h"]),
        type_custom_match(FSM(v8int32), TCM(ML_TableFormat), type_table_index_match):
            TemplateOperatorFormat("memcpy({0}, ((ml_int8_t*)({1}+ {2})), 32)", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity=3, require_header=["string.h"]),

        type_custom_match(FSM(v2float32), TCM(ML_TableFormat), type_table_index_match):
            TemplateOperatorFormat("memcpy({0}, ((ml_float2_t*)({1}+ {2})), 8)", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity=3, require_header=["string.h"]),
        type_custom_match(FSM(v4float32), TCM(ML_TableFormat), type_table_index_match):
            TemplateOperatorFormat("memcpy({0}, ((ml_float4_t*)({1}+ {2})), 16)", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity=3, require_header=["string.h"]),
        type_custom_match(FSM(v8float32), TCM(ML_TableFormat), type_table_index_match):
            TemplateOperatorFormat("memcpy({0}, ((ml_float8_t*)({1}+ {2})), 32)", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity=3, require_header=["string.h"]),

        type_custom_match(FSM(v2float32), TCM(ML_Pointer_Format), type_table_index_match):
            TemplateOperatorFormat("memcpy({0}, ((ml_float2_t*)({1}+ {2})), 8)", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity=3, require_header=["string.h"]),
        type_custom_match(FSM(v4float32), TCM(ML_Pointer_Format), type_table_index_match):
            TemplateOperatorFormat("memcpy({0}, ((ml_float4_t*)({1}+ {2})), 16)", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity=3, require_header=["string.h"]),
        type_custom_match(FSM(v8float32), TCM(ML_TableFormat), type_table_index_match):
            TemplateOperatorFormat("memcpy({0}, ((ml_float8_t*)({1}+ {2})), 32)", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity=3, require_header=["string.h"]),

        type_custom_match(FSM(v2int64), TCM(ML_TableFormat), type_table_index_match):
            TemplateOperatorFormat("memcpy({0}, ((ml_long2_t*)({1}+ {2})), 16)", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity=3, require_header=["string.h"]),
        type_custom_match(FSM(v4int64), TCM(ML_TableFormat), type_table_index_match):
            TemplateOperatorFormat("memcpy({0}, ((ml_long4_t*)({1}+ {2})), 32)", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity=3, require_header=["string.h"]),
        type_custom_match(FSM(v8int64), TCM(ML_TableFormat), type_table_index_match):
            TemplateOperatorFormat("memcpy({0}, ((ml_long8_t*)({1}+ {2})), 64)", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity=3, require_header=["string.h"]),

        type_custom_match(FSM(v2float64), TCM(ML_TableFormat), type_table_index_match):
            TemplateOperatorFormat("memcpy({0}, ((ml_double2_t*)({1}+ {2})), 16)", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity=3, require_header=["string.h"]),
        type_custom_match(FSM(v4float64), TCM(ML_TableFormat), type_table_index_match):
            TemplateOperatorFormat("memcpy({0}, ((ml_double4_t*)({1}+ {2})), 32)", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity=3, require_header=["string.h"]),
        type_custom_match(FSM(v8float64), TCM(ML_TableFormat), type_table_index_match):
            TemplateOperatorFormat("memcpy({0}, ((ml_double4_t*)({1}+ {2})), 64)", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity=3, require_header=["string.h"]),

        type_custom_match(FSM(v2float64), TCM(ML_Pointer_Format), type_table_index_match):
            TemplateOperatorFormat("memcpy({0}, ((ml_double2_t*)({1}+ {2})), 16)", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity=3, require_header=["string.h"]),
        type_custom_match(FSM(v4float64), TCM(ML_Pointer_Format), type_table_index_match):
            TemplateOperatorFormat("memcpy({0}, ((ml_double4_t*)({1}+ {2})), 32)", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity=3, require_header=["string.h"]),
        type_custom_match(FSM(v8float64), TCM(ML_Pointer_Format), type_table_index_match):
            TemplateOperatorFormat("memcpy({0}, ((ml_double8_t*)({1}+ {2})), 64)", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity=3, require_header=["string.h"]),
      },
    },
  },
  TableStore: {
    None: {
        lambda _: True: {
            type_custom_match(FSM(ML_Void), FSM(v2int32), TCM(ML_TableFormat), type_table_index_match):
                TemplateOperatorFormat("memcpy(((ml_int2_t*)({1}+ {2})), &({0}),  8)", arg_map = {0: FO_Arg(0), 1: FO_Arg(1), 2: FO_Arg(2)}, arity=3, void_function=True, require_header=["string.h"]),
            type_custom_match(FSM(ML_Void), FSM(v4int32), TCM(ML_TableFormat), type_table_index_match):
                TemplateOperatorFormat("memcpy(((ml_int4_t*)({1}+ {2})), &({0}), 16)", arg_map = {0: FO_Arg(0), 1: FO_Arg(1), 2: FO_Arg(2)}, arity=3,void_function=True,  require_header=["string.h"]),
            type_custom_match(FSM(ML_Void), FSM(v8int32), TCM(ML_TableFormat), type_table_index_match):
                TemplateOperatorFormat("memcpy(((ml_int8_t*)({1}+ {2})), &({0}), 32)", arg_map = {0: FO_Arg(0), 1: FO_Arg(1), 2: FO_Arg(2)}, arity=3,void_function=True,  require_header=["string.h"]),

            type_custom_match(FSM(ML_Void), FSM(v2int32), TCM(ML_Pointer_Format), type_table_index_match):
                TemplateOperatorFormat("memcpy(((ml_int2_t*)({1}+ {2})), &({0}),  8)", arg_map = {0: FO_Arg(0), 1: FO_Arg(1), 2: FO_Arg(2)}, arity=3, void_function=True, require_header=["string.h"]),
            type_custom_match(FSM(ML_Void), FSM(v4int32), TCM(ML_Pointer_Format), type_table_index_match):
                TemplateOperatorFormat("memcpy(((ml_int4_t*)({1}+ {2})), &({0}), 16)", arg_map = {0: FO_Arg(0), 1: FO_Arg(1), 2: FO_Arg(2)}, arity=3, void_function=True, require_header=["string.h"]),
            type_custom_match(FSM(ML_Void), FSM(v8int32), TCM(ML_Pointer_Format), type_table_index_match):
                TemplateOperatorFormat("memcpy(((ml_int8_t*)({1}+ {2})), &({0}), 32)", arg_map = {0: FO_Arg(0), 1: FO_Arg(1), 2: FO_Arg(2)}, arity=3, void_function=True, require_header=["string.h"]),

            type_custom_match(FSM(ML_Void), FSM(v2float32), TCM(ML_TableFormat), type_table_index_match):
                TemplateOperatorFormat("memcpy(((ml_float2_t*)({1}+ {2})), &({0}),  8)", arg_map = {0: FO_Arg(0), 1: FO_Arg(1), 2: FO_Arg(2)}, arity=3, void_function=True, require_header=["string.h"]),
            type_custom_match(FSM(ML_Void), FSM(v4float32), TCM(ML_TableFormat), type_table_index_match):
                TemplateOperatorFormat("memcpy(((ml_float4_t*)({1}+ {2})), &({0}), 16)", arg_map = {0: FO_Arg(0), 1: FO_Arg(1), 2: FO_Arg(2)}, arity=3,void_function=True,  require_header=["string.h"]),
            type_custom_match(FSM(ML_Void), FSM(v8float32), TCM(ML_TableFormat), type_table_index_match):
                TemplateOperatorFormat("memcpy(((ml_float8_t*)({1}+ {2})), &({0}), 32)", arg_map = {0: FO_Arg(0), 1: FO_Arg(1), 2: FO_Arg(2)}, arity=3,void_function=True,  require_header=["string.h"]),

            type_custom_match(FSM(ML_Void), FSM(v2float32), TCM(ML_Pointer_Format), type_table_index_match):
                TemplateOperatorFormat("memcpy(((ml_float2_t*)({1}+ {2})), &({0}),  8)", arg_map = {0: FO_Arg(0), 1: FO_Arg(1), 2: FO_Arg(2)}, arity=3, void_function=True, require_header=["string.h"]),
            type_custom_match(FSM(ML_Void), FSM(v4float32), TCM(ML_Pointer_Format), type_table_index_match):
                TemplateOperatorFormat("memcpy(((ml_float4_t*)({1}+ {2})), &({0}), 16)", arg_map = {0: FO_Arg(0), 1: FO_Arg(1), 2: FO_Arg(2)}, arity=3, void_function=True, require_header=["string.h"]),
            type_custom_match(FSM(ML_Void), FSM(v8float32), TCM(ML_Pointer_Format), type_table_index_match):
                TemplateOperatorFormat("memcpy(((ml_float8_t*)({1}+ {2})), &({0}), 32)", arg_map = {0: FO_Arg(0), 1: FO_Arg(1), 2: FO_Arg(2)}, arity=3, void_function=True, require_header=["string.h"]),

            type_custom_match(FSM(ML_Void), FSM(v2float64), TCM(ML_TableFormat), type_table_index_match):
                TemplateOperatorFormat("memcpy(((ml_double2_t*)({1}+ {2})), &({0}),  16)", arg_map = {0: FO_Arg(0), 1: FO_Arg(1), 2: FO_Arg(2)}, arity=3, void_function=True, require_header=["string.h"]),
            type_custom_match(FSM(ML_Void), FSM(v4float64), TCM(ML_TableFormat), type_table_index_match):
                TemplateOperatorFormat("memcpy(((ml_double4_t*)({1}+ {2})), &({0}), 32)", arg_map = {0: FO_Arg(0), 1: FO_Arg(1), 2: FO_Arg(2)}, arity=3,void_function=True,  require_header=["string.h"]),
            type_custom_match(FSM(ML_Void), FSM(v8float64), TCM(ML_TableFormat), type_table_index_match):
                TemplateOperatorFormat("memcpy(((ml_double8_t*)({1}+ {2})), &({0}), 64)", arg_map = {0: FO_Arg(0), 1: FO_Arg(1), 2: FO_Arg(2)}, arity=3,void_function=True,  require_header=["string.h"]),

            type_custom_match(FSM(ML_Void), FSM(v2float64), TCM(ML_Pointer_Format), type_table_index_match):
                TemplateOperatorFormat("memcpy(((ml_double2_t*)({1}+ {2})), &({0}),  16)", arg_map = {0: FO_Arg(0), 1: FO_Arg(1), 2: FO_Arg(2)}, arity=3, void_function=True, require_header=["string.h"]),
            type_custom_match(FSM(ML_Void), FSM(v4float64), TCM(ML_Pointer_Format), type_table_index_match):
                TemplateOperatorFormat("memcpy(((ml_double4_t*)({1}+ {2})), &({0}), 32)", arg_map = {0: FO_Arg(0), 1: FO_Arg(1), 2: FO_Arg(2)}, arity=3, void_function=True, require_header=["string.h"]),
            type_custom_match(FSM(ML_Void), FSM(v4float64), TCM(ML_Pointer_Format), type_table_index_match):
                TemplateOperatorFormat("memcpy(((ml_double8_t*)({1}+ {2})), &({0}), 64)", arg_map = {0: FO_Arg(0), 1: FO_Arg(1), 2: FO_Arg(2)}, arity=3, void_function=True, require_header=["string.h"]),
        },
    },
  },
  Max: {
      None: {
          lambda _: True: {
              type_uniform_op2_match_restrain(SUPPORTED_VECTOR_FORMATS):
                  ComplexOperator(optree_modifier=max_legalizer),
          }
      },
  },
  Min: {
      None: {
          lambda _: True: {
              type_uniform_op2_match_restrain(SUPPORTED_VECTOR_FORMATS):
                  ComplexOperator(optree_modifier=min_legalizer),
          }
      },
  },
  BitLogicAnd: {
    None: {
       lambda _: True: dict(
            (
                type_strict_match(vformat, vformat, vformat),
                SymbolOperator("&", arity=2),
            ) for vformat in BASIC_INTEGER_VFORMAT_LIST
        )
    },
  },
  BitLogicOr: {
    None: {
       lambda _: True: dict(
            (
                type_strict_match(vformat, vformat, vformat),
                SymbolOperator("|", arity=2),
            ) for vformat in BASIC_INTEGER_VFORMAT_LIST
        )
    },
  },
  BitLogicXor: {
    None: {
       lambda _: True: dict(
            (
                type_strict_match(vformat, vformat, vformat),
                SymbolOperator("^", arity=2),
            ) for vformat in BASIC_INTEGER_VFORMAT_LIST
        )
    },
  },
  BitLogicNegate: {
    None: {
       lambda _: True: dict(
            (
                type_strict_match(vformat, vformat),
                SymbolOperator("~", arity=1),
            ) for vformat in BASIC_INTEGER_VFORMAT_LIST
        )
    },
  },
  BitLogicLeftShift: {
    None: {
       lambda _: True: dict(
            (
                type_strict_match(vformat, vformat, vformat),
                SymbolOperator("<<", arity=2),
            ) for vformat in BASIC_INTEGER_VFORMAT_LIST
        )
    },
  },
  BitLogicRightShift: {
    None: {
       lambda _: True: dict(
            (
                type_strict_match(vformat, vformat, vformat),
                SymbolOperator(">>", arity=2),
            ) for vformat in BASIC_INTEGER_VFORMAT_LIST
        )
    },
  },
  BitArithmeticRightShift: {
    None: {
       lambda _: True: dict(
            (
                type_strict_match(vformat, vformat, vformat),
                SymbolOperator(">>", arity=2),
            ) for vformat in BASIC_INTEGER_VFORMAT_LIST
        )
    },
  },
  Select: {
    None: {
       lambda _: True: {
        type_strict_match(v2int32, v2bool, v2int32, v2int32): ML_VectorLib_Function("ML_VSELECT", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "2"}, arity = 3),
        type_strict_match(v3int32, v3bool, v3int32, v3int32): ML_VectorLib_Function("ML_VSELECT", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "4"}, arity = 3, output_precision = v3int32),
        type_strict_match(v4int32, v4bool, v4int32, v4int32): ML_VectorLib_Function("ML_VSELECT", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "4"}, arity = 3, output_precision = v4int32),
        type_strict_match(v8int32, v8bool, v8int32, v8int32): ML_VectorLib_Function("ML_VSELECT", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "8"}, arity = 3, output_precision = v8int32),
        # floating-point select
        type_strict_match(v2float32, v2bool, v2float32, v2float32):
            ML_VectorLib_Function("ML_VSELECT", arg_map={0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "2"}, arity=3, output_precision=v2float32),
        type_strict_match(v3float32, v3bool, v3float32, v3float32):
            ML_VectorLib_Function("ML_VSELECT", arg_map={0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "3"}, arity=3, output_precision=v3float32),
        type_strict_match(v4float32, v4bool, v4float32, v4float32):
            ML_VectorLib_Function("ML_VSELECT", arg_map={0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "4"}, arity=3, output_precision=v4float32),
        type_strict_match(v8float32, v8bool, v8float32, v8float32):
            ML_VectorLib_Function("ML_VSELECT", arg_map={0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "8"}, arity=3, output_precision=v8float32),
        # long int select
        type_strict_match(v2int64, v2lbool, v2int64, v2int64): ML_VectorLib_Function("ML_VSELECT", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "2"}, arity = 3),
        type_strict_match(v3int64, v3lbool, v3int64, v3int64): ML_VectorLib_Function("ML_VSELECT", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "4"}, arity = 3, output_precision = v3int64),
        type_strict_match(v4int64, v4lbool, v4int64, v4int64): ML_VectorLib_Function("ML_VSELECT", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "4"}, arity = 3, output_precision = v4int64),
        type_strict_match(v8int64, v8lbool, v8int64, v8int64): ML_VectorLib_Function("ML_VSELECT", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "8"}, arity = 3, output_precision = v8int64),
        # double select
        type_strict_match(v4float64, v4lbool, v4float64, v4float64):
            ML_VectorLib_Function("ML_VSELECT", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "4"}, arity = 3, output_precision = v4float64),
        type_strict_match(v8float64, v8lbool, v8float64, v8float64):
            ML_VectorLib_Function("ML_VSELECT", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "8"}, arity = 3, output_precision = v8float64),
      },
    },
  },
  Modulo: {
    None: {
       lambda _: True: dict(
            (
                type_strict_match(vformat, vformat, vformat),
                SymbolOperator("%", arity=2)
            ) for vformat in BASIC_INTEGER_VFORMAT_LIST
        )
    },
  },
  Division: {
    None: {
       lambda _: True: dict(
            (
                type_strict_match(vformat, vformat, vformat),
                SymbolOperator("/", arity=2)
            ) for vformat in BASIC_VFORMAT_LIST
        )
    },
  },
  Addition: {
    None: {
      lambda _: True:
        dict(
          sum(
          [
            [
              (type_strict_match(
                  VECTOR_TYPE_MAP[scalar_type][vector_size],
                  VECTOR_TYPE_MAP[scalar_type][vector_size],
                  VECTOR_TYPE_MAP[scalar_type][vector_size]
                ), SymbolOperator("+", arity = 2)
              ) for vector_size in supported_vector_size
            ] for scalar_type in [ML_Binary32, ML_Binary64, ML_Int32, ML_UInt32, ML_Int64, ML_UInt64]
          ], [])
        )
     }
  },
  Subtraction: {
    None: {
      lambda _: True:
        dict(
          sum(
          [
            [
              (type_strict_match(
                  VECTOR_TYPE_MAP[scalar_type][vector_size],
                  VECTOR_TYPE_MAP[scalar_type][vector_size],
                  VECTOR_TYPE_MAP[scalar_type][vector_size]
                ), SymbolOperator("-", arity = 2)
              ) for vector_size in supported_vector_size
            ] for scalar_type in [ML_Binary32, ML_Binary64, ML_Int32, ML_UInt32]
          ], [])
        )
     }
  },
  Multiplication: {
    None: {
      lambda _: True:
        dict(
          sum(
          [
            [
              (type_strict_match(
                  VECTOR_TYPE_MAP[scalar_type][vector_size],
                  VECTOR_TYPE_MAP[scalar_type][vector_size],
                  VECTOR_TYPE_MAP[scalar_type][vector_size]
                ), SymbolOperator("*", arity = 2)
              ) for vector_size in supported_vector_size
            ] for scalar_type in [ML_Binary32, ML_Binary64, ML_Int32, ML_UInt32]
          ], [])
        )
     }
  },
  FusedMultiplyAdd: {
    FusedMultiplyAdd.Standard: {
       lambda _: True: {
        type_strict_match(v2float32, v2float32, v2float32, v2float32): ML_VectorLib_Function("ml_vfmaf2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2)}, arity = 2, output_precision = v2float32),
        #type_strict_match(v3float32, v3float32, v3float32, v3float32): ML_VectorLib_Function("ml_vfmaf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2)}, arity = 2, output_precision = v3float32),
        type_strict_match(v4float32, v4float32, v4float32, v4float32): ML_VectorLib_Function("ml_vfmaf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2)}, arity = 2, output_precision = v4float32),
        type_strict_match(v8float32, v8float32, v8float32, v8float32): ML_VectorLib_Function("ml_vfmaf8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2)}, arity = 2, output_precision = v8float32),
        # double precision
        type_strict_match(v4float64, v4float64, v4float64, v4float64):
            ML_VectorLib_Function("ml_vfmad4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2)}, arity = 2, output_precision = v4float64),
        type_strict_match(v8float64, v8float64, v8float64, v8float64):
            ML_VectorLib_Function("ml_vfmad8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2)}, arity = 2, output_precision = v8float64),
      },
    },
    FusedMultiplyAdd.Subtract: {
       lambda _: True: {
        type_strict_match(v2float32, v2float32, v2float32, v2float32):
            ML_VectorLib_Function("ml_vfmsf2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2)}, arity = 2, output_precision = v2float32),
        type_strict_match(v3float32, v3float32, v3float32, v3float32):
            ML_VectorLib_Function("ml_vfmsf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2)}, arity = 2, output_precision = v3float32),
        type_strict_match(v4float32, v4float32, v4float32, v4float32):
            ML_VectorLib_Function("ml_vfmsf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2)}, arity = 2, output_precision = v4float32),
        type_strict_match(v8float32, v8float32, v8float32, v8float32):
            ML_VectorLib_Function("ml_vfmsf8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2)}, arity = 2, output_precision = v8float32),
        # double-precision versions
        type_strict_match(v4float64, v4float64, v4float64, v4float64):
            ML_VectorLib_Function("ml_vfmsd4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2)}, arity = 2, output_precision = v4float64),
        type_strict_match(v8float64, v8float64, v8float64, v8float64):
            ML_VectorLib_Function("ml_vfmsd8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2)}, arity = 2, output_precision = v8float64),
      },
    },
    FusedMultiplyAdd.SubtractNegate: {
       lambda _: True: {
        type_strict_match(v2float32, v2float32, v2float32, v2float32):
            ML_VectorLib_Function("ml_vfmsnf2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2)}, arity = 2, output_precision = v2float32),
        type_strict_match(v3float32, v3float32, v3float32, v3float32):
            ML_VectorLib_Function("ml_vfmsnf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2)}, arity = 2, output_precision = v3float32),
        type_strict_match(v4float32, v4float32, v4float32, v4float32):
            ML_VectorLib_Function("ml_vfmsnf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2)}, arity = 2, output_precision = v4float32),
        type_strict_match(v8float32, v8float32, v8float32, v8float32):
            ML_VectorLib_Function("ml_vfmsnf8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2)}, arity = 2, output_precision = v8float32),
        # double-precision versions
        type_strict_match(v4float64, v4float64, v4float64, v4float64):
            ML_VectorLib_Function("ml_vfmsnd4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2)}, arity = 2, output_precision = v4float64),
        type_strict_match(v8float64, v8float64, v8float64, v8float64):
            ML_VectorLib_Function("ml_vfmsnd8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2)}, arity = 2, output_precision = v8float64),
      },
    },
  },
  ExponentInsertion: {
    ExponentInsertion.Default: {
      lambda _: True: {
        type_strict_match(v2float32, v2int32): ML_VectorLib_Function("ml_vexp_insertion_f2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v2float32),
        #type_strict_match(v3float32, v3int32): ML_VectorLib_Function("ml_vexp_insertion_f4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v3float32),
        type_strict_match(v4float32, v4int32): ML_VectorLib_Function("ml_vexp_insertion_f4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v4float32),
        type_strict_match(v8float32, v8int32): ML_VectorLib_Function("ml_vexp_insertion_f8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v8float32),
        # double precision versions
        type_strict_match(v4float64, v4int64):
            ML_VectorLib_Function("ml_vexp_insertion_d4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v4float64),
        type_strict_match(v8float64, v8int64):
            ML_VectorLib_Function("ml_vexp_insertion_d8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v8float64),
      }
    },
  },
  ExponentExtraction: {
    None: {
      lambda _: True: {
        type_strict_match(v2int32, v2float32):
            ML_VectorLib_Function("ml_vexp_extraction_f2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v2int32),
        type_strict_match(v3int32, v3float32):
            ML_VectorLib_Function("ml_vexp_extraction_f4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v3int32),
        type_strict_match(v4int32, v4float32):
            ML_VectorLib_Function("ml_vexp_extraction_f4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v4int32),
        type_strict_match(v8int32, v8float32):
            ML_VectorLib_Function("ml_vexp_extraction_f8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v8int32),
        # double-precision variants
        type_strict_match(v4int32, v4float64):
            ML_VectorLib_Function("ml_vexp_extraction_d4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v4int32),
        type_strict_match(v8int32, v8float64):
            ML_VectorLib_Function("ml_vexp_extraction_d8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v8int32),
        type_strict_match(v4int64, v4float64):
            ML_VectorLib_Function("ml_vexp_extraction_d4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v4int64),
        type_strict_match(v8int64, v8float64):
            ML_VectorLib_Function("ml_vexp_extraction_d8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v8int64),
      }
    },
  },
  MantissaExtraction: {
    None: {
      lambda _: True: {
        # single-precision
        type_strict_match(v2float32, v2float32): ML_VectorLib_Function("ml_vmantissa_extraction_f2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v2float32),
        #type_strict_match(v3float32, v3float32): ML_VectorLib_Function("ml_vmantissa_extraction_f4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v3float32),
        type_strict_match(v4float32, v4float32): ML_VectorLib_Function("ml_vmantissa_extraction_f4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v4float32),
        type_strict_match(v8float32, v8float32): ML_VectorLib_Function("ml_vmantissa_extraction_f8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v8float32),
        # double precision
        type_strict_match(v4float64, v4float64):
            ML_VectorLib_Function("ml_vmantissa_extraction_d4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision=v4float64),
        type_strict_match(v8float64, v8float64):
            ML_VectorLib_Function("ml_vmantissa_extraction_d8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision=v8float64),
      }
    },
  },
  NearestInteger: {
    None: {
      lambda _: True : {
        type_strict_match(v2int32, v2float32): ML_VectorLib_Function("ml_vnearbyintf2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v2int32),
        type_strict_match(v4int32, v4float32): ML_VectorLib_Function("ml_vnearbyintf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v4int32),
        #type_strict_match(v3int32, v3float32): ML_VectorLib_Function("ml_vnearbyintf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v3int32),
        type_strict_match(v8int32, v8float32): ML_VectorLib_Function("ml_vnearbyintf8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v8int32),


        type_strict_match(v2float32, v2float32): ML_VectorLib_Function("ml_vrintf2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v2float32),
        type_strict_match(v3float32, v3float32): ML_VectorLib_Function("ml_vrintf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v3float32),
        type_strict_match(v4float32, v4float32): ML_VectorLib_Function("ml_vrintf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v4float32),
        type_strict_match(v8float32, v8float32): ML_VectorLib_Function("ml_vrintf8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v8float32),

        # double precision versions
        type_strict_match(v4int64, v4float64): ML_VectorLib_Function("ml_vnearbyintd4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v4int64),
        type_strict_match(v8int64, v8float64): ML_VectorLib_Function("ml_vnearbyintd8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v8int64),

        type_strict_match(v2float64, v2float64):
            ML_VectorLib_Function("ml_vrintd2", arg_map={0: FO_ResultRef(0), 1: FO_Arg(0)}, arity=1, output_precision=v2float64),
        type_strict_match(v4float64, v4float64):
            ML_VectorLib_Function("ml_vrintd4", arg_map={0: FO_ResultRef(0), 1: FO_Arg(0)}, arity=1, output_precision=v4float64),
        type_strict_match(v8float64, v8float64):
            ML_VectorLib_Function("ml_vrintd8", arg_map={0: FO_ResultRef(0), 1: FO_Arg(0)}, arity=1, output_precision=v8float64),
      }
    }
  },
  Floor: {
    None: {
        lambda _: True: {

            # TODO fixme, error in 3rd parameter of VECTORIZE_OP1
            type_strict_match(v2float32, v2float32):
                ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "floor", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "2"}, arity = 1, output_precision = v2float32),
            type_strict_match(v4float32, v4float32):
                ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "floor", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = v4float32),
            # TODO fixme, error in 3rd parameter of VECTORIZE_OP1
            type_strict_match(v8float32, v8float32):
                ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "floor", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "8"}, arity = 1, output_precision = v8float32),

            # TODO fixme, error in 3rd parameter of VECTORIZE_OP1
            type_strict_match(v2float64, v2float64):
                ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "floor", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "2"}, arity = 1, output_precision = v2float64),
            type_strict_match(v4float64, v4float64):
                ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "floor", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = v4float64),
            # TODO fixme, error in 3rd parameter of VECTORIZE_OP1
            type_strict_match(v8float64, v8float64):
                ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "floor", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "8"}, arity = 1, output_precision = v8float64),
        },
     },

  },
  Negation: {
    None: {
      lambda _: True:
        dict(
          sum(
          [
            [
              (
                type_strict_match(VECTOR_TYPE_MAP[scalar_type][vector_size], VECTOR_TYPE_MAP[scalar_type][vector_size]),
                SymbolOperator("-", arity=1)
              ) for vector_size in supported_vector_size
            ] for scalar_type in [ML_Binary32, ML_Binary64, ML_Int32, ML_UInt32, ML_Int64, ML_UInt64]
          ]
          , []
          )
        )
     }
  },
  Conversion: {
    None: {
      lambda optree: True: {
        # 2-elt vectors
        type_strict_match(v2uint32, v2int32) : ML_VectorLib_Function("ML_VCONV", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: "2"}, arity = 3),
        type_strict_match(v2int32, v2uint32) :  ML_VectorLib_Function("ML_VCONV", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: "2"}, arity = 3),
        type_strict_match(v2float32, v2int32) :  ML_VectorLib_Function("ML_VCONV", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: "2"}, arity = 3),
        type_strict_match(v2int32, v2float32) :  ML_VectorLib_Function("ML_VCONV", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: "2"}, arity = 3),
        # 4-elt vectors
        type_strict_match(v4uint32, v4int32) : ML_VectorLib_Function("ML_VCONV", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: "4"}, arity = 3),
        type_strict_match(v4int32, v4uint32) :  ML_VectorLib_Function("ML_VCONV", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: "4"}, arity = 3),
        type_strict_match(v4float32, v4int32) :  ML_VectorLib_Function("ML_VCONV", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: "4"}, arity = 3),
        type_strict_match(v4int32, v4float32) :  ML_VectorLib_Function("ML_VCONV", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: "4"}, arity = 3),
        # 8-elt vectors
        type_strict_match(v8uint32, v8int32) : ML_VectorLib_Function("ML_VCONV", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: "8"}, arity = 3),
        type_strict_match(v8int32, v8uint32) :  ML_VectorLib_Function("ML_VCONV", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: "8"}, arity = 3),
        type_strict_match(v8float32, v8int32) :  ML_VectorLib_Function("ML_VCONV", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: "8"}, arity = 3),
        type_strict_match(v8int32, v8float32) :  ML_VectorLib_Function("ML_VCONV", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: "8"}, arity = 3),
        type_strict_match(v8int64, v8int32) : ML_VectorLib_Function("ML_VCONV", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: "8"}, arity = 3),
        # conversion from and to double precision
        type_strict_match(v2float64, v2float32) :  ML_VectorLib_Function("ML_VCONV", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: "2"}, arity = 3),
        type_strict_match(v2float32, v2float64) :  ML_VectorLib_Function("ML_VCONV", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: "2"}, arity = 3),
        type_strict_match(v4float32, v4float64) :  ML_VectorLib_Function("ML_VCONV", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: "4"}, arity = 3),
        type_strict_match(v4float64, v4float32) :  ML_VectorLib_Function("ML_VCONV", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: "4"}, arity = 3),

        # 64-bit values
        # integer to float
        type_strict_match(v4float64, v4int64):
            ML_VectorLib_Function("ML_VCONV", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: "4"}, arity = 3),
        type_strict_match(v8float64, v8int64):
            ML_VectorLib_Function("ML_VCONV", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: "8"}, arity = 3),
        # float to integer
        type_strict_match(v4int64, v4float64):
            ML_VectorLib_Function("ML_VCONV", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: "4"}, arity = 3),
        type_strict_match(v8int64, v8float64):
            ML_VectorLib_Function("ML_VCONV", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: "8"}, arity = 3),
        type_strict_match(v8float64, v8int32):
            ML_VectorLib_Function("ML_VCONV", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: "8"}, arity = 3),
        # cross size conversion
        type_strict_match(v4uint32, v4uint64) :  ML_VectorLib_Function("ML_VCONV", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: "4"}, arity = 3),
        type_strict_match(v4int64, v4int32) :  ML_VectorLib_Function("ML_VCONV", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: "4"}, arity = 3),
        type_strict_match(v4float64, v4int32) :  ML_VectorLib_Function("ML_VCONV", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: "4"}, arity = 3),

        type_strict_match(v8uint32, v8uint64) :  ML_VectorLib_Function("ML_VCONV", arg_map={0: FO_ResultRef(0), 1: FO_Arg(0), 2: "4"}, arity=3),
        type_strict_match(v8int64, v8int32) :  ML_VectorLib_Function("ML_VCONV", arg_map={0: FO_ResultRef(0), 1: FO_Arg(0), 2: "4"}, arity=3),
      },
    },
  },
  CountLeadingZeros: {
    None: {
      lambda optree: True: {
        type_strict_match(v4uint32, v4uint32) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "__builtin_clz", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = v4uint32),
      },
    },
  },
  Abs: {
    None: {
      lambda optree: True: {
        # single-precision
        type_strict_match(v2float32, v2float32):
            ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "fabsf", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "2"}, arity = 1, output_precision = v2float32, require_header = ["math.h"]),
        type_strict_match(v3float32, v3float32):
            ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "fabsf", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "3"}, arity = 1, output_precision = v3float32, require_header = ["math.h"]),
        type_strict_match(v4float32, v4float32):
            ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "fabsf", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = v4float32, require_header = ["math.h"]),
        type_strict_match(v8float32, v8float32):
            ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "fabsf", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "8"}, arity = 1, output_precision = v8float32, require_header = ["math.h"]),
        # double precision
        type_strict_match(v2float64, v2float64):
            ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "fabs", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "2"}, arity = 1, output_precision = v2float64, require_header = ["math.h"]),
        type_strict_match(v3float64, v3float64):
            ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "fabs", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "3"}, arity = 1, output_precision = v3float64, require_header = ["math.h"]),
        type_strict_match(v4float64, v4float64):
            ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "fabs", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = v4float64, require_header = ["math.h"]),
        type_strict_match(v8float64, v8float64):
            ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "fabs", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "8"}, arity = 1, output_precision = v8float64, require_header = ["math.h"]),
      },
    },
  },
  Trunc: {
    None: {
      lambda optree: True: {
        # single-precision
        type_strict_match(v2float32, v2float32) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "truncf", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "2"}, arity = 1, output_precision = v2float32, require_header = ["math.h"]),
        type_strict_match(v3float32, v3float32) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "truncf", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "3"}, arity = 1, output_precision = v3float32, require_header = ["math.h"]),
        type_strict_match(v4float32, v4float32) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "truncf", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = v4float32, require_header = ["math.h"]),
        type_strict_match(v8float32, v8float32) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "truncf", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "8"}, arity = 1, output_precision = v8float32, require_header = ["math.h"]),
        # double precision
        type_strict_match(v2float64, v2float64) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "trunc", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "2"}, arity = 1, output_precision = v2float64, require_header = ["math.h"]),
        type_strict_match(v3float64, v3float64) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "trunc", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "3"}, arity = 1, output_precision = v3float64, require_header = ["math.h"]),
        type_strict_match(v4float64, v4float64) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "trunc", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = v4float64, require_header = ["math.h"]),
        type_strict_match(v8float64, v8float64) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "trunc", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "8"}, arity = 1, output_precision = v8float64, require_header = ["math.h"]),
      },
    },
  },
  TypeCast: {
      None: {
          lambda optree: True: {
              type_strict_match(v4float32, v4int32):
                  ML_VectorLib_Function("v4si_cast2_v4sf", arity=1, output_precision = v4float32),
              type_strict_match(v4int32, v4float32):
                  ML_VectorLib_Function("v4sf_cast2_v4si", arity=1, output_precision = v4float32),

              type_strict_match(v3float32, v3int32) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "float_from_32b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = v3float32),
              type_strict_match(v3int32, v3float32) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "float_to_32b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = v3int32),

              type_strict_match(v2float32, v2int32) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "float_from_32b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "2"}, arity = 1, output_precision = v2float32),
              type_strict_match(v2int32, v2float32) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "float_to_32b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "2"}, arity = 1, output_precision = v2int32),

              type_strict_match(v4float32, v4int32) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "float_from_32b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = v4float32),
              type_strict_match(v4int32, v4float32) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "float_to_32b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = v4int32),

              type_strict_match(v4float32, v4uint32) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "float_from_32b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = v4float32),
              type_strict_match(v4uint32, v4float32) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "float_to_32b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = v4uint32),
              # 2-element vector variants
              type_strict_match(v2float32, v2uint32) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "float_from_32b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "2"}, arity = 1, output_precision = v2float32),
              type_strict_match(v2uint32, v2float32) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "float_to_32b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "2"}, arity = 1, output_precision = v2uint32),

              # 4-element integer variants
              type_strict_match(v4uint32, v4int32) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "(uint32_t)", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = v4uint32),
              type_strict_match(v4int32, v4uint32) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "(int32_t)", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = v4int32),

              # 4-element 64-bit integer variants
              type_strict_match(v4uint64, v4int64) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "(uint64_t)", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = v4uint64),
              type_strict_match(v4int64, v4uint64) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "(int64_t)", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = v4int64),
              type_strict_match(v4int64, v4float64):
                ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "double_to_64b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = v4int64),
              type_strict_match(v4float64, v4int64):
                ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "double_from_64b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = v4float64),
              type_strict_match(v4uint64, v4float64):
                ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "double_to_64b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = v4uint64),
              type_strict_match(v4float64, v4uint64):
                ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "double_from_64b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = v4float64),
              # 8-element 64-bit variants
              type_strict_match(v8int64, v8float64):
                ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "double_to_64b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "8"}, arity = 1, output_precision = v8int64),
              type_strict_match(v8float64, v8int64):
                ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "double_from_64b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "8"}, arity = 1, output_precision = v8float64),
              type_strict_match(v8uint64, v8float64):
                ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "double_to_64b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "8"}, arity = 1, output_precision = v8uint64),
              type_strict_match(v8float64, v8uint64):
                ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "double_from_64b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "8"}, arity = 1, output_precision = v8float64),

              type_strict_match(v8uint64, v8int64):
                ML_VectorLib_Function("VECTORIZE_OP1", arg_map={0: "(uint64_t)", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity=1, output_precision=v8uint64),
              type_strict_match(v8int64, v8uint64):
                ML_VectorLib_Function("VECTORIZE_OP1", arg_map={0: "(int64_t)", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity=1, output_precision=v8int64),

              # 8-element variants
              type_strict_match(v8float32, v8int32):
                ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "float_from_32b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "8"}, arity = 1, output_precision = v8float32),
              type_strict_match(v8int32, v8float32):
                ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "float_to_32b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "8"}, arity = 1, output_precision = v8int32),

              # unsigned version for 8-element vectors
              type_strict_match(v8uint32, v8float32):
                ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "float_to_32b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "8"}, arity = 1, output_precision = v8uint32),
              type_strict_match(v8float32, v8uint32):
                ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "float_from_32b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "8"}, arity = 1, output_precision = v8float32),
              # between integer casts
              type_strict_match(v8uint32, v8int32) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "(uint32_t)", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = v8uint32),
              type_strict_match(v8int32, v8uint32) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "(int32_t)", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = v8int32),
          },
      },
  },


     #   type_strict_match(v2float32, v2float32): ML_VectorLib_Function("ml_vnegf2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v2float32),
     #   type_strict_match(v4float32, v4float32): ML_VectorLib_Function("ml_vnegf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v4float32),
     #   type_strict_match(v8float32, v8float32): ML_VectorLib_Function("ml_vnegf8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v8float32),
  #    }
  #  }
  #},
  VectorElementSelection:
  {
    None: {
      legal_vector_element_selection: {
        lambda rformat, opformat, indexformat, optree: True: TemplateOperator("%s[%s]", arity = 2, require_header = ["ml_support_lib.h"]),
      },
      lambda optree: not(legal_vector_element_selection(optree)): {
          type_strict_match_list([ML_SingleSingle], LIST_SINGLE_MULTI_PRECISION_VECTOR_FORMATS, [ML_Integer]):
            ComplexOperator(optree_modifier=legalize_multi_precision_vector_element_selection),
          type_strict_match_list([ML_DoubleDouble], LIST_DOUBLE_MULTI_PRECISION_VECTOR_FORMATS, [ML_Integer]):
            ComplexOperator(optree_modifier=legalize_multi_precision_vector_element_selection),
      }
    },
  },
  LogicalNot: {
    #None: {
    #  lambda _: True: {
    #    type_strict_match(v2bool, v2bool): ML_VectorLib_Function("ml_vnotb2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v2int32),
    #    type_strict_match(v3bool, v3bool): ML_VectorLib_Function("ml_vnotb3", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v3int32),
    #    type_strict_match(v4bool, v4bool): ML_VectorLib_Function("ml_vnotb4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v4int32),
    #    type_strict_match(v8bool, v8bool): ML_VectorLib_Function("ml_vnotb8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v8int32),
    #  },
    #},
    None: {
      lambda _: True: dict(
        (type_strict_match(vformat, vformat),
        # TODO/FIXME: trying to use bitwise operator for logical operation
        SymbolOperator("~", arity=1)) for vformat in VECTOR_BOOLEAN_FORMAT_LIST
      )
        
    },
  },
  LogicalOr: {
    None: {
      lambda _: True: dict(
        (type_strict_match(vformat, vformat, vformat),
        # TODO/FIXME: trying to use bitwise operator for logical operation
        SymbolOperator("|", arity=2)) for vformat in VECTOR_BOOLEAN_FORMAT_LIST
      )
        
    },
  },
  LogicalAnd: {
    None: {
      lambda _: True: dict(
       (
            type_strict_match(vbool_format, vbool_format, vbool_format),
            # TODO/FIXME: trying to use bitwise operator for logical operation
            SymbolOperator("&", arity=2)
        ) for vbool_format in VECTOR_BOOLEAN_FORMAT_LIST
      )
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
                      type_strict_match_list(
                        [
                          #vector_type[ML_Int32][vector_size],
                          VECTOR_TYPE_MAP[ML_Bool][scalar_type.get_bit_size()][vector_size]
                        ],
                        [
                          VECTOR_TYPE_MAP[scalar_type][vector_size]
                        ],
                        [
                          VECTOR_TYPE_MAP[scalar_type][vector_size]
                        ]
                      )
                      ,
                      SymbolOperator(COMPARATOR_SYMBOL_MAP[comp_specifier], arity=2)
                    ) for scalar_type in [
                        ML_Binary32, ML_Binary64,
                        ML_Int32, ML_UInt32,
                        ML_Int64, ML_UInt64
                        ]
                  ] for vector_size in supported_vector_size
                ], []
              )
            )
          )
      }
    ) for comp_specifier in [ Comparison.Equal, Comparison.NotEqual,
                              Comparison.Greater, Comparison.GreaterOrEqual,
                              Comparison.Less, Comparison.LessOrEqual ]
  ),
  Test: {
    Test.IsMaskAllZero: {
      lambda _: True: {
        type_strict_match_list([ML_Bool, ML_Int32], [v2bool]): ML_VectorLib_Function("ml_is_vmask2_zero", arity = 1, output_precision = ML_Int32),
        type_strict_match_list([ML_Bool, ML_Int32], [v3bool]): ML_VectorLib_Function("ml_is_vmask3_zero", arity = 1, output_precision = ML_Int32),
        type_strict_match_list([ML_Bool, ML_Int32], [v4bool]): ML_VectorLib_Function("ml_is_vmask4_zero", arity = 1, output_precision = ML_Int32),
        type_strict_match_list([ML_Bool, ML_Int32], [v8bool]): ML_VectorLib_Function("ml_is_vmask8_zero", arity = 1, output_precision = ML_Int32),
      },
    },
    Test.IsMaskAnyZero: {
      lambda _: True: {
        type_strict_match_list([ML_Bool, ML_Int32], [v2bool]): ML_VectorLib_Function("ml_is_vmask2_any_zero", arity = 1, output_precision = ML_Int32),
        type_strict_match_list([ML_Bool, ML_Int32], [v3bool]): ML_VectorLib_Function("ml_is_vmask3_any_zero", arity = 1, output_precision = ML_Int32),
        type_strict_match_list([ML_Bool, ML_Int32], [v4bool]): ML_VectorLib_Function("ml_is_vmask4_any_zero", arity = 1, output_precision = ML_Int32),
        type_strict_match_list([ML_Bool, ML_Int32], [v8bool]): ML_VectorLib_Function("ml_is_vmask8_any_zero", arity = 1, output_precision = ML_Int32),
      },
    },
    Test.IsMaskNotAnyZero: {
      lambda _: True: {
        type_strict_match_list([ML_Bool, ML_Int32], [v2bool]): ML_VectorLib_Function("ml_is_vmask2_not_any_zero", arity = 1, output_precision = ML_Int32),
        type_strict_match_list([ML_Bool, ML_Int32], [v3bool]): ML_VectorLib_Function("ml_is_vmask3_not_any_zero", arity = 1, output_precision = ML_Int32),
        type_strict_match_list([ML_Bool, ML_Int32], [v4bool]): ML_VectorLib_Function("ml_is_vmask4_not_any_zero", arity = 1, output_precision = ML_Int32),
        type_strict_match_list([ML_Bool, ML_Int32], [v8bool]): ML_VectorLib_Function("ml_is_vmask8_not_any_zero", arity = 1, output_precision = ML_Int32),
      },
    },
    Test.IsMaskNotAllZero: {
      lambda _: True: {
        type_strict_match_list([ML_Bool, ML_Int32], [v2bool]): ML_VectorLib_Function("ml_is_vmask2_not_all_zero", arity = 1, output_precision = ML_Int32),
        type_strict_match_list([ML_Bool, ML_Int32], [v3bool]): ML_VectorLib_Function("ml_is_vmask3_not_all_zero", arity = 1, output_precision = ML_Int32),
        type_strict_match_list([ML_Bool, ML_Int32], [v4bool]): ML_VectorLib_Function("ml_is_vmask4_not_all_zero", arity = 1, output_precision = ML_Int32),
        type_strict_match_list([ML_Bool, ML_Int32], [v8bool]): ML_VectorLib_Function("ml_is_vmask8_not_all_zero", arity = 1, output_precision = ML_Int32),
      },
    },
    Test.IsInfOrNaN: {
      lambda _: True: {
        type_strict_match(v2bool, v2float32): ML_VectorLib_Function("ml_vtestf2_is_nan_or_inf", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v2int32),
        type_strict_match(v3bool, v3float32): ML_VectorLib_Function("ml_vtestf3_is_nan_or_inf", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v3int32),
        type_strict_match(v4bool, v4float32): ML_VectorLib_Function("ml_vtestf4_is_nan_or_inf", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v4int32),
        type_strict_match(v8bool, v8float32): ML_VectorLib_Function("ml_vtestf8_is_nan_or_inf", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v8int32),

        type_strict_match(v4lbool, v4float64):
            ML_VectorLib_Function("ml_vtestd4_is_nan_or_inf", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision=v4lbool),
        type_strict_match(v8lbool, v8float64):
            ML_VectorLib_Function("ml_vtestd8_is_nan_or_inf", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision=v8lbool),
      },
    },
    Test.IsInfty: {
      lambda _: True: {
        type_strict_match(v2bool, v2float32): ML_VectorLib_Function("ml_vtestf2_is_inf", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v2int32),
        type_strict_match(v3bool, v3float32): ML_VectorLib_Function("ml_vtestf3_is_inf", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v3int32),
        type_strict_match(v4bool, v4float32): ML_VectorLib_Function("ml_vtestf4_is_inf", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v4int32),
        type_strict_match(v8bool, v8float32): ML_VectorLib_Function("ml_vtestf8_is_inf", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v8int32),
      },
    },
    Test.IsNaN: {
      lambda _: True: {
        type_strict_match(v2bool, v2float32): ML_VectorLib_Function("ml_vtestf2_is_nan", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v2int32),
        type_strict_match(v3bool, v3float32): ML_VectorLib_Function("ml_vtestf3_is_nan", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v3int32),
        type_strict_match(v4bool, v4float32): ML_VectorLib_Function("ml_vtestf4_is_nan", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v4int32),
        type_strict_match(v8bool, v8float32): ML_VectorLib_Function("ml_vtestf8_is_nan", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v8int32),

        type_strict_match(v4lbool, v4float64):
            ML_VectorLib_Function("ml_vtestd4_is_nan", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision=v4lbool),
        type_strict_match(v8lbool, v8float64):
            ML_VectorLib_Function("ml_vtestd8_is_nan", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision=v8lbool),
      },
    },
    Test.IsZero: {
      lambda _: True: {
        type_strict_match(v2bool, v2float32): ML_VectorLib_Function("ml_vtestf2_is_zero", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v2int32),
        type_strict_match(v3bool, v3float32): ML_VectorLib_Function("ml_vtestf3_is_zero", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v3int32),
        type_strict_match(v4bool, v4float32): ML_VectorLib_Function("ml_vtestf4_is_zero", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v4int32),
        type_strict_match(v8bool, v8float32): ML_VectorLib_Function("ml_vtestf8_is_zero", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v8int32),
      },
    },

    Test.IsSubnormal: {
      lambda _: True: {
        type_strict_match(v2bool, v2float32): ML_VectorLib_Function("ml_vtestf2_is_subnormal", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v2int32),
        type_strict_match(v3bool, v3float32): ML_VectorLib_Function("ml_vtestf3_is_subnormal", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v3int32),
        type_strict_match(v4bool, v4float32): ML_VectorLib_Function("ml_vtestf4_is_subnormal", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v4int32),
        type_strict_match(v8bool, v8float32): ML_VectorLib_Function("ml_vtestf8_is_subnormal", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v8int32),
        # double precision
        type_strict_match(v4lbool, v4float64):
            ML_VectorLib_Function("ml_vtestd4_is_subnormal", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v4lbool),
        type_strict_match(v8lbool, v8float64):
            ML_VectorLib_Function("ml_vtestd8_is_subnormal", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v8lbool),
      },
    },
  },
  ComponentSelection: {
      ComponentSelection.Hi: {
          lambda optree: True: dict(
            (
                type_strict_match(field_format, op_format),
                TemplateOperator("%s.hi", arity=1)
            ) for (field_format, op_format) in [
              (v2float64, v2dualfloat64),
              (v3float64, v3dualfloat64),
              (v4float64, v4dualfloat64),
              (v8float64, v8dualfloat64),

              (v2float32, v2dualfloat32),
              (v3float32, v3dualfloat32),
              (v4float32, v4dualfloat32),
              (v8float32, v8dualfloat32),

              (v2float64, v2trifloat64),
              (v3float64, v3trifloat64),
              (v4float64, v4trifloat64),
              (v8float64, v8trifloat64),

              (v2float32, v2trifloat32),
              (v3float32, v3trifloat32),
              (v4float32, v4trifloat32),
              (v8float32, v8trifloat32),
            ]
        )
      },
      ComponentSelection.Me: {
          lambda optree: True: dict(
            (
                type_strict_match(field_format, op_format),
                TemplateOperator("%s.me", arity=1)
            ) for (field_format, op_format) in [
              (v2float64, v2trifloat64),
              (v3float64, v3trifloat64),
              (v4float64, v4trifloat64),
              (v8float64, v8trifloat64),

              (v2float32, v2trifloat32),
              (v3float32, v3trifloat32),
              (v4float32, v4trifloat32),
              (v8float32, v8trifloat32),
            ]
        )
      },
      ComponentSelection.Lo: {
          lambda optree: True: dict(
            (
                type_strict_match(field_format, op_format),
                TemplateOperator("%s.lo", arity=1)
            ) for (field_format, op_format) in [
              (v2float64, v2dualfloat64),
              (v3float64, v3dualfloat64),
              (v4float64, v4dualfloat64),
              (v8float64, v8dualfloat64),

              (v2float32, v2dualfloat32),
              (v3float32, v3dualfloat32),
              (v4float32, v4dualfloat32),
              (v8float32, v8dualfloat32),

              (v2float64, v2trifloat64),
              (v3float64, v3trifloat64),
              (v4float64, v4trifloat64),
              (v8float64, v8trifloat64),

              (v2float32, v2trifloat32),
              (v3float32, v3trifloat32),
              (v4float32, v4trifloat32),
              (v8float32, v8trifloat32),
            ]
        )
      },
  },
  BuildFromComponent: {
      None: {
          lambda optree: True: dict(
            [(
                type_strict_match(op_format, field_format, field_format, field_format),
                TemplateOperatorFormat("((%s) {{.hi={0} , .me={1}, .lo={2}}})" % op_format.name[C_Code], arity=3)
            ) for (field_format, op_format) in [
              (v2float64, v2trifloat64),
              (v3float64, v3trifloat64),
              (v4float64, v4trifloat64),
              (v8float64, v8trifloat64),

              (v2float32, v2trifloat32),
              (v3float32, v3trifloat32),
              (v4float32, v4trifloat32),
              (v8float32, v8trifloat32),
            ]] + [
            (
                type_strict_match(op_format, field_format, field_format),
                TemplateOperatorFormat("((%s) {{.hi={0} , .lo={1}}})" % op_format.name[C_Code], arity=2)
            ) for (field_format, op_format) in [
              (v2float64, v2dualfloat64),
              (v3float64, v3dualfloat64),
              (v4float64, v4dualfloat64),
              (v8float64, v8dualfloat64),

              (v2float32, v2dualfloat32),
              (v3float32, v3dualfloat32),
              (v4float32, v4dualfloat32),
              (v8float32, v8dualfloat32),
            ]]
        )
      },
  },
}

vector_gappa_code_generation_table = {
}

@UniqueTargetDecorator
class VectorBackend(GenericProcessor):
  target_name = "vector"

  code_generation_table = {
    C_Code: vector_c_code_generation_table,
    OpenCL_Code: vector_opencl_code_generation_table,
  }

  def __init__(self, *args):
    GenericProcessor.__init__(self, *args)
    self.simplified_rec_op_map[OpenCL_Code] = self.generate_supported_op_map(language = OpenCL_Code)


  def is_supported_operation(self, optree, language = C_Code, debug = False, fallback = True,  key_getter = lambda self, optree: self.get_operation_keys(optree)):
    """ return whether or not the operation performed by optree is supported by any level of the processor hierarchy """
    language_supported =  self.is_map_supported_operation(self.simplified_rec_op_map, optree, language, debug = debug, key_getter = key_getter)
    # fallback to C_Code
    if language is OpenCL_Code and fallback:
      return language_supported or self.is_map_supported_operation(self.simplified_rec_op_map, optree, language = C_Code, debug = debug, key_getter = key_getter)
    else:
      return language_supported

  def get_recursive_implementation(self, optree, language = None, table_getter = lambda self: self.code_generation_table,  key_getter = lambda self, optree: self.get_operation_keys(optree)):
    """ overloading of AbstractBackend's get_recursive_implementation method
        to allow fallback to C implementation if no OpenCL implementation
        is found """
    possible_impl = ImplemList()
    if self.is_local_supported_operation(optree, language = language, table_getter = table_getter, key_getter = key_getter):
      local_implementation = self.get_implementation(optree, language, table_getter = table_getter, key_getter = key_getter)
      if is_impl_list(local_implementation):
        possible_impl += local_implementation
      else:
        return local_implementation
    else:
      for parent_proc in self.parent_architecture:
        if parent_proc.is_local_supported_operation(optree, language = language, table_getter = table_getter, key_getter = key_getter):
          parent_implementation = parent_proc.get_implementation(optree, language, table_getter = table_getter, key_getter = key_getter)
          if is_impl_list(parent_implementation):
            possible_impl += parent_implementation
          else:
            return parent_implementation

    if len(possible_impl) > 0:
        # if there is at least one OpenCL implementation, even a weak match
        # we return it
        match, implementation = possible_impl[0]
        return implementation

    ## fallback to C_Code when no OpenCL_Code implementation is found
    if language is OpenCL_Code:
      return self.get_recursive_implementation(optree, C_Code, table_getter, key_getter = key_getter)

    # no implementation were found
    Log.report(Log.Verbose, "[VectorBackend] Tested architecture(s) for language %s:" % str(language))
    for parent_proc in self.parent_architecture:
      Log.report(Log.Verbose, "  %s " % parent_proc)
    Log.report(Log.Error, "the following operation is not supported by vector_backend %s: \n%s" % (self.__class__, optree.get_str(depth = 2, display_precision = True, memoization_map = {})))

  def instanciate_pass_pipeline(self, pass_scheduler, processor, extra_passes, language=C_Code):
    """ instanciate an optimization pass pipeline for VectorBackend targets """
    EXTRA_VECTOR_PASSES = ["beforecodegen:vector_mask_test_legalization", "beforecodegen:virtual_vector_bool_legalization"]
    return GenericProcessor.instanciate_pass_pipeline(self, pass_scheduler, processor,
                                                      EXTRA_VECTOR_PASSES + extra_passes,
                                                      language=language)

# debug message
Log.report(LOG_BACKEND_INIT, "Initializing vector backend target")
