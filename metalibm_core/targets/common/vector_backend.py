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


from metalibm_core.utility.log_report import *
from metalibm_core.code_generation.generator_utility import *
from metalibm_core.code_generation.complex_generator import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_table import ML_TableFormat
from metalibm_core.core.ml_operations import *
from metalibm_core.code_generation.generic_processor import GenericProcessor, LibFunctionConstructor

from metalibm_core.core.target import TargetRegister


ML_VectorLib_Function = LibFunctionConstructor(["support_lib/ml_vector_lib.h"])

# OpenCL vector support library 
ML_OCL_VectorLib_Function = LibFunctionConstructor(["support_lib/ml_ocl_vector_lib.h"])

OpenCL_Builtin = LibFunctionConstructor([])

vector_type = {
  ML_Binary32: {
    2: v2float32,
    3: v3float32,
    4: v4float32,
    8: v8float32
  },
  ML_Binary64: {
    2: v2float64,
    3: v3float64,
    4: v4float64,
    8: v8float64
  },
  ML_Int32: {
    2: v2int32,
    3: v3int32,
    4: v4int32,
    8: v8int32
  },
  ML_UInt32: {
    2: v2uint32,
    3: v3uint32,
    4: v4uint32,
    8: v8uint32
  },
  ML_Bool: {
    2: v2bool,
    3: v3bool,
    4: v4bool,
    8: v8bool
  },
}
scalar_type_letter = {
  ML_Binary32: "f",
  ML_Binary64: "d",
  ML_UInt32:   "u",
  ML_Int32:    "i",
}

supported_vector_size = [2, 3, 4, 8]

## Predicate to test if a VectorElementSelection
#  is legal for the vector_backend_target, i.e.
#  vector operand precision is a compound format
def legal_vector_element_selection(optree):
  compound_format = isinstance(
    optree.get_input(0).get_precision(), 
    ML_CompoundVectorFormat
  )
  return compound_format

vector_opencl_code_generation_table = {
  BitLogicLeftShift: {
    None: {
      lambda _: True: 
        dict(
          sum(
          [
            [
              (type_strict_match(
                  vector_type[scalar_type][vector_size], 
                  vector_type[scalar_type][vector_size],
                  vector_type[scalar_type][vector_size]
                ), SymbolOperator(" << ", arity = 2)
              ) for vector_size in supported_vector_size
            ] for scalar_type in [ML_Binary32, ML_Binary64, ML_Int32, ML_UInt32]
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
                  vector_type[scalar_type][vector_size], 
                  vector_type[scalar_type][vector_size],
                  vector_type[scalar_type][vector_size]
                ), SymbolOperator(" >> ", arity = 2)
              ) for vector_size in supported_vector_size
            ] for scalar_type in [ML_Binary32, ML_Binary64, ML_Int32, ML_UInt32]
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
                  vector_type[scalar_type][vector_size], 
                  vector_type[scalar_type][vector_size],
                  vector_type[scalar_type][vector_size]
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
                  vector_type[scalar_type][vector_size], 
                  vector_type[scalar_type][vector_size],
                  vector_type[scalar_type][vector_size]
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
                  vector_type[scalar_type][vector_size], 
                  vector_type[scalar_type][vector_size],
                  vector_type[scalar_type][vector_size]
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
                  vector_type[scalar_type][vector_size], 
                  vector_type[scalar_type][vector_size],
                  vector_type[scalar_type][vector_size]
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
                  vector_type[scalar_type][vector_size], 
                  vector_type[scalar_type][vector_size],
                  vector_type[scalar_type][vector_size]
                ), SymbolOperator("/", arity = 2)
              ) for vector_size in supported_vector_size
            ] for scalar_type in [ML_Binary32, ML_Binary64, ML_Int32, ML_UInt32]
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
                  vector_type[scalar_type][vector_size], 
                  vector_type[scalar_type][vector_size],
                  vector_type[scalar_type][vector_size]
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
                  vector_type[scalar_type][vector_size], 
                  vector_type[ML_Bool][vector_size], 
                  vector_type[scalar_type][vector_size],
                  vector_type[scalar_type][vector_size]
                ), TemplateOperator("%s ? %s : %s", arity = 3)
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
                  vector_type[scalar_type][vector_size], 
                  vector_type[scalar_type][vector_size],
                  vector_type[scalar_type][vector_size]
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
                  vector_type[scalar_type][vector_size],
                  vector_type[scalar_type][vector_size]
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
                  vector_type[scalar_type][vector_size],
                  vector_type[scalar_type][vector_size],
                  vector_type[scalar_type][vector_size],
                  vector_type[scalar_type][vector_size]
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
      lambda _: True: {
        type_strict_match(v2bool, v2bool, v2bool): SymbolOperator("||", arity = 2),
        type_strict_match(v3bool, v3bool, v3bool): SymbolOperator("||", arity = 2),
        type_strict_match(v4bool, v4bool, v4bool): SymbolOperator("||", arity = 2),
        type_strict_match(v8bool, v8bool, v8bool): SymbolOperator("||", arity = 2),
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
                      type_strict_match_list(
                        [
                          #vector_type[ML_Int32][vector_size], 
                          vector_type[ML_Bool][vector_size]
                        ],
                        [
                          vector_type[scalar_type][vector_size]
                        ],
                        [
                          vector_type[scalar_type][vector_size]
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
        type_strict_match_list([ML_Bool, ML_Int32], [v2bool]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask2_zero", arity = 1, output_precision = ML_Int32), 
        type_strict_match_list([ML_Bool, ML_Int32], [v3bool]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask3_zero", arity = 1, output_precision = ML_Int32), 
        type_strict_match_list([ML_Bool, ML_Int32], [v4bool]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask4_zero", arity = 1, output_precision = ML_Int32), 
        type_strict_match_list([ML_Bool, ML_Int32], [v8bool]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask8_zero", arity = 1, output_precision = ML_Int32), 
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

vector_c_code_generation_table = {
  VectorAssembling: {
    None: {
      lambda _: True: {
      },
    },
  },
  TableLoad: {
    None: {
      lambda _: True: {
        # single precision loading
        type_custom_match(FSM(v2float32), TCM(ML_TableFormat), FSM(v2int32)): ML_VectorLib_Function("ML_VLOAD", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: "2"}, arity = 4),
        type_custom_match(FSM(v2float32), TCM(ML_TableFormat), FSM(v2int32), FSM(v2int32)): ML_VectorLib_Function("ML_VLOAD2D", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "2"}, arity = 5),
        type_custom_match(FSM(v4float32), TCM(ML_TableFormat), FSM(v4int32)): ML_VectorLib_Function("ML_VLOAD", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: "4"}, arity = 4),
        type_custom_match(FSM(v4float32), TCM(ML_TableFormat), FSM(v4int32), FSM(v4int32)): ML_VectorLib_Function("ML_VLOAD2D", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "4"}, arity = 5),
        # double precision loading
        type_custom_match(FSM(v4float64), TCM(ML_TableFormat), FSM(v4int32)): ML_VectorLib_Function("ML_VLOAD", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: "4"}, arity = 4),
        type_custom_match(FSM(v4float64), TCM(ML_TableFormat), FSM(v4int32), FSM(v4int32)): ML_VectorLib_Function("ML_VLOAD2D", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "4"}, arity = 5),
        type_custom_match(FSM(v2float64), TCM(ML_TableFormat), FSM(v2int32)): ML_VectorLib_Function("ML_VLOAD", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: "2"}, arity = 4),
        type_custom_match(FSM(v2float64), TCM(ML_TableFormat), FSM(v2int32), FSM(v2int32)): ML_VectorLib_Function("ML_VLOAD2D", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "2"}, arity = 5),
      },
    },
  },
  BitLogicAnd: {
    None: {
       lambda _: True: {
        type_strict_match(v2int32, v2int32, v2int32): ML_VectorLib_Function("ml_vbwandi2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2),
        type_strict_match(v3int32, v3int32, v3int32): ML_VectorLib_Function("ml_vbwandi4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v3int32),
        type_strict_match(v4int32, v4int32, v4int32): ML_VectorLib_Function("ml_vbwandi4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v4int32),
        type_strict_match(v8int32, v8int32, v8int32): ML_VectorLib_Function("ml_vbwandi8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v8int32),
        # unsigned versions
        type_strict_match(v2uint32, v2uint32, v2uint32): ML_VectorLib_Function("ml_vbwandu2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2),
        type_strict_match(v3uint32, v3uint32, v3uint32): ML_VectorLib_Function("ml_vbwandu3", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2),
        type_strict_match(v4uint32, v4uint32, v4uint32): ML_VectorLib_Function("ml_vbwandu4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2),
        type_strict_match(v8uint32, v8uint32, v8uint32): ML_VectorLib_Function("ml_vbwandu8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2),
      },
    },
  },
  BitLogicLeftShift: {
    None: {
       lambda _: True: {
        type_strict_match(v4int32, v4int32, v4int32): ML_VectorLib_Function("ml_vslli4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v4int32),
        type_strict_match(v4uint32, v4uint32, v4uint32): ML_VectorLib_Function("ml_vsllu4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v4uint32),
      },
    },
  },
  BitLogicRightShift: {
    None: {
       lambda _: True: {
        type_strict_match(v2int32, v2int32, v2int32): ML_VectorLib_Function("ml_vsrai2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v2int32),
        type_strict_match(v4int32, v4int32, v4int32): ML_VectorLib_Function("ml_vsrai4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v4int32),
        type_strict_match(v4uint32, v4uint32, v4uint32): ML_VectorLib_Function("ml_vsrau4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v4uint32),
      },
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
        type_strict_match(v2float32, v2bool, v2float32, v2float32): ML_VectorLib_Function("ML_VSELECT", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "2"}, arity = 3, output_precision = v2float32),
        type_strict_match(v4float32, v4bool, v4float32, v4float32): ML_VectorLib_Function("ML_VSELECT", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "4"}, arity = 3, output_precision = v4float32),
      },
    },
  },
  Modulo: {
    None: {
       lambda _: True: {
        type_strict_match(v2int32, v2int32, v2int32): ML_VectorLib_Function("ml_vmodi2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2),
        type_strict_match(v3int32, v3int32, v3int32): ML_VectorLib_Function("ml_vmodi4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v3int32),
        type_strict_match(v4int32, v4int32, v4int32): ML_VectorLib_Function("ml_vmodi4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v4int32),
        type_strict_match(v8int32, v8int32, v8int32): ML_VectorLib_Function("ml_vmodi8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v8int32),
      },
    },
  },
  Division: {
    None: {
       lambda _: True: {
        type_strict_match(v2int32, v2int32, v2int32): ML_VectorLib_Function("ml_vdivi2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2),
        type_strict_match(v3int32, v3int32, v3int32): ML_VectorLib_Function("ml_vdivi4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v3int32),
        type_strict_match(v4int32, v4int32, v4int32): ML_VectorLib_Function("ml_vdivi4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v4int32),
        type_strict_match(v8int32, v8int32, v8int32): ML_VectorLib_Function("ml_vdivi8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v8int32),

        type_strict_match(v2float32, v2float32, v2float32): ML_VectorLib_Function("ml_vdivf2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2),
        type_strict_match(v3float32, v3float32, v3float32): ML_VectorLib_Function("ml_vdivf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v3float32),
        type_strict_match(v4float32, v4float32, v4float32): ML_VectorLib_Function("ml_vdivf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v4float32),
        type_strict_match(v8float32, v8float32, v8float32): ML_VectorLib_Function("ml_vdivf8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v8float32),
      },
    },
  },
  Addition: {
    None: {
       lambda _: True: {
        type_strict_match(v2int32, v2int32, v2int32): ML_VectorLib_Function("ml_vaddi2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2),
        type_strict_match(v3int32, v3int32, v3int32): ML_VectorLib_Function("ml_vaddi4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v3int32),
        type_strict_match(v4int32, v4int32, v4int32): ML_VectorLib_Function("ml_vaddi4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v4int32),
        type_strict_match(v8int32, v8int32, v8int32): ML_VectorLib_Function("ml_vaddi8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v8int32),
        # single precision
        type_strict_match(v2float32, v2float32, v2float32): ML_VectorLib_Function("ml_vaddf2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v2float32),
        type_strict_match(v3float32, v3float32, v3float32): ML_VectorLib_Function("ml_vaddf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v3float32),
        type_strict_match(v4float32, v4float32, v4float32): ML_VectorLib_Function("ml_vaddf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v4float32),
        type_strict_match(v8float32, v8float32, v8float32): ML_VectorLib_Function("ml_vaddf8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v8float32),
        # double precision
        type_strict_match(v2float64, v2float64, v2float64): ML_VectorLib_Function("ml_vaddd2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v2float64),
        type_strict_match(v3float64, v3float64, v3float64): ML_VectorLib_Function("ml_vaddd4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v3float64),
        type_strict_match(v4float64, v4float64, v4float64): ML_VectorLib_Function("ml_vaddd4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v4float64),
        type_strict_match(v8float64, v8float64, v8float64): ML_VectorLib_Function("ml_vaddd8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v8float64),
      },
    },
  },
  Subtraction: {
    None: {
       lambda _: True: {
        # single precision
        type_strict_match(v2float32, v2float32, v2float32): ML_VectorLib_Function("ml_vsubf2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v2float32),
        type_strict_match(v3float32, v3float32, v3float32): ML_VectorLib_Function("ml_vsubf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v3float32),
        type_strict_match(v4float32, v4float32, v4float32): ML_VectorLib_Function("ml_vsubf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v4float32),
        type_strict_match(v8float32, v8float32, v8float32): ML_VectorLib_Function("ml_vsubf8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v8float32),
        # double precision
        type_strict_match(v2float64, v2float64, v2float64): ML_VectorLib_Function("ml_vsubd2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v2float64),
        type_strict_match(v3float64, v3float64, v3float64): ML_VectorLib_Function("ml_vsubd4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v3float64),
        type_strict_match(v4float64, v4float64, v4float64): ML_VectorLib_Function("ml_vsubd4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v4float64),
        type_strict_match(v8float64, v8float64, v8float64): ML_VectorLib_Function("ml_vsubd8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v8float64),

        type_strict_match(v2uint32, v2uint32, v2uint32): ML_VectorLib_Function("ml_vsubu2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v2uint32),
        type_strict_match(v3uint32, v3uint32, v3uint32): ML_VectorLib_Function("ml_vsubu4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v3uint32),
        type_strict_match(v4uint32, v4uint32, v4uint32): ML_VectorLib_Function("ml_vsubu4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v4uint32),
        type_strict_match(v8uint32, v8uint32, v8uint32): ML_VectorLib_Function("ml_vsubu8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v8uint32),

        type_strict_match(v2int32, v2int32, v2int32): ML_VectorLib_Function("ml_vsubi2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v2int32),
        type_strict_match(v3int32, v3int32, v3int32): ML_VectorLib_Function("ml_vsubi4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v3int32),
        type_strict_match(v4int32, v4int32, v4int32): ML_VectorLib_Function("ml_vsubi4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v4int32),
        type_strict_match(v8int32, v8int32, v8int32): ML_VectorLib_Function("ml_vsubi8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v8int32),
      },
    },
  },
  #  Negate: {
  #    None: {
  #       lambda _: True: {
  #        type_strict_match(v2int32, v2int32): ML_VectorLib_Function("ml_vnegi2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1),
  #        type_strict_match(v4int32, v4int32): ML_VectorLib_Function("ml_vnegi4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v4int32),
  #        type_strict_match(v8int32, v8int32): ML_VectorLib_Function("ml_vnegi8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v8int32),
  #
  #        type_strict_match(v2float32, v2float32): ML_VectorLib_Function("ml_vnegf2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v2float32),
  #        type_strict_match(v4float32, v4float32): ML_VectorLib_Function("ml_vnegf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v4float32),
  #        type_strict_match(v8float32, v8float32): ML_VectorLib_Function("ml_vnegf8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v8float32),
  #      },
  #    },
  #  },
  Multiplication: {
    None: {
       lambda _: True: {
        # single precision
        type_strict_match(v2float32, v2float32, v2float32): ML_VectorLib_Function("ml_vmulf2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v2float32),
        type_strict_match(v3float32, v3float32, v3float32): ML_VectorLib_Function("ml_vmulf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v3float32),
        type_strict_match(v4float32, v4float32, v4float32): ML_VectorLib_Function("ml_vmulf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v4float32),
        type_strict_match(v8float32, v8float32, v8float32): ML_VectorLib_Function("ml_vmulf8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v8float32),
        # double precision
        type_strict_match(v2float64, v2float64, v2float64): ML_VectorLib_Function("ml_vmuld2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v2float64),
        type_strict_match(v3float64, v3float64, v3float64): ML_VectorLib_Function("ml_vmuld4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v3float64),
        type_strict_match(v4float64, v4float64, v4float64): ML_VectorLib_Function("ml_vmuld4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v4float64),
      },
    },
  },
  FusedMultiplyAdd: {
    FusedMultiplyAdd.Standard: {
       lambda _: True: {
        type_strict_match(v2float32, v2float32, v2float32, v2float32): ML_VectorLib_Function("ml_vfmaf2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2)}, arity = 2, output_precision = v2float32),
        type_strict_match(v3float32, v3float32, v3float32, v3float32): ML_VectorLib_Function("ml_vfmaf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2)}, arity = 2, output_precision = v3float32),
        type_strict_match(v4float32, v4float32, v4float32, v4float32): ML_VectorLib_Function("ml_vfmaf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2)}, arity = 2, output_precision = v4float32),
        type_strict_match(v8float32, v8float32, v8float32, v8float32): ML_VectorLib_Function("ml_vfmaf8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2)}, arity = 2, output_precision = v8float32),
      },
    },
  },
  ExponentInsertion: {
    ExponentInsertion.Default: {
      lambda _: True: {
        type_strict_match(v2float32, v2int32): ML_VectorLib_Function("ml_vexp_insertion_f2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v2float32),
        type_strict_match(v3float32, v3int32): ML_VectorLib_Function("ml_vexp_insertion_f4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v3float32),
        type_strict_match(v4float32, v4int32): ML_VectorLib_Function("ml_vexp_insertion_f4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v4float32),
        type_strict_match(v8float32, v8int32): ML_VectorLib_Function("ml_vexp_insertion_f8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v8float32),
      }
    },
  },
  ExponentExtraction: {
    None: {
      lambda _: True: {
        type_strict_match(v2int32, v2float32): ML_VectorLib_Function("ml_vexp_extraction_f2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v2int32),
        type_strict_match(v3int32, v3float32): ML_VectorLib_Function("ml_vexp_extraction_f4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v3int32),
        type_strict_match(v4int32, v4float32): ML_VectorLib_Function("ml_vexp_extraction_f4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v4int32),
        type_strict_match(v8int32, v8float32): ML_VectorLib_Function("ml_vexp_extraction_f8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v8int32),
      }
    },
  },
  MantissaExtraction: {
    None: {
      lambda _: True: {
        type_strict_match(v2float32, v2float32): ML_VectorLib_Function("ml_vmantissa_extraction_f2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v2int32),
        type_strict_match(v3float32, v3float32): ML_VectorLib_Function("ml_vmantissa_extraction_f4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v3int32),
        type_strict_match(v4float32, v4float32): ML_VectorLib_Function("ml_vmantissa_extraction_f4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v4int32),
        type_strict_match(v8float32, v8float32): ML_VectorLib_Function("ml_vmantissa_extraction_f8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v8int32),
      }
    },
  },
  NearestInteger: {
    None: {
      lambda _: True : {
        type_strict_match(v2int32, v2float32): ML_VectorLib_Function("ml_vnearbyintf2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v2int32),
        type_strict_match(v4int32, v4float32): ML_VectorLib_Function("ml_vnearbyintf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v4int32),
        type_strict_match(v3int32, v3float32): ML_VectorLib_Function("ml_vnearbyintf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v3int32),
        type_strict_match(v8int32, v8float32): ML_VectorLib_Function("ml_vnearbyintf8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v8int32),

        type_strict_match(v2float32, v2float32): ML_VectorLib_Function("ml_vrintf2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v2float32),
        type_strict_match(v3float32, v3float32): ML_VectorLib_Function("ml_vrintf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v3float32),
        type_strict_match(v4float32, v4float32): ML_VectorLib_Function("ml_vrintf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v4float32),
        type_strict_match(v8float32, v8float32): ML_VectorLib_Function("ml_vrintf8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v8float32),
      }
    }
  },
  Negation: {
    None: {
      lambda _: True: 
        dict(
          sum(
          [
            [
              (
                type_strict_match(vector_type[scalar_type][vector_size], vector_type[scalar_type][vector_size])
              , 
                ML_VectorLib_Function("ml_vneg%s%d" % (scalar_type_letter[scalar_type], vector_size), arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = vector_type[scalar_type][vector_size])
              ) for vector_size in supported_vector_size
            ] for scalar_type in [ML_Binary32, ML_Binary64, ML_Int32, ML_UInt32]
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
        # conversion from and to double precision
        type_strict_match(v2float64, v2float32) :  ML_VectorLib_Function("ML_VCONV", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: "2"}, arity = 3),
        type_strict_match(v2float32, v2float64) :  ML_VectorLib_Function("ML_VCONV", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: "2"}, arity = 3),
        type_strict_match(v4float32, v4float64) :  ML_VectorLib_Function("ML_VCONV", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: "4"}, arity = 3),
        type_strict_match(v4float64, v4float32) :  ML_VectorLib_Function("ML_VCONV", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: "4"}, arity = 3),
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
        type_strict_match(v4float32, v4float32) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "fabsf", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = v4float32, require_header = ["math.h"]),
      },
    },
  },
  Trunc: {
    None: {
      lambda optree: True: {
        type_strict_match(v2float32, v2float32) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "truncf", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "2"}, arity = 1, output_precision = v2float32, require_header = ["math.h"]),
        type_strict_match(v3float32, v3float32) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "truncf", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "3"}, arity = 1, output_precision = v3float32, require_header = ["math.h"]),
        type_strict_match(v4float32, v4float32) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "truncf", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = v4float32, require_header = ["math.h"]),
        type_strict_match(v8float32, v8float32) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "truncf", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "8"}, arity = 1, output_precision = v8float32, require_header = ["math.h"]),
      },
    },
  },
  TypeCast: {
      None: {
          lambda optree: True: {
              #type_strict_match(ML_Binary64, ML_Int64) : ML_Utils_Function("double_from_64b_encoding", arity = 1),
              #type_strict_match(ML_Binary64, ML_UInt64): ML_Utils_Function("double_from_64b_encoding", arity = 1),
              #type_strict_match(ML_Int64, ML_Binary64) : ML_Utils_Function("double_to_64b_encoding", arity = 1),
              #type_strict_match(ML_UInt64, ML_Binary64): ML_Utils_Function("double_to_64b_encoding", arity = 1),
              type_strict_match(v8float32, v8int32) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "float_from_32b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = v8float32),
              type_strict_match(v8int32, v8float32) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "float_to_32b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = v8int32),

              type_strict_match(v4float32, v4int32) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "float_from_32b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = v4float32),
              type_strict_match(v4int32, v4float32) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "float_to_32b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = v4int32),

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
              #ML_Utils_Function("float_from_32b_encoding", arity = 1),
              #type_strict_match(ML_Binary32, ML_UInt32): ML_Utils_Function("float_from_32b_encoding", arity = 1),
              #type_strict_match(ML_Int32, ML_Binary32) : ML_Utils_Function("float_to_32b_encoding", arity = 1),
              #type_strict_match(ML_UInt32, ML_Binary32): ML_Utils_Function("float_to_32b_encoding", arity = 1),
              #type_strict_match(ML_UInt64, ML_Binary32): ML_Utils_Function("(uint64_t) float_to_32b_encoding", arity = 1),
              #type_strict_match(ML_Binary32, ML_UInt64): ML_Utils_Function("float_from_32b_encoding", arity = 1),
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
        lambda rformat, opformat, indexformat, optree: True: TemplateOperator("%s._[%s]", arity = 2, require_header = ["support_lib/ml_vector_format.h"]),
      },
    },
  },
  LogicalNot: {
    None: {
      lambda _: True: {
        type_strict_match(v2bool, v2bool): ML_VectorLib_Function("ml_vnotb2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v2int32),
        type_strict_match(v3bool, v3bool): ML_VectorLib_Function("ml_vnotb3", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v3int32),
        type_strict_match(v4bool, v4bool): ML_VectorLib_Function("ml_vnotb4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v4int32),
        type_strict_match(v8bool, v8bool): ML_VectorLib_Function("ml_vnotb8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v8int32),
      },
    },
  },
  LogicalOr: {
    None: {
      lambda _: True: {
        type_strict_match(v2bool, v2bool, v2bool): ML_VectorLib_Function("ml_vorb2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v2int32),
        type_strict_match(v3bool, v3bool, v3bool): ML_VectorLib_Function("ml_vorb3", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v3int32),
        type_strict_match(v4bool, v4bool, v4bool): ML_VectorLib_Function("ml_vorb4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v4int32),
        type_strict_match(v8bool, v8bool, v8bool): ML_VectorLib_Function("ml_vorb8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v8int32),
      },
    },
  },
  LogicalAnd: {
    None: {
      lambda _: True: {
        type_strict_match(v2bool, v2bool, v2bool): ML_VectorLib_Function("ml_vandb2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v2int32),
        type_strict_match(v3bool, v3bool, v3bool): ML_VectorLib_Function("ml_vandb3", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v3int32),
        type_strict_match(v4bool, v4bool, v4bool): ML_VectorLib_Function("ml_vandb4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v4int32),
        type_strict_match(v8bool, v8bool, v8bool): ML_VectorLib_Function("ml_vandb8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = v8int32),
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
                      type_strict_match_list(
                        [
                          #vector_type[ML_Int32][vector_size], 
                          vector_type[ML_Bool][vector_size]
                        ],
                        [
                          vector_type[scalar_type][vector_size]
                        ],
                        [
                          vector_type[scalar_type][vector_size]
                        ]
                      )
                      , 
                      ML_VectorLib_Function("ml_comp_%s_%s%d" % (comp_specifier.opcode, scalar_type_letter[scalar_type], vector_size), arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = vector_type[ML_Bool][vector_size])
                    )  for scalar_type in [ML_Binary32, ML_Binary64, ML_Int32, ML_UInt32]
                  ] for vector_size in supported_vector_size
                ], []
              )
            )
          )
      }
    ) for comp_specifier in [Comparison.Equal, Comparison.NotEqual, Comparison.Greater, Comparison.GreaterOrEqual, Comparison.Less, Comparison.LessOrEqual]
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
      },
    },
    
    Test.IsSubnormal: {
      lambda _: True: {
        type_strict_match(v2bool, v2float32): ML_VectorLib_Function("ml_vtestf2_is_subnormal", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v2int32),
        type_strict_match(v3bool, v3float32): ML_VectorLib_Function("ml_vtestf3_is_subnormal", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v3int32),
        type_strict_match(v4bool, v4float32): ML_VectorLib_Function("ml_vtestf4_is_subnormal", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v4int32),
        type_strict_match(v8bool, v8float32): ML_VectorLib_Function("ml_vtestf8_is_subnormal", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = v8int32),
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
    if self.is_local_supported_operation(optree, language = language, table_getter = table_getter, key_getter = key_getter):
      local_implementation = self.get_implementation(optree, language, table_getter = table_getter, key_getter = key_getter)
      return local_implementation
    else:
      for parent_proc in self.parent_architecture:
        if parent_proc.is_local_supported_operation(optree, language = language, table_getter = table_getter, key_getter = key_getter):
          return parent_proc.get_implementation(optree, language, table_getter = table_getter, key_getter = key_getter)

    ## fallback to C_Code when no OpenCL_Code implementation is found
    if language is OpenCL_Code:
      return self.get_recursive_implementation(optree, C_Code, table_getter, key_getter = key_getter)

    # no implementation were found
    Log.report(Log.Verbose, "[VectorBackend] Tested architecture(s) for language %s:" % str(language))
    for parent_proc in self.parent_architecture:
      Log.report(Log.Verbose, "  %s " % parent_proc)
    Log.report(Log.Error, "the following operation is not supported by %s: \n%s" % (self.__class__, optree.get_str(depth = 2, display_precision = True, memoization_map = {}))) 

      
# debug message
print "Initializing vector backend target"
