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
from metalibm_core.core.ml_operations import *
from metalibm_core.code_generation.generic_processor import GenericProcessor, LibFunctionConstructor

from metalibm_core.core.target import TargetRegister


ML_VectorLib_Function = LibFunctionConstructor(["support_lib/ml_vector_lib.h"])

# OpenCL vector support library 
ML_OCL_VectorLib_Function = LibFunctionConstructor(["support_lib/ml_ocl_vector_lib.h"])

OpenCL_Builtin = LibFunctionConstructor([])

vector_type = {
  ML_Binary32: {
    2: ML_Float2,
    3: ML_Float3,
    4: ML_Float4,
    8: ML_Float8
  },
  ML_Binary64: {
    2: ML_Double2,
    3: ML_Double3,
    4: ML_Double4,
    8: ML_Double8
  },
  ML_Int32: {
    2: ML_Int2,
    3: ML_Int3,
    4: ML_Int4,
    8: ML_Int8
  },
  ML_UInt32: {
    2: ML_UInt2,
    3: ML_UInt3,
    4: ML_UInt4,
    8: ML_UInt8
  },
  ML_Bool: {
    2: ML_Bool2,
    3: ML_Bool3,
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

supported_vector_size = [2, 3, 4, 8]

vector_opencl_code_generation_table = {

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
        type_strict_match(ML_Int2, ML_Float2): OpenCL_Builtin("nearbyint", arity = 1, output_precision = ML_Int2),
        type_strict_match(ML_Int3, ML_Float3): OpenCL_Builtin("nearbyint", arity = 1, output_precision = ML_Int3),
        type_strict_match(ML_Int4, ML_Float4): OpenCL_Builtin("nearbyint", arity = 1, output_precision = ML_Int4),
        type_strict_match(ML_Int8, ML_Float8): OpenCL_Builtin("nearbyint", arity = 1, output_precision = ML_Int8),

        type_strict_match(ML_Float2, ML_Float2): OpenCL_Builtin("rint", arity = 1, output_precision = ML_Float2),
        type_strict_match(ML_Float3, ML_Float3): OpenCL_Builtin("rint", arity = 1, output_precision = ML_Float3),
        type_strict_match(ML_Float4, ML_Float4): OpenCL_Builtin("rint", arity = 1, output_precision = ML_Float4),
        type_strict_match(ML_Float8, ML_Float8): OpenCL_Builtin("rint", arity = 1, output_precision = ML_Float8),
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
        type_strict_match(ML_Bool2, ML_Bool2): SymbolOperator("!", arity = 1),
        type_strict_match(ML_Bool3, ML_Bool3): SymbolOperator("!", arity = 1),
        type_strict_match(ML_Bool4, ML_Bool4): SymbolOperator("!", arity = 1),
        type_strict_match(ML_Bool8, ML_Bool8): SymbolOperator("!", arity = 1),
      },
    },
  },
  LogicalAnd: {
    None: {
      lambda _: True: {
        type_strict_match(ML_Bool2, ML_Bool2, ML_Bool2): SymbolOperator("&&", arity = 2),
        type_strict_match(ML_Bool3, ML_Bool3, ML_Bool3): SymbolOperator("&&", arity = 2),
        type_strict_match(ML_Bool4, ML_Bool4, ML_Bool4): SymbolOperator("&&", arity = 2),
        type_strict_match(ML_Bool8, ML_Bool8, ML_Bool8): SymbolOperator("&&", arity = 2),
      },
    },
  },
  LogicalOr: {
    None: {
      lambda _: True: {
        type_strict_match(ML_Bool2, ML_Bool2, ML_Bool2): SymbolOperator("||", arity = 2),
        type_strict_match(ML_Bool3, ML_Bool3, ML_Bool3): SymbolOperator("||", arity = 2),
        type_strict_match(ML_Bool4, ML_Bool4, ML_Bool4): SymbolOperator("||", arity = 2),
        type_strict_match(ML_Bool8, ML_Bool8, ML_Bool8): SymbolOperator("||", arity = 2),
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

              type_strict_match(ML_Float2, ML_Int2) : ML_VectorLib_Function("as_float2", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = ML_Float2),
              type_strict_match(ML_Int2, ML_Float2) : ML_VectorLib_Function("as_int2", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = ML_Int2),

              type_strict_match(ML_Float3, ML_Int3) : ML_VectorLib_Function("as_float3", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = ML_Float3),
              type_strict_match(ML_Int3, ML_Float3) : ML_VectorLib_Function("as_int3", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = ML_Int3),

              type_strict_match(ML_Float4, ML_Int4) : ML_VectorLib_Function("as_float4", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = ML_Float4),
              type_strict_match(ML_Int4, ML_Float4) : ML_VectorLib_Function("as_int4", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = ML_Int4),

  }}},
  ExponentExtraction: {
      None: {
          lambda optree: True: {
              type_strict_match(ML_Int2, ML_Float2) : ML_OCL_VectorLib_Function("ml_ocl_exp_extraction_vf2", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = ML_Int4),
              type_strict_match(ML_Int3, ML_Float3) : ML_OCL_VectorLib_Function("ml_ocl_exp_extraction_vf3", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = ML_Int3),
              type_strict_match(ML_Int4, ML_Float4) : ML_OCL_VectorLib_Function("ml_ocl_exp_extraction_vf4", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = ML_Int4),
              type_strict_match(ML_Int8, ML_Float8) : ML_OCL_VectorLib_Function("ml_ocl_exp_extraction_vf8", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = ML_Int4),
  }}},
  ExponentInsertion: {
    ExponentInsertion.Default: {
          lambda optree: True: {
              type_strict_match(ML_Float2, ML_Int2) : ML_OCL_VectorLib_Function("ml_ocl_exp_insertion_vf2", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = ML_Int4),
              type_strict_match(ML_Float3, ML_Int3) : ML_OCL_VectorLib_Function("ml_ocl_exp_insertion_vf3", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = ML_Int3),
              type_strict_match(ML_Float4, ML_Int4) : ML_OCL_VectorLib_Function("ml_ocl_exp_insertion_vf4", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = ML_Int4),
              type_strict_match(ML_Float8, ML_Int8) : ML_OCL_VectorLib_Function("ml_ocl_exp_insertion_vf8", arg_map = {0: FO_Arg(0)}, arity = 1, output_precision = ML_Int4),
  }}},

  Test: {
    Test.IsMaskAllZero: {
      lambda _: True: {
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool2]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask2_zero", arity = 1, output_precision = ML_Int32), 
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool3]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask3_zero", arity = 1, output_precision = ML_Int32), 
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool4]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask4_zero", arity = 1, output_precision = ML_Int32), 
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool8]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask8_zero", arity = 1, output_precision = ML_Int32), 
      },
    },
    Test.IsMaskAnyZero: {
      lambda _: True: {
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool2]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask2_any_zero", arity = 1, output_precision = ML_Int32), 
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool3]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask3_any_zero", arity = 1, output_precision = ML_Int32), 
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool4]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask4_any_zero", arity = 1, output_precision = ML_Int32), 
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool8]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask8_any_zero", arity = 1, output_precision = ML_Int32), 
      },
    },
    Test.IsMaskNotAnyZero: {
      lambda _: True: {
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool2]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask2_not_any_zero", arity = 1, output_precision = ML_Int32), 
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool3]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask3_not_any_zero", arity = 1, output_precision = ML_Int32), 
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool4]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask4_not_any_zero", arity = 1, output_precision = ML_Int32), 
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool8]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask8_not_any_zero", arity = 1, output_precision = ML_Int32), 
      },
    },
    Test.IsMaskNotAllZero: {
      lambda _: True: {
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool2]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask2_not_all_zero", arity = 1, output_precision = ML_Int32), 
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool3]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask3_not_all_zero", arity = 1, output_precision = ML_Int32), 
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool4]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask4_not_all_zero", arity = 1, output_precision = ML_Int32), 
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool8]): ML_OCL_VectorLib_Function("ml_ocl_is_vmask8_not_all_zero", arity = 1, output_precision = ML_Int32), 
      },
    },
    Test.IsInfOrNaN: {
      lambda _: True: {
        type_strict_match(ML_Bool2, ML_Float2): ML_OCL_VectorLib_Function("ml_ocl_vtestf2_is_nan_or_inf", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Int2),
        type_strict_match(ML_Bool4, ML_Float3): ML_OCL_VectorLib_Function("ml_ocl_vtestf4_is_nan_or_inf", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Int3),
        type_strict_match(ML_Bool4, ML_Float4): ML_OCL_VectorLib_Function("ml_ocl_vtestf4_is_nan_or_inf", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Int4),
        type_strict_match(ML_Bool8, ML_Float8): ML_OCL_VectorLib_Function("ml_ocl_vtestf8_is_nan_or_inf", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Int8),
      },
    }
  },
}

vector_c_code_generation_table = {
  BitLogicAnd: {
    None: {
       lambda _: True: {
        type_strict_match(ML_Int2, ML_Int2, ML_Int2): ML_VectorLib_Function("ml_vbwandi2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2),
        type_strict_match(ML_Int3, ML_Int3, ML_Int3): ML_VectorLib_Function("ml_vbwandi4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Int3),
        type_strict_match(ML_Int4, ML_Int4, ML_Int4): ML_VectorLib_Function("ml_vbwandi4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Int4),
        type_strict_match(ML_Int8, ML_Int8, ML_Int8): ML_VectorLib_Function("ml_vbwandi8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Int8),
        },
      },
  },
  Select: {
    None: {
       lambda _: True: {
        type_strict_match(ML_Int2, ML_Bool2, ML_Int2, ML_Int2): ML_VectorLib_Function("ML_VSELECT", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "2"}, arity = 3),
        type_strict_match(ML_Int3, ML_Bool3, ML_Int3, ML_Int3): ML_VectorLib_Function("ML_VSELECT", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "4"}, arity = 3, output_precision = ML_Int3),
        type_strict_match(ML_Int4, ML_Bool4, ML_Int4, ML_Int4): ML_VectorLib_Function("ML_VSELECT", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "4"}, arity = 3, output_precision = ML_Int4),
        type_strict_match(ML_Int8, ML_Bool8, ML_Int8, ML_Int8): ML_VectorLib_Function("ML_VSELECT", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2), 4: "8"}, arity = 3, output_precision = ML_Int8),
      },
    },
  },
  Modulo: {
    None: {
       lambda _: True: {
        type_strict_match(ML_Int2, ML_Int2, ML_Int2): ML_VectorLib_Function("ml_vmodi2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2),
        type_strict_match(ML_Int3, ML_Int3, ML_Int3): ML_VectorLib_Function("ml_vmodi4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Int3),
        type_strict_match(ML_Int4, ML_Int4, ML_Int4): ML_VectorLib_Function("ml_vmodi4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Int4),
        type_strict_match(ML_Int8, ML_Int8, ML_Int8): ML_VectorLib_Function("ml_vmodi8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Int8),
      },
    },
  },
  Division: {
    None: {
       lambda _: True: {
        type_strict_match(ML_Int2, ML_Int2, ML_Int2): ML_VectorLib_Function("ml_vdivi2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2),
        type_strict_match(ML_Int3, ML_Int3, ML_Int3): ML_VectorLib_Function("ml_vdivi4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Int3),
        type_strict_match(ML_Int4, ML_Int4, ML_Int4): ML_VectorLib_Function("ml_vdivi4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Int4),
        type_strict_match(ML_Int8, ML_Int8, ML_Int8): ML_VectorLib_Function("ml_vdivi8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Int8),
      },
    },
  },
  Addition: {
    None: {
       lambda _: True: {
        type_strict_match(ML_Int2, ML_Int2, ML_Int2): ML_VectorLib_Function("ml_vaddi2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2),
        type_strict_match(ML_Int3, ML_Int3, ML_Int3): ML_VectorLib_Function("ml_vaddi4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Int3),
        type_strict_match(ML_Int4, ML_Int4, ML_Int4): ML_VectorLib_Function("ml_vaddi4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Int4),
        type_strict_match(ML_Int8, ML_Int8, ML_Int8): ML_VectorLib_Function("ml_vaddi8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Int8),

        type_strict_match(ML_Float2, ML_Float2, ML_Float2): ML_VectorLib_Function("ml_vaddf2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Float2),
        type_strict_match(ML_Float3, ML_Float3, ML_Float3): ML_VectorLib_Function("ml_vaddf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Float3),
        type_strict_match(ML_Float4, ML_Float4, ML_Float4): ML_VectorLib_Function("ml_vaddf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Float4),
        type_strict_match(ML_Float8, ML_Float8, ML_Float8): ML_VectorLib_Function("ml_vaddf8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Float8),
      },
    },
  },
  Subtraction: {
    None: {
       lambda _: True: {
        type_strict_match(ML_Float2, ML_Float2, ML_Float2): ML_VectorLib_Function("ml_vsubf2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Float2),
        type_strict_match(ML_Float3, ML_Float3, ML_Float3): ML_VectorLib_Function("ml_vsubf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Float3),
        type_strict_match(ML_Float4, ML_Float4, ML_Float4): ML_VectorLib_Function("ml_vsubf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Float4),
        type_strict_match(ML_Float8, ML_Float8, ML_Float8): ML_VectorLib_Function("ml_vsubf8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Float8),
      },
    },
  },
  #  Negate: {
  #    None: {
  #       lambda _: True: {
  #        type_strict_match(ML_Int2, ML_Int2): ML_VectorLib_Function("ml_vnegi2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1),
  #        type_strict_match(ML_Int4, ML_Int4): ML_VectorLib_Function("ml_vnegi4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Int4),
  #        type_strict_match(ML_Int8, ML_Int8): ML_VectorLib_Function("ml_vnegi8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Int8),
  #
  #        type_strict_match(ML_Float2, ML_Float2): ML_VectorLib_Function("ml_vnegf2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Float2),
  #        type_strict_match(ML_Float4, ML_Float4): ML_VectorLib_Function("ml_vnegf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Float4),
  #        type_strict_match(ML_Float8, ML_Float8): ML_VectorLib_Function("ml_vnegf8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Float8),
  #      },
  #    },
  #  },
  Multiplication: {
    None: {
       lambda _: True: {
        type_strict_match(ML_Float2, ML_Float2, ML_Float2): ML_VectorLib_Function("ml_vmulf2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Float2),
        type_strict_match(ML_Float3, ML_Float3, ML_Float3): ML_VectorLib_Function("ml_vmulf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Float3),
        type_strict_match(ML_Float4, ML_Float4, ML_Float4): ML_VectorLib_Function("ml_vmulf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Float4),
        type_strict_match(ML_Float8, ML_Float8, ML_Float8): ML_VectorLib_Function("ml_vmulf8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Float8),
      },
    },
  },
  FusedMultiplyAdd: {
    FusedMultiplyAdd.Standard: {
       lambda _: True: {
        type_strict_match(ML_Float2, ML_Float2, ML_Float2, ML_Float2): ML_VectorLib_Function("ml_vfmaf2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2)}, arity = 2, output_precision = ML_Float2),
        type_strict_match(ML_Float3, ML_Float3, ML_Float3, ML_Float3): ML_VectorLib_Function("ml_vfmaf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2)}, arity = 2, output_precision = ML_Float3),
        type_strict_match(ML_Float4, ML_Float4, ML_Float4, ML_Float4): ML_VectorLib_Function("ml_vfmaf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2)}, arity = 2, output_precision = ML_Float4),
        type_strict_match(ML_Float8, ML_Float8, ML_Float8, ML_Float8): ML_VectorLib_Function("ml_vfmaf8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1), 3: FO_Arg(2)}, arity = 2, output_precision = ML_Float8),
      },
    },
  },
  ExponentInsertion: {
    ExponentInsertion.Default: {
      lambda _: True: {
        type_strict_match(ML_Float2, ML_Int2): ML_VectorLib_Function("ml_vexp_insertion_f2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Float2),
        type_strict_match(ML_Float3, ML_Int3): ML_VectorLib_Function("ml_vexp_insertion_f4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Float3),
        type_strict_match(ML_Float4, ML_Int4): ML_VectorLib_Function("ml_vexp_insertion_f4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Float4),
        type_strict_match(ML_Float8, ML_Int8): ML_VectorLib_Function("ml_vexp_insertion_f8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Float8),
      }
    },
  },
  ExponentExtraction: {
    None: {
      lambda _: True: {
        type_strict_match(ML_Int2, ML_Float2): ML_VectorLib_Function("ml_vexp_extraction_f2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Int2),
        type_strict_match(ML_Int3, ML_Float3): ML_VectorLib_Function("ml_vexp_extraction_f4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Int3),
        type_strict_match(ML_Int4, ML_Float4): ML_VectorLib_Function("ml_vexp_extraction_f4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Int4),
        type_strict_match(ML_Int8, ML_Float8): ML_VectorLib_Function("ml_vexp_extraction_f8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Int8),
      }
    },
  },
  NearestInteger: {
    None: {
      lambda _: True : {
        type_strict_match(ML_Int2, ML_Float2): ML_VectorLib_Function("ml_vnearbyintf2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Int2),
        type_strict_match(ML_Int4, ML_Float4): ML_VectorLib_Function("ml_vnearbyintf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Int4),
        type_strict_match(ML_Int3, ML_Float3): ML_VectorLib_Function("ml_vnearbyintf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Int3),
        type_strict_match(ML_Int8, ML_Float8): ML_VectorLib_Function("ml_vnearbyintf8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Int8),

        type_strict_match(ML_Float2, ML_Float2): ML_VectorLib_Function("ml_vrintf2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Float2),
        type_strict_match(ML_Float3, ML_Float3): ML_VectorLib_Function("ml_vrintf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Float3),
        type_strict_match(ML_Float4, ML_Float4): ML_VectorLib_Function("ml_vrintf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Float4),
        type_strict_match(ML_Float8, ML_Float8): ML_VectorLib_Function("ml_vrintf8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Float8),
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
  TypeCast: {
      None: {
          lambda optree: True: {
              #type_strict_match(ML_Binary64, ML_Int64) : ML_Utils_Function("double_from_64b_encoding", arity = 1),
              #type_strict_match(ML_Binary64, ML_UInt64): ML_Utils_Function("double_from_64b_encoding", arity = 1),
              #type_strict_match(ML_Int64, ML_Binary64) : ML_Utils_Function("double_to_64b_encoding", arity = 1),
              #type_strict_match(ML_UInt64, ML_Binary64): ML_Utils_Function("double_to_64b_encoding", arity = 1),
              type_strict_match(ML_Float4, ML_Int4) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "float_from_32b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = ML_Float4),
              type_strict_match(ML_Int4, ML_Float4) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "float_to_32b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = ML_Int4),

              type_strict_match(ML_Float3, ML_Int3) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "float_from_32b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = ML_Float3),
              type_strict_match(ML_Int3, ML_Float3) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "float_to_32b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "4"}, arity = 1, output_precision = ML_Int3),

              type_strict_match(ML_Float2, ML_Int2) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "float_from_32b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "2"}, arity = 1, output_precision = ML_Float2),
              type_strict_match(ML_Int2, ML_Float2) : ML_VectorLib_Function("VECTORIZE_OP1", arg_map = {0: "float_to_32b_encoding", 1: FO_ResultRef(0), 2: FO_Arg(0), 3: "2"}, arity = 1, output_precision = ML_Int2),

              #ML_Utils_Function("float_from_32b_encoding", arity = 1),
              #type_strict_match(ML_Binary32, ML_UInt32): ML_Utils_Function("float_from_32b_encoding", arity = 1),
              #type_strict_match(ML_Int32, ML_Binary32) : ML_Utils_Function("float_to_32b_encoding", arity = 1),
              #type_strict_match(ML_UInt32, ML_Binary32): ML_Utils_Function("float_to_32b_encoding", arity = 1),
              #type_strict_match(ML_UInt64, ML_Binary32): ML_Utils_Function("(uint64_t) float_to_32b_encoding", arity = 1),
              #type_strict_match(ML_Binary32, ML_UInt64): ML_Utils_Function("float_from_32b_encoding", arity = 1),
          },
      },
  },


     #   type_strict_match(ML_Float2, ML_Float2): ML_VectorLib_Function("ml_vnegf2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Float2),
     #   type_strict_match(ML_Float4, ML_Float4): ML_VectorLib_Function("ml_vnegf4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Float4),
     #   type_strict_match(ML_Float8, ML_Float8): ML_VectorLib_Function("ml_vnegf8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Float8),
  #    }
  #  }
  #},
  VectorElementSelection: 
  {
    None: {
       lambda _: True: {
        lambda rformat, opformat, indexformat, optree: True: TemplateOperator("%s._[%s]", arity = 2),
      },
    },
  },
  LogicalNot: {
    None: {
      lambda _: True: {
        type_strict_match(ML_Bool2, ML_Bool2): ML_VectorLib_Function("ml_vnotb2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Int2),
        type_strict_match(ML_Bool3, ML_Bool3): ML_VectorLib_Function("ml_vnotb3", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Int3),
        type_strict_match(ML_Bool4, ML_Bool4): ML_VectorLib_Function("ml_vnotb4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Int4),
        type_strict_match(ML_Bool8, ML_Bool8): ML_VectorLib_Function("ml_vnotb8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Int8),
      },
    },
  },
  LogicalAnd: {
    None: {
      lambda _: True: {
        type_strict_match(ML_Bool2, ML_Bool2, ML_Bool2): ML_VectorLib_Function("ml_vandb2", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Int2),
        type_strict_match(ML_Bool3, ML_Bool3, ML_Bool3): ML_VectorLib_Function("ml_vandb3", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Int3),
        type_strict_match(ML_Bool4, ML_Bool4, ML_Bool4): ML_VectorLib_Function("ml_vandb4", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Int4),
        type_strict_match(ML_Bool8, ML_Bool8, ML_Bool8): ML_VectorLib_Function("ml_vandb8", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0), 2: FO_Arg(1)}, arity = 2, output_precision = ML_Int8),
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
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool2]): ML_VectorLib_Function("ml_is_vmask2_zero", arity = 1, output_precision = ML_Int32), 
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool3]): ML_VectorLib_Function("ml_is_vmask3_zero", arity = 1, output_precision = ML_Int32), 
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool4]): ML_VectorLib_Function("ml_is_vmask4_zero", arity = 1, output_precision = ML_Int32), 
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool8]): ML_VectorLib_Function("ml_is_vmask8_zero", arity = 1, output_precision = ML_Int32), 
      },
    },
    Test.IsMaskAnyZero: {
      lambda _: True: {
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool2]): ML_VectorLib_Function("ml_is_vmask2_any_zero", arity = 1, output_precision = ML_Int32), 
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool3]): ML_VectorLib_Function("ml_is_vmask3_any_zero", arity = 1, output_precision = ML_Int32), 
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool4]): ML_VectorLib_Function("ml_is_vmask4_any_zero", arity = 1, output_precision = ML_Int32), 
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool8]): ML_VectorLib_Function("ml_is_vmask8_any_zero", arity = 1, output_precision = ML_Int32), 
      },
    },
    Test.IsMaskNotAnyZero: {
      lambda _: True: {
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool2]): ML_VectorLib_Function("ml_is_vmask2_not_any_zero", arity = 1, output_precision = ML_Int32), 
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool3]): ML_VectorLib_Function("ml_is_vmask3_not_any_zero", arity = 1, output_precision = ML_Int32), 
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool4]): ML_VectorLib_Function("ml_is_vmask4_not_any_zero", arity = 1, output_precision = ML_Int32), 
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool8]): ML_VectorLib_Function("ml_is_vmask8_not_any_zero", arity = 1, output_precision = ML_Int32), 
      },
    },
    Test.IsMaskNotAllZero: {
      lambda _: True: {
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool2]): ML_VectorLib_Function("ml_is_vmask2_not_all_zero", arity = 1, output_precision = ML_Int32), 
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool3]): ML_VectorLib_Function("ml_is_vmask3_not_all_zero", arity = 1, output_precision = ML_Int32), 
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool4]): ML_VectorLib_Function("ml_is_vmask4_not_all_zero", arity = 1, output_precision = ML_Int32), 
        type_strict_match_list([ML_Bool, ML_Int32], [ML_Bool8]): ML_VectorLib_Function("ml_is_vmask8_not_all_zero", arity = 1, output_precision = ML_Int32), 
      },
    },
    Test.IsInfOrNaN: {
      lambda _: True: {
        type_strict_match(ML_Bool2, ML_Float2): ML_VectorLib_Function("ml_vtestf2_is_nan_or_inf", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Int2),
        type_strict_match(ML_Bool3, ML_Float3): ML_VectorLib_Function("ml_vtestf3_is_nan_or_inf", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Int3),
        type_strict_match(ML_Bool4, ML_Float4): ML_VectorLib_Function("ml_vtestf4_is_nan_or_inf", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Int4),
        type_strict_match(ML_Bool8, ML_Float8): ML_VectorLib_Function("ml_vtestf8_is_nan_or_inf", arg_map = {0: FO_ResultRef(0), 1: FO_Arg(0)}, arity = 1, output_precision = ML_Int8),
      },
    }
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


  def is_supported_operation(self, optree, language = C_Code, debug = False, fallback = True):
    """ return whether or not the operation performed by optree is supported by any level of the processor hierarchy """
    language_supported =  self.is_map_supported_operation(self.simplified_rec_op_map, optree, language, debug = debug)
    # fallback to C_Code
    if language is OpenCL_Code and fallback: 
      return language_supported or self.is_map_supported_operation(self.simplified_rec_op_map, optree, language = C_Code, debug = debug)
    else:
      return language_supported

  def get_recursive_implementation(self, optree, language = None, table_getter = lambda self: self.code_generation_table):
    if self.is_local_supported_operation(optree, language = language, table_getter = table_getter):
      local_implementation = self.get_implementation(optree, language, table_getter = table_getter)
      return local_implementation
    else:
      for parent_proc in self.parent_architecture:
        if parent_proc.is_local_supported_operation(optree, language = language, table_getter = table_getter):
          return parent_proc.get_implementation(optree, language, table_getter = table_getter)

    ## fallback to C_Code when no OpenCL_Code implementation is found
    if language is OpenCL_Code:
      return self.get_recursive_implementation(optree, C_Code, table_getter)

    # no implementation were found
    Log.report(Log.Verbose, "[VectorBackend] Tested architecture(s) for language %s:" % str(language))
    for parent_proc in self.parent_architecture:
      Log.report(Log.Verbose, "  %s " % parent_proc)
    Log.report(Log.Error, "the following operation is not supported by %s: \n%s" % (self.__class__, optree.get_str(depth = 2, display_precision = True, memoization_map = {}))) 

      

