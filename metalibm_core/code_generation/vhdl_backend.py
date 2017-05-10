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
from .complex_generator import *
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


## Copy the value of the init_stage attribute field
#  from @p src node to @p dst node
def copy_init_stage(src, dst):
  init_stage = src.attributes.get_dyn_attribute("init_stage")
  dst.attributes.init_stage = init_stage
  

def zext_modifier(optree):
  init_stage = optree.attributes.get_dyn_attribute("init_stage")
  
  ext_input = optree.get_input(0)
  ext_size = optree.ext_size
  assert ext_size >= 0
  if ext_size == 0:
    Log.report(Log.Warning, "zext_modifer called with ext_size=0 on {}".format(optree.get_str()))
    return ext_input
  else:
    precision = ML_StdLogicVectorFormat(ext_size)
    ext_precision = ML_StdLogicVectorFormat(ext_size + ext_input.get_precision().get_bit_size())
    result = Concatenation(Constant(0, precision = precision), ext_input, precision = ext_precision, tag = optree.get_tag(), init_stage = init_stage)
    copy_init_stage(optree, result)
    return result

## Operation code generator modifier for Sign Extension
def sext_modifier(optree):
  init_stage = optree.attributes.get_dyn_attribute("init_stage")

  ext_size = optree.ext_size
  ext_input = optree.get_input(0)
  if ext_size == 0:
    Log.report(Log.Warning, "zext_modifer called with ext_size=0 on {}".format(optree.get_str()))
    return ext_input
  else:
    ext_precision = ML_StdLogicVectorFormat(ext_size + ext_input.get_precision().get_bit_size())
    op_size = ext_input.get_precision().get_bit_size()
    sign_digit = VectorElementSelection(ext_input, Constant(op_size -1, precision = ML_Integer), precision = ML_StdLogic, init_stage = init_stage)
    precision = ML_StdLogicVectorFormat(ext_size)
    return Concatenation(Replication(sign_digit, precision = precision, init_stage = init_stage), optree, precision = ext_precision, tag = optree.get_tag(), init_stage = init_stage)

def negation_modifer(optree):
  init_stage = optree.attributes.get_dyn_attribute("init_stage")

  neg_input = optree.get_input(0)
  precision = optree.get_precision()
  return Addition(
    BitLogicNegate(neg_input, precision = precision, init_stage = init_stage),
    Constant(1, precision = ML_StdLogic),
    precision = precision,
    tag = optree.get_tag(),
    init_stage = init_stage
  )

def mantissa_extraction_modifier(optree):
  init_stage = optree.attributes.get_dyn_attribute("init_stage")
  op = optree.get_input(0)

  op_precision = op.get_precision().get_base_format()
  exp_prec = ML_StdLogicVectorFormat(op_precision.get_exponent_size())
  field_prec = ML_StdLogicVectorFormat(op_precision.get_field_size())

  exp_op   = ExponentExtraction(op, precision = exp_prec, init_stage = init_stage)
  field_op = SubSignalSelection(
    TypeCast(
      op,
      precision = op.get_precision().get_support_format(),
      init_stage = init_stage
    )
    , 0, op_precision.get_field_size() - 1, precision = field_prec,
    init_stage = init_stage
  ) 

  implicit_digit = Select(
    Comparison(
      exp_op,
      Constant(
        op_precision.get_zero_exponent_value(),
        precision = exp_prec,
        init_stage = init_stage
      ),
      precision = ML_Bool,
      specifier = Comparison.Equal,
      init_stage = init_stage
    ),
    Constant(0, precision = ML_StdLogic),
    Constant(1, precision = ML_StdLogic),
    precision = ML_StdLogic,
    tag = "implicit_digit",
    init_stage = init_stage
  )
  return Concatenation(
    implicit_digit,
    field_op,
    precision = ML_StdLogicVectorFormat(op_precision.get_mantissa_size()),
    tag = optree.get_tag(),
    debug = optree.get_debug(),
    init_stage = init_stage
  )

      

def truncate_generator(optree):
  truncate_input = optree.get_input(0)
  result_size = optree.get_precision().get_bit_size()
  return TemplateOperator("%%s(%d downto 0)" % (result_size - 1), arity = 1)

def conversion_generator(optree):
  output_size = optree.get_precision().get_bit_size()
  return TemplateOperator("std_logic_vector(to_unsigned(%s, {output_size}))".format(output_size = output_size), arity = 1)

def copy_sign_generator(optree):
  sign_input = optree.get_input(0)
  sign_index = sign_input.get_precision().get_bit_size() - 1
  return TemplateOperator("%%s(%d)" % (sign_index), arity = 1)

def sub_signal_generator(optree):
  sign_input = optree.get_input(0)
  inf_index = optree.get_inf_index()
  sup_index = optree.get_sup_index()
  return TemplateOperator("%s({sup_index} downto {inf_index})".format(inf_index = inf_index, sup_index = sup_index), arity = 1, force_folding = True)


## fixed point operation generation block
def fixed_point_op_modifier(optree, op_ctor = Addition):
  init_stage = optree.attributes.get_dyn_attribute("init_stage")

  # left hand side and right hand side operand extraction
  lhs = optree.get_input(0)
  rhs = optree.get_input(1)
  lhs_prec = lhs.get_precision().get_base_format()
  rhs_prec = rhs.get_precision().get_base_format()
  optree_prec = optree.get_precision().get_base_format()
  result_frac_size = max(lhs_prec.get_frac_size(), rhs_prec.get_frac_size(), optree_prec.get_frac_size())
  result_integer_size = max(lhs_prec.get_integer_size(), rhs_prec.get_integer_size(), optree_prec.get_integer_size())
  assert optree_prec.get_frac_size() >= result_frac_size
  assert optree_prec.get_integer_size() >= result_integer_size
  lhs_casted = TypeCast(lhs, precision = ML_StdLogicVectorFormat(lhs_prec.get_bit_size()), init_stage = init_stage)
  rhs_casted = TypeCast(rhs, precision = ML_StdLogicVectorFormat(rhs_prec.get_bit_size()), init_stage = init_stage)

  lhs_ext = (sext if lhs_prec.get_signed() else zext)(
    rzext(lhs_casted, result_frac_size - lhs_prec.get_frac_size()),
    result_integer_size - lhs_prec.get_integer_size()
  )
  lhs_ext = SignCast(lhs_ext, precision = lhs_ext.get_precision(), specifier = SignCast.Signed if lhs_prec.get_signed() else  SignCast.Unsigned)

  rhs_ext = (sext if rhs_prec.get_signed() else zext)(
    rzext(rhs_casted, result_frac_size - rhs_prec.get_frac_size()),
    result_integer_size - rhs_prec.get_integer_size()
  )
  rhs_ext = SignCast(rhs_ext, precision = rhs_ext.get_precision(), specifier = SignCast.Signed if rhs_prec.get_signed() else  SignCast.Unsigned)
  return TypeCast(
    op_ctor(
      lhs_ext,
      rhs_ext,
      precision = ML_StdLogicVectorFormat(optree_prec.get_bit_size()),
    ),
    init_stage = init_stage,
    precision = optree_prec,
  )

def fixed_point_add_modifier(optree):
  return fixed_point_op_modifier(optree, op_ctor = Addition)
def fixed_point_sub_modifier(optree):
  return fixed_point_op_modifier(optree, op_ctor = Subtraction)

def fixed_point_mul_modifier(optree):
  init_stage = optree.attributes.get_dyn_attribute("init_stage")

  # left hand side and right hand side operand extraction
  lhs = optree.get_input(0)
  rhs = optree.get_input(1)
  lhs_prec = lhs.get_precision().get_base_format()
  rhs_prec = rhs.get_precision().get_base_format()
  optree_prec = optree.get_precision().get_base_format()
  result_frac_size = max(lhs_prec.get_frac_size() + rhs_prec.get_frac_size(), optree_prec.get_frac_size())
  result_integer_size = max(lhs_prec.get_integer_size() +  rhs_prec.get_integer_size(), optree_prec.get_integer_size())
  assert optree_prec.get_frac_size() >= result_frac_size
  assert optree_prec.get_integer_size() >= result_integer_size
  lhs_casted = TypeCast(lhs, precision = ML_StdLogicVectorFormat(lhs_prec.get_bit_size()), init_stage = init_stage)
  lhs_casted = SignCast(lhs_casted, precision = lhs_casted.get_precision(), specifier = SignCast.Signed if lhs_prec.get_signed() else  SignCast.Unsigned)
  rhs_casted = TypeCast(rhs, precision = ML_StdLogicVectorFormat(rhs_prec.get_bit_size()), init_stage = init_stage)
  rhs_casted = SignCast(rhs_casted, precision = rhs_casted.get_precision(), specifier = SignCast.Signed if rhs_prec.get_signed() else  SignCast.Unsigned)

  mult_prec = ML_StdLogicVectorFormat(result_frac_size + result_integer_size)
  raw_result = Multiplication(
    lhs_casted,
    rhs_casted,
    precision = mult_prec,
    init_stage = init_stage
  )
  rext_result = rzext(raw_result, optree_prec.get_frac_size() - result_frac_size)
  result = (sext if (optree_prec.get_signed()) else zext)(rext_result, optree_prec.get_integer_size() - result_integer_size)
  return TypeCast(result, precision = optree_prec, init_stage = init_stage)

vhdl_comp_symbol = {
  Comparison.Equal: "=", 
  Comparison.NotEqual: "/=",
  Comparison.Less: "<",
  Comparison.LessOrEqual: "<=",
  Comparison.GreaterOrEqual: ">=",
  Comparison.Greater: ">",
  Comparison.LessSigned: "<",
  Comparison.LessOrEqualSigned: "<=",
  Comparison.GreaterOrEqualSigned: ">=",
  Comparison.GreaterSigned: ">",
}

def get_vhdl_bool_cst(self, value):
  if value:
    return "true"
  else:
    return "false"

## Updating standard format name for VHDL Code
ML_Integer.name[VHDL_Code] = "integer"
ML_Bool.name[VHDL_Code] = "boolean"
ML_Bool.get_cst_map[VHDL_Code] = get_vhdl_bool_cst

# class Match custom std logic vector format
MCSTDLOGICV = TCM(ML_StdLogicVectorFormat)

# class match custom fixed point format
MCFixedPoint = TCM(ML_Base_FixedPoint_Format)

formal_generation_table = {
  Addition: {
    None: {
      lambda optree: True: {
        type_strict_match(ML_Integer, ML_Integer, ML_Integer): SymbolOperator("+", arity = 2, force_folding = False),
      },
    },
  },
  Subtraction: {
    None: {
      lambda optree: True: {
        type_strict_match(ML_Integer, ML_Integer, ML_Integer): SymbolOperator("-", arity = 2, force_folding = False),
      },
    },
  },
  Multiplication: {
    None: {
      lambda optree: True: {
        type_strict_match(ML_Integer, ML_Integer, ML_Integer): SymbolOperator("*", arity = 2, force_folding = False),
      },
    },
  },
}

vhdl_code_generation_table = {
  Addition: {
    None: {
      exclude_std_logic: 
          build_simplified_operator_generation_nomap([v8int32, v8uint32, ML_Int16, ML_UInt16, ML_Int32, ML_UInt32, ML_Int64, ML_UInt64, ML_Int128,ML_UInt128], 2, SymbolOperator("+", arity = 2, force_folding = True), cond = (lambda _: True)),
      include_std_logic:
      {
        type_custom_match(MCSTDLOGICV, MCSTDLOGICV, MCSTDLOGICV):  SymbolOperator("+", arity = 2, force_folding = True),
        type_custom_match(MCSTDLOGICV, MCSTDLOGICV, FSM(ML_StdLogic)):  SymbolOperator("+", arity = 2, force_folding = True),
        type_custom_match(MCSTDLOGICV, FSM(ML_StdLogic), MCSTDLOGICV):  SymbolOperator("+", arity = 2, force_folding = True),
      },
      # fallback
      lambda _: True: {
        type_custom_match(MCFixedPoint, MCFixedPoint, MCFixedPoint): ComplexOperator(optree_modifier = fixed_point_add_modifier),
      }
    }
  },
  Subtraction: {
    None: {
      exclude_std_logic: 
          build_simplified_operator_generation_nomap([v8int32, v8uint32, ML_Int16, ML_UInt16, ML_Int32, ML_UInt32, ML_Int64, ML_UInt64, ML_Int128,ML_UInt128], 2, SymbolOperator("-", arity = 2, force_folding = True), cond = (lambda _: True)),
      include_std_logic:
      {
        type_custom_match(MCSTDLOGICV, MCSTDLOGICV, MCSTDLOGICV):  SymbolOperator("-", arity = 2, force_folding = True),
      },
      # fallback
      lambda _: True: {
        type_custom_match(MCFixedPoint, MCFixedPoint, MCFixedPoint): ComplexOperator(optree_modifier = fixed_point_sub_modifier),
      }
    }
  },
  Multiplication: {
    None: {
      lambda optree: True: {
        type_custom_match(MCSTDLOGICV, MCSTDLOGICV, MCSTDLOGICV): SymbolOperator("*", arity = 2, force_folding = True),
        type_custom_match(MCFixedPoint, MCFixedPoint, MCFixedPoint): ComplexOperator(optree_modifier = fixed_point_mul_modifier),
      },
    },
  },
  BitLogicNegate: {
    None: {
      lambda optree: True: {
        type_strict_match(ML_StdLogic, ML_StdLogic): FunctionOperator("not", arity=1),
        type_custom_match(MCSTDLOGICV, MCSTDLOGICV): FunctionOperator("not", arity=1), 
      },
    },
  },
  Negation: {
    None: {
      lambda optree: True: {
        type_custom_match(MCSTDLOGICV, MCSTDLOGICV): ComplexOperator(optree_modifier = negation_modifer), 
      },
    },
  },
  LogicalAnd: {
    None: {
      lambda _: True : {
        type_strict_match(ML_Bool, ML_Bool, ML_Bool): SymbolOperator("and", arity = 2, force_folding = False),
      },
   }, 
  },
  LogicalOr: {
    None: {
      lambda _: True : {
        type_strict_match(ML_Bool, ML_Bool, ML_Bool): SymbolOperator("or", arity = 2, force_folding = False),
      },
   }, 
  },
  Event: {
    None: {
      lambda _: True : {
        type_strict_match(ML_Bool, ML_StdLogic): SymbolOperator("\'event", lspace = "", inverse = True, arity = 1, force_folding = False), 
      },
    },
  },
  Comparison: 
      dict (
        [(specifier,
          { 
              lambda _: True: {
                  type_custom_match(FSM(ML_Bool), FSM(ML_Binary64), FSM(ML_Binary64)): 
                    SymbolOperator(vhdl_comp_symbol[specifier], arity = 2, force_folding = False),
                  type_custom_match(FSM(ML_Bool), FSM(ML_Binary32), FSM(ML_Binary32)): 
                    SymbolOperator(vhdl_comp_symbol[specifier], arity = 2, force_folding = False),
                  type_custom_match(FSM(ML_Bool), FSM(ML_Binary16), FSM(ML_Binary16)): 
                    SymbolOperator(vhdl_comp_symbol[specifier], arity = 2, force_folding = False),
                  type_custom_match(FSM(ML_Bool), TCM(ML_StdLogicVectorFormat), TCM(ML_StdLogicVectorFormat)): 
                    SymbolOperator(vhdl_comp_symbol[specifier], arity = 2, force_folding = False),
                  type_strict_match(ML_Bool, ML_StdLogic, ML_StdLogic):
                    SymbolOperator(vhdl_comp_symbol[specifier], arity = 2, force_folding = False),
              },
              #build_simplified_operator_generation([ML_Int32, ML_Int64, ML_UInt64, ML_UInt32, ML_Binary32, ML_Binary64], 2, SymbolOperator(">=", arity = 2), result_precision = ML_Int32),
          }) for specifier in [Comparison.Equal, Comparison.NotEqual, Comparison.Greater, Comparison.GreaterOrEqual, Comparison.Less, Comparison.LessOrEqual]] 
          + 
          [(specifier, 
            { 
                lambda _: True: {
                    type_custom_match(FSM(ML_Bool), TCM(ML_StdLogicVectorFormat), TCM(ML_StdLogicVectorFormat)): 
                      TemplateOperator("signed(%%s) %s signed(%%s)" % vhdl_comp_symbol[specifier], arity = 2, force_folding = False),
                },
            }) for specifier in [Comparison.GreaterSigned, Comparison.GreaterOrEqualSigned, Comparison.LessSigned, Comparison.LessOrEqualSigned] 
          ]
  ),
  ExponentExtraction: {
    None: {
      lambda _: True: {
        type_custom_match(TCM(ML_StdLogicVectorFormat), FSM(ML_Binary64)): SymbolOperator("(62 downto 52)", lspace = "", inverse = True, arity = 1, force_folding = True), 
        type_custom_match(TCM(ML_StdLogicVectorFormat), FSM(ML_Binary32)): SymbolOperator("(30 downto 23)", lspace = "", inverse = True, arity = 1, force_folding = True), 
        type_custom_match(TCM(ML_StdLogicVectorFormat), FSM(ML_Binary16)): SymbolOperator("(14 downto 10)", lspace = "", inverse = True, arity = 1, force_folding = True), 
      },
    },
  },
  ZeroExt: {
    None: {
      lambda _: True: {
        type_custom_match(TCM(ML_StdLogicVectorFormat), TCM(ML_StdLogicVectorFormat)): ComplexOperator(optree_modifier = zext_modifier), 
      },
    }
  },
  SignExt: {
    None: {
      lambda _: True: {
        type_custom_match(TCM(ML_StdLogicVectorFormat), TCM(ML_StdLogicVectorFormat)): ComplexOperator(optree_modifier = sext_modifier), 
      },
    }
  },
  Concatenation: {
    None: {
      lambda _: True: {
        type_custom_match(TCM(ML_StdLogicVectorFormat), TCM(ML_StdLogicVectorFormat), TCM(ML_StdLogicVectorFormat)): SymbolOperator("&", arity = 2, force_folding = True),
        type_custom_match(TCM(ML_StdLogicVectorFormat), FSM(ML_StdLogic), TCM(ML_StdLogicVectorFormat)): SymbolOperator("&", arity = 2, force_folding = True),
        type_custom_match(TCM(ML_StdLogicVectorFormat), TCM(ML_StdLogicVectorFormat), FSM(ML_StdLogic)): SymbolOperator("&", arity = 2, force_folding = True),
      },
    },
  },
  VectorElementSelection: {
    None: {
        # make sure index accessor is a Constant (or fallback to C implementation)
       lambda optree: True:  {
        type_custom_match(FSM(ML_StdLogic), TCM(ML_StdLogicVectorFormat), type_all_match): TemplateOperator("%s(%s)", arity = 2),
      },
    },
  },
  Replication: {
    None: {
        # make sure index accessor is a Constant (or fallback to C implementation)
       lambda optree: True:  {
        type_custom_match(FSM(ML_StdLogic), FSM(ML_StdLogic)): IdentityOperator(),
        type_custom_match(TCM(ML_StdLogicVectorFormat), FSM(ML_StdLogic), FSM(ML_Integer)): TemplateOperatorFormat("({1!d} - 1 downto 0 => {0:s}"),
      },
    },
  },
  Conversion: {
    None: {
      lambda optree: True: {
        type_custom_match(TCM(ML_StdLogicVectorFormat), FSM(ML_Integer)): DynamicOperator(conversion_generator),
      }
    },
  },
  MantissaExtraction: {
    None: {
      lambda optree: True: {
        type_custom_match(MCSTDLOGICV, FSM(ML_Binary16)): ComplexOperator(optree_modifier = mantissa_extraction_modifier), # TemplateOperator("%s(22 downto 0)", arity = 1), 
        type_custom_match(MCSTDLOGICV, FSM(ML_Binary32)): ComplexOperator(optree_modifier = mantissa_extraction_modifier), # TemplateOperator("%s(22 downto 0)", arity = 1), 
        type_custom_match(MCSTDLOGICV, FSM(ML_Binary64)): ComplexOperator(optree_modifier = mantissa_extraction_modifier), # TemplateOperator("%s(22 downto 0)", arity = 1), 
      },
    },
  },
  CopySign: {
    None: {
      lambda optree: True: {
        type_custom_match(FSM(ML_StdLogic), ML_Binary16): TemplateOperator("%s(15)", arity = 1),
        type_custom_match(FSM(ML_StdLogic), ML_Binary32): TemplateOperator("%s(31)", arity = 1),
        type_custom_match(FSM(ML_StdLogic), ML_Binary64): TemplateOperator("%s(63)", arity = 1),
        type_custom_match(FSM(ML_StdLogic), MCSTDLOGICV): DynamicOperator(copy_sign_generator),
      },
    },
  },
  BitLogicXor: {
    None: {
      lambda optree: True: {
        type_strict_match(ML_StdLogic, ML_StdLogic, ML_StdLogic): SymbolOperator("xor", arity = 2),
        type_custom_match(TCM(ML_StdLogicVectorFormat), TCM(ML_StdLogicVectorFormat), TCM(ML_StdLogicVectorFormat)): SymbolOperator("xor", arity = 2),
      },
    },
  },
  BitLogicAnd: {
    None: {
      lambda optree: True: {
        type_strict_match(ML_StdLogic, ML_StdLogic, ML_StdLogic): SymbolOperator("and", arity = 2),
        type_custom_match(TCM(ML_StdLogicVectorFormat), TCM(ML_StdLogicVectorFormat), TCM(ML_StdLogicVectorFormat)): SymbolOperator("and", arity = 2),
      },
    },
  },
  BitLogicOr: {
    None: {
      lambda optree: True: {
        type_strict_match(ML_StdLogic, ML_StdLogic, ML_StdLogic): SymbolOperator("or", arity = 2),
        type_custom_match(TCM(ML_StdLogicVectorFormat), TCM(ML_StdLogicVectorFormat), TCM(ML_StdLogicVectorFormat)): SymbolOperator("or", arity = 2),
      },
    },
  },
  Truncate: {
    None: {
      lambda optree: True: {
        type_custom_match(TCM(ML_StdLogicVectorFormat), TCM(ML_StdLogicVectorFormat)): DynamicOperator(truncate_generator),
      },
    },
  },
  SignCast: {
    SignCast.Signed: {
      lambda optree: True: {
        type_custom_match(MCSTDLOGICV, MCSTDLOGICV): FunctionOperator("signed", arity = 1, force_folding = False),
      },
    },
    SignCast.Unsigned: {
      lambda optree: True: {
        type_custom_match(MCSTDLOGICV, MCSTDLOGICV): FunctionOperator("unsigned", arity = 1, force_folding = False),
      },
    },
  },
  TypeCast: {
    None: {
      lambda optree: True: {
        type_custom_match(FSM(ML_Binary16), TCM(ML_StdLogicVectorFormat)): IdentityOperator(output_precision = ML_Binary16, no_parenthesis = True),
        type_custom_match(FSM(ML_Binary16), FSM(ML_Binary16)): IdentityOperator(output_precision = ML_Binary16, no_parenthesis = True),
        type_custom_match(MCSTDLOGICV, FSM(ML_Binary16)): IdentityOperator(no_parenthesis = True),

        type_custom_match(FSM(ML_Binary32), TCM(ML_StdLogicVectorFormat)): IdentityOperator(output_precision = ML_Binary32, no_parenthesis = True),
        type_custom_match(FSM(ML_Binary32), FSM(ML_Binary32)): IdentityOperator(output_precision = ML_Binary32, no_parenthesis = True),
        type_custom_match(MCSTDLOGICV, FSM(ML_Binary32)): IdentityOperator(no_parenthesis = True),

        type_custom_match(FSM(ML_Binary64), TCM(ML_StdLogicVectorFormat)): IdentityOperator(output_precision = ML_Binary64, no_parenthesis = True),
        type_custom_match(FSM(ML_Binary64), FSM(ML_Binary64)): IdentityOperator(output_precision = ML_Binary64, no_parenthesis = True),
        type_custom_match(MCSTDLOGICV, FSM(ML_Binary64)): IdentityOperator(no_parenthesis = True),

        type_custom_match(MCSTDLOGICV, MCFixedPoint): IdentityOperator(no_parenthesis = True),
        type_custom_match(MCFixedPoint, MCSTDLOGICV): IdentityOperator(no_parenthesis = True),
      },
    },
  },
  BitLogicRightShift: {
    None: {
      lambda optree: True: {
        type_custom_match(MCSTDLOGICV, MCSTDLOGICV, MCSTDLOGICV): TemplateOperator("std_logic_vector(shift_right(unsigned(%s), to_integer(unsigned(%s))))", arity = 2, force_folding = True),
      },
    },
  },
  BitLogicLeftShift: {
    None: {
      lambda optree: True: {
        type_custom_match(MCSTDLOGICV, MCSTDLOGICV, MCSTDLOGICV): TemplateOperator("std_logic_vector(shift_left(unsigned(%s), to_integer(unsigned(%s))))", arity = 2, force_folding = True),
      },
    },
  },
  CountLeadingZeros: {
    None: {
      lambda optree: True: {
        type_custom_match(MCSTDLOGICV, MCSTDLOGICV): FunctionOperator("count_leading_zeros", arity = 1),
      },
    },
  },
  SpecificOperation: {
    SpecificOperation.CopySign: {
      lambda optree: True: {
        type_custom_match(FSM(ML_StdLogic), FSM(ML_Binary16)): TemplateOperator("%s(15)", arity = 1),
        type_custom_match(FSM(ML_StdLogic), FSM(ML_Binary32)): TemplateOperator("%s(31)", arity = 1),
        type_custom_match(FSM(ML_StdLogic), FSM(ML_Binary64)): TemplateOperator("%s(63)", arity = 1),
        type_custom_match(FSM(ML_StdLogic), MCSTDLOGICV): DynamicOperator(copy_sign_generator),
      },
    },
  },
  SubSignalSelection: {
    None: {
      lambda optree: True: {
        type_custom_match(MCSTDLOGICV, MCSTDLOGICV): DynamicOperator(sub_signal_generator),
      },
    },
  },
}

class FormalBackend(AbstractBackend):
  """ description of VHDL's Backend """
  target_name = "formal_backend"
  TargetRegister.register_new_target(target_name, lambda _: FormalBackend)

  code_generation_table = {
    VHDL_Code: formal_generation_table,
    C_Code: formal_generation_table,
    Gappa_Code: {}
  }

  def __init__(self):
    AbstractBackend.__init__(self)
    print "initializing Formal target"
 

class VHDLBackend(FormalBackend):
  """ description of VHDL's Backend """
  target_name = "vhdl_backend"
  TargetRegister.register_new_target(target_name, lambda _: VHDLBackend)


  code_generation_table = {
    VHDL_Code: vhdl_code_generation_table,
    Gappa_Code: {}
  }

  def __init__(self):
    AbstractBackend.__init__(self)
    print "initializing VHDL target"
      
