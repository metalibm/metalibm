# -*- coding: utf-8 -*-

import sys

import sollya

from sollya import S2, Interval, ceil, floor, round, inf, sup, log, exp, expm1, log2, guessdegree, dirtyinfnorm, RN, RD, cbrt
from sollya import parse as sollya_parse

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_table import ML_Table
from metalibm_core.code_generation.vhdl_backend import VHDLBackend
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_entity import ML_Entity, ML_EntityBasis, DefaultEntityArgTemplate
from metalibm_core.code_generation.generator_utility import FunctionOperator, FO_Result, FO_Arg


from metalibm_core.utility.ml_template import *
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import *
from metalibm_core.utility.num_utils   import ulp
from metalibm_core.utility.gappa_utils import is_gappa_installed


from metalibm_core.core.ml_hdl_format import *
from metalibm_core.core.ml_hdl_operations import *

class ML_HW_Adder(ML_Entity("ml_hw_adder")):
  def __init__(self, 
             arg_template = DefaultEntityArgTemplate, 
             precision = ML_Int32, 
             accuracy  = ML_Faithful,
             libm_compliant = True, 
             debug_flag = False, 
             fuse_fma = True, 
             fast_path_extract = True,
             target = VHDLBackend(), 
             output_file = "my_hw_adder.vhd", 
             entity_name = "my_hw_adder",
             language = VHDL_Code,
             vector_size = 1):
    # initializing I/O precision
    precision = ArgDefault.select_value([arg_template.precision, precision])
    io_precisions = [precision] * 2

    # initializing base class
    ML_EntityBasis.__init__(self, 
      base_name = "ml_hw_adder",
      entity_name = entity_name,
      output_file = output_file,

      io_precisions = io_precisions,
      abs_accuracy = None,

      backend = target,
      fuse_fma = fuse_fma,
      fast_path_extract = fast_path_extract,

      debug_flag = debug_flag,
      language = language,
      arg_template = arg_template
    )

    self.accuracy  = accuracy
    self.precision = precision

  def generate_scheme(self):
    precision = ML_StdLogicVectorFormat(32)
    # declaring main input variable
    vx = self.implementation.add_input_signal("x", precision) 
    vy = self.implementation.add_input_signal("y", precision) 

    clk = self.implementation.add_input_signal("clk", ML_StdLogic)
    reset = self.implementation.add_input_signal("reset", ML_StdLogic)

    self.implementation.start_new_stage()

    vr_add = Addition(vx, vy, tag = "vr", precision = precision)
    vr_sub = Subtraction(vx, vy, tag = "vr_sub", precision = precision)
    print "vr_add: ", vr_add.attributes.init_stage

    self.implementation.start_new_stage()

    vr_out = Select(
      Comparison(vx, Constant(1, precision = precision), precision = ML_Bool, specifier = Comparison.Equal),
      vr_add,
      Select(
        Comparison(vx, Constant(1, precision = precision), precision = ML_Bool, specifier = Comparison.LessOrEqual),
        vr_sub,
        vx
      ),
      precision = precision,
      tag = "vr_out"
    )

    for sig in [vx, vy, vr_add, vr_sub, vr_out]:
      print "%s, stage=%d" % (sig.get_tag(), sig.attributes.init_stage)
    #vr_d = Signal("vr_d", precision = vr.get_precision())

    #process_statement = Statement(
    #  ConditionBlock(LogicalAnd(Event(clk, precision = ML_Bool), Comparison(clk, Constant(1, precision = ML_StdLogic), specifier = Comparison.Equal, precision = ML_Bool), precision = ML_Bool), ReferenceAssign(vr_d, vr))
    #)
    #process = Process(process_statement, sensibility_list = [clk, reset])
    #self.implementation.add_process(process)

    #self.implementation.add_output_signal("r_d", vr_d)
    #self.implementation.add_output_signal("r", vr)
    self.implementation.add_output_signal("vr_out", vr_out)

    return [self.implementation]

  standard_test_cases =[sollya_parse(x) for x in  ["1.1", "1.5"]]


if __name__ == "__main__":
    # auto-test
    arg_template = ML_EntityArgTemplate(default_entity_name = "new_hw_adder", default_output_file = "ml_hw_adder.vhd" )
    # argument extraction 
    args = parse_arg_index_list = arg_template.arg_extraction()

    ml_hw_adder      = ML_HW_Adder(args)

    ml_hw_adder.gen_implementation()
