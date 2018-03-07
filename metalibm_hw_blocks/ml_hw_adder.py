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
# last-modified:    Mar  7th, 2018
# Author(s): Nicolas Brunie <nbrunie@kalray.eu>
###############################################################################
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

    self.implementation.start_new_stage()

    vr_out = Select(
      Comparison(vx, Constant(1, precision = precision), precision = ML_Bool, specifier = Comparison.Equal),
      vr_add,
      Select(
        Comparison(vx, Constant(1, precision = precision), precision = ML_Bool, specifier = Comparison.LessOrEqual),
        vr_sub,
        vx,
        precision = precision
      ),
      precision = precision,
      tag = "vr_res"
    )

    #for sig in [vx, vy, vr_add, vr_sub, vr_out]:
    #  print "%s, stage=%d" % (sig.get_tag(), sig.attributes.init_stage)

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
