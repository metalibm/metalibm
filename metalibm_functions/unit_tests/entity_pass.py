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

from sollya import Interval, ceil, floor, round, inf, sup, log, exp, expm1, log2, guessdegree, dirtyinfnorm, RN, RD
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

from metalibm_core.core.passes import *


from metalibm_core.core.ml_hdl_format import *
from metalibm_core.core.ml_hdl_operations import *

from metalibm_functions.unit_tests.utils import TestRunner

# global list to check pass execution
executed_id_list = []

class LocalPass(OptimizationPass):
  def __init__(self, descriptor, final_id):
    OptimizationPass.__init__(self, descriptor)
    self.final_id = final_id

  def execute(self, optree):
    Log.report(Log.Verbose, "executing pass {}".format(self.descriptor))
    executed_id_list.append(self.final_id)
    return optree

class ML_UT_EntityPass(ML_Entity("ml_lzc"), TestRunner):
  @staticmethod
  def get_default_args(width = 32):
    return DefaultEntityArgTemplate(
             precision = ML_Int32,
             debug_flag = False,
             target = VHDLBackend(),
             output_file = "my_lzc.vhd",
             entity_name = "my_lzc",
             language = VHDL_Code,
             width = width,
           )

  def __init__(self, arg_template = None):
    # building default arg_template if necessary
    arg_template = ML_UT_EntityPass.get_default_args() if arg_template is None else arg_template
    # initializing I/O precision
    self.width = arg_template.width
    precision = arg_template.precision
    io_precisions = [precision] * 2
    Log.report(Log.Info, "generating LZC with width={}".format(self.width))

    # initializing base class
    ML_EntityBasis.__init__(self,
      base_name = "ml_lzc",
      arg_template = arg_template
    )

    pass_scheduler = self.get_pass_scheduler()
    pass_5 = LocalPass("pass 5", 5)
    pass_3 = LocalPass("pass 3", 3)
    pass_4 = LocalPass("pass 4", 4)
    pass_1 = LocalPass("pass 1", 1)
    pass_2 = LocalPass("pass 2", 2)
    pass_3_deps = CombineAnd(
      AfterPassById(pass_5.get_pass_id()),
      CombineAnd(
        AfterPassById(pass_2.get_pass_id()),
        AfterPassByClass(LocalPass)
      )
    )
    pass_4_deps = CombineAnd(
      AfterPassById(pass_3.get_pass_id()),
      pass_3_deps
    )
    pass_5_deps = CombineOr(
      AfterPassById(pass_3.get_pass_id()),
      AfterPassById(pass_2.get_pass_id())
    )
    # registerting pass in arbitrary order
    pass_scheduler.register_pass(
      pass_4,
      pass_dep = pass_4_deps,
      pass_slot = PassScheduler.JustBeforeCodeGen
    )
    pass_scheduler.register_pass(
      pass_5,
      pass_dep = pass_5_deps,
      pass_slot = PassScheduler.JustBeforeCodeGen
    )
    pass_scheduler.register_pass(
      pass_3,
      pass_dep = pass_3_deps,
      pass_slot = PassScheduler.JustBeforeCodeGen)
    pass_scheduler.register_pass(pass_1, pass_slot = PassScheduler.Start)
    pass_scheduler.register_pass(pass_2, pass_slot = PassScheduler.JustBeforeCodeGen)

    self.accuracy  = arg_template.accuracy
    self.precision = arg_template.precision

  def numeric_emulate(self, io_map):
    def count_leading_zero(v, w):
      tmp = v
      lzc = -1
      for i in range(w):
        if tmp & 2**(w - 1 - i):
          return i
      return w
    result = {}
    result["vr_out"] = count_leading_zero(io_map["x"], self.width)
    return result

  def generate_scheme(self):
    lzc_width = int(floor(log2(self.width))) + 1
    Log.report(Log.Info, "width of lzc out is {}".format(lzc_width))
    input_precision = ML_StdLogicVectorFormat(self.width)
    precision = ML_StdLogicVectorFormat(lzc_width)
    # declaring main input variable
    vx = self.implementation.add_input_signal("x", input_precision)
    vr_out = Signal("lzc", precision = precision, var_type = Variable.Local)
    iterator = Variable("i", precision = ML_Integer, var_type = Variable.Local)
    lzc_loop = RangeLoop(
      iterator,
      Interval(0, self.width - 1),
      ConditionBlock(
        Comparison(
          VectorElementSelection(vx, iterator, precision = ML_StdLogic),
          Constant(1, precision = ML_StdLogic),
          specifier = Comparison.Equal,
          precision = ML_Bool
        ),
        ReferenceAssign(
          vr_out,
          Conversion(
            Subtraction(
              Constant(self.width - 1, precision = ML_Integer),
              iterator,
              precision = ML_Integer
            ),
          precision = precision),
        )
      ),
      specifier = RangeLoop.Increasing,
    )
    lzc_process = Process(
      Statement(
        ReferenceAssign(vr_out, Constant(self.width, precision = precision)),
        lzc_loop,
      ),
      sensibility_list = [vx]
    )

    self.implementation.add_process(lzc_process)


    self.implementation.add_output_signal("vr_out", vr_out)

    return [self.implementation]

  @staticmethod
  def get_default_args(**kw):
    root_arg = {
      "entity_name" : "new_entity_pass",
      "output_file"   : "ut_entity_pass.c",
      "width"         : 32,
      "precision" : ML_Int32
    }
    root_arg.update(kw)
    return DefaultEntityArgTemplate(**root_arg)

  @staticmethod
  def __call__(args):
    # just ignore args here and trust default constructor? seems like a bad idea.
    ml_ut_block_lzcnt = ML_UT_EntityPass(args)
    ml_ut_block_lzcnt.gen_implementation()

    expected_id_list = [2, 5, 3, 4]

    Log.report(Log.Verbose, "expected_id_list: ", expected_id_list)
    assert reduce(
      lambda lhs, rhs: lhs and rhs,
      [exp == real for exp,real in zip(executed_id_list, expected_id_list)],
      True
    )

    return True

# registering top class for unit test
run_test = ML_UT_EntityPass

if __name__ == "__main__":
    # auto-test
    arg_template = ML_EntityArgTemplate(default_entity_name = "new_lzc", default_output_file = "ml_lzc.vhd", default_arg = ML_UT_EntityPass.get_default_args())
    arg_template.parser.add_argument("--width", dest = "width", type=int, default = 32, help = "set input width value (in bits)")
    # argument extraction
    args = parse_arg_index_list = arg_template.arg_extraction()

    ml_lzc           = ML_UT_EntityPass(args)

    ml_lzc.gen_implementation()

    expected_id_list = [2, 5, 3, 4]

    Log.report(Log.Verbose, "expected_id_list: ", expected_id_list)
    assert reduce(
      lambda lhs, rhs: lhs and rhs,
      [exp == real for exp,real in zip(executed_id_list, expected_id_list)],
      True
    )
