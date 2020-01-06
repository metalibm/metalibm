# -*- coding: utf-8 -*-
# This file is part of metalibm (https://github.com/kalray/metalibm)

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

import argparse
import sys
import re

from sollya import Interval

# import meta-function script from metalibm_functions directory
import metalibm_hw_blocks.lzc
import metalibm_hw_blocks.lza
import metalibm_hw_blocks.ml_fp_adder
import metalibm_hw_blocks.ml_fp_mpfma
import metalibm_hw_blocks.ml_fixed_mpfma
import metalibm_hw_blocks.ml_fp_div
import metalibm_hw_blocks.bipartite_approx
import metalibm_hw_blocks.mult_array as mult_array
import metalibm_hw_blocks.compound_adder as compound_adder
from metalibm_hw_blocks.mult_array import multiplication_descriptor_parser as mult_array_parser

from metalibm_core.core.ml_formats import  \
  ML_Binary16, ML_Binary32, ML_Binary64, ML_Int32

from metalibm_core.utility.ml_template import (
    target_instanciate, DefaultEntityArgTemplate
)
from metalibm_core.core.ml_hdl_format import HdlVirtualFormat

from valid.test_utils import *

class EntitySchemeTest(NewSchemeTest):
  ## Build an argument template from dict
  def build_arg_template(self, **kw):
    default_arg = self.ctor.get_default_args(**kw)
    return default_arg

# list of non-regression tests
# details on NewSchemeTest object can be found in valid.test_utils module
#   Each object requires a title, a function constructor and a list
#   of test cases (each is a dictionnary of parameters -> values)
new_scheme_function_list = [
  EntitySchemeTest(
    "basic Leading Zero Count",
    metalibm_hw_blocks.lzc.ML_LeadingZeroCounter,
    [
        {"width": 32, "simulator": "ghdl", "execute_trigger": True, "auto_test": 10},
        {"width": 13, "simulator": "ghdl", "execute_trigger": True, "auto_test": 10},]
  ),
  EntitySchemeTest(
    "basic floating-point adder",
    metalibm_hw_blocks.ml_fp_adder.FP_Adder,
    [{"precision": ML_Binary32}, {"precision": ML_Binary64},]
  ),
  EntitySchemeTest(
    "mixed-precision fused multiply-add",
    metalibm_hw_blocks.ml_fp_mpfma.FP_MPFMA,
    [
    {},
    {"precision": HdlVirtualFormat(ML_Binary16), "acc_precision": HdlVirtualFormat(ML_Binary32)}],
  ),
  EntitySchemeTest(
    "fixed-point accumulation MPFMA",
    metalibm_hw_blocks.ml_fixed_mpfma.FP_FIXED_MPFMA,
    [
      {},
      {"precision": HdlVirtualFormat(ML_Binary16), "extra_digits": 16},
      {"precision": HdlVirtualFormat(ML_Binary16), "extra_digits": 16, "sign_magnitude": True},
      {"precision": HdlVirtualFormat(ML_Binary16), "extra_digits": 16, "pipelined": True},
    ],
  ),
  EntitySchemeTest(
    "floating-point division",
    metalibm_hw_blocks.ml_fp_div.FP_Divider,
    [
        {"build_enable": True, "simulator": "ghdl"},
    ]
  ),
  EntitySchemeTest(
    "bipartite approximation operator",
    metalibm_hw_blocks.bipartite_approx.BipartiteApprox,
    [
        {"build_enable": True, "simulator": "ghdl"},
    ]
  ),
  EntitySchemeTest(
    "basic Leading Zero Anticipator",
    metalibm_hw_blocks.lza.ML_LeadingZeroAnticipator,
    [
        {"width": 32, "simulator": "ghdl", "execute_trigger": True, "auto_test": 10},
        {"width": 13, "simulator": "ghdl", "execute_trigger": True, "auto_test": 10},
        {"width": 32, "signed": True, "simulator": "ghdl", "execute_trigger": True, "auto_test": 10},
        {"width": 13, "signed": True, "simulator": "ghdl", "execute_trigger": True, "auto_test": 10},
    ]
  ),
  EntitySchemeTest(
    "multiplication array",
    mult_array.MultArray,
    [
        {"op_expr": mult_array_parser("FS9.0xFS9.0+FU13.0"), "build_enable": True, "simulator": "ghdl"},
        {"dummy_mode": True, "method": mult_array.ReductionMethod.Wallace, "build_enable": True, "simulator": "ghdl"},
        {"booth_mode": True, "method": mult_array.ReductionMethod.Dadda_4to2,
         "op_expr": mult_array_parser("FS9.4xFS9.-2+FU13.3") , "build_enable": True, "simulator": "ghdl"},
        {"dummy_mode": True, "method": mult_array.ReductionMethod.Wallace_4to2,
         "op_expr": mult_array_parser("FS9.4xFS9.-2+FU13.3xFS3.3") , "build_enable": True, "simulator": "ghdl"},
        {"booth_mode": True, "method": mult_array.ReductionMethod.Dadda, "build_enable": True, "simulator": "ghdl"},
        # covering tag option
        {"dummy_mode": True, "method": mult_array.ReductionMethod.Wallace_4to2,
         "pipelined": True,
         "op_expr": mult_array_parser("FS9.4xFS9.-2[0,mytag]+FU13.3[1,other_tag]") , "build_enable": True, "simulator": "ghdl"},
    ]
  ),
  EntitySchemeTest(
    "compound adder",
    compound_adder.CompoundAdder,
    [
        {"auto_test": 10, "execute_trigger": True, "simulator": "ghdl"},
    ]
  ),
]

test_tag_map = {}
for test in new_scheme_function_list:
  test_tag_map[test.get_tag_title()] = test

## Command line action to set break on error in load module
class ListTestAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        for test in  new_scheme_function_list:
          print(test.get_tag_title())
        exit(0)

# generate list of test object from string
# of comma separated test's tag
def parse_test_list(test_list):
  test_tags = test_list.split(",")
  return [test_tag_map[tag] for tag in test_tags]

arg_parser = argparse.ArgumentParser(" Metalibm non-regression tests")
# enable debug mode
arg_parser.add_argument("--debug", dest = "debug", action = "store_const", 
                        default = False, const = True, 
                        help = "enable debug mode")
# listing available tests
arg_parser.add_argument("--list", action = ListTestAction, help = "list available test", nargs = 0) 

# select list of tests to be executed
arg_parser.add_argument("--execute", dest = "test_list", type = parse_test_list, default = new_scheme_function_list, help = "list of comma separated test to be executed") 

arg_parser.add_argument("--match", dest = "match_regex", type = str, default = ".*", help = "list of comma separated match regexp to be used for test selection") 




args = arg_parser.parse_args(sys.argv[1:])

success = True
# list of TestResult objects generated by execution
# of new scheme tests
result_details = []

for test_scheme in args.test_list:
  if re.search(args.match_regex, test_scheme.get_tag_title()) != None:
    test_result = test_scheme.perform_all_test(debug = args.debug)
    result_details.append(test_result)
    if not test_result.get_result(): 
      success = False

# Printing test summary for new scheme
for result in result_details:
  print(result.get_details())

if success:
  print("OVERALL SUCCESS")
  exit(0)
else:
  print("OVERALL FAILURE")
  exit(1)
