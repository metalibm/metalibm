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


from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_operations import *
from metalibm_core.code_generation.vhdl_backend import VHDLBackend
from metalibm_core.core.ml_entity import (
    ML_Entity, ML_EntityBasis, DefaultEntityArgTemplate
)

from metalibm_core.utility.ml_template import ML_EntityArgTemplate


from metalibm_core.core.ml_hdl_format import ML_StdLogicVectorFormat
from metalibm_core.core.ml_hdl_operations import *

from metalibm_functions.unit_tests.utils import TestRunner


class UT_RTL_Report(ML_Entity("ut_rtl_report"), TestRunner):
  @staticmethod
  def get_default_args(**kw):
    default_dict = {
         "precision": ML_Int32,
         "debug_flag": False,
         "target": VHDLBackend(),
         "output_file": "ut_rtl_report.vhd",
         "entity_name": "ut_rtl_report",
         "language": VHDL_Code,
    }
    default_dict.update(kw)
    return DefaultEntityArgTemplate(**default_dict)

  def __init__(self, arg_template = None):
    # initializing I/O precision
    precision = arg_template.precision
    io_precisions = [precision] * 2

    # initializing base class
    ML_EntityBasis.__init__(self, 
      base_name = "ut_rtl_report",
      arg_template = arg_template
    )

    self.precision = arg_template.precision

  def generate_scheme(self):
    """ generate main architecture for UT_RTL_Report """

    main = Statement()
    # basic string
    main.add(Report("displaying simple string"))
    # string from std_logic_vector conversion
    cst_format = ML_StdLogicVectorFormat(12)
    cst = Constant(17, precision = cst_format)
    main.add(Report(Conversion(cst, precision = ML_String)))
    # string from concatenation of several elements
    complex_string = Concatenation(
        "displaying concatenated string",
        Conversion(cst, precision = ML_String),
        precision = ML_String
    )
    main.add(Report(complex_string))

    main.add(Wait(100))

    # main process
    main_process = Process(main)
    self.implementation.add_process(main_process)

    return [self.implementation]

  @staticmethod
  def __call__(args):
    # just ignore args here and trust default constructor?
    # seems like a bad idea.
    ut_rtl_report = UT_RTL_Report(args)
    ut_rtl_report.gen_implementation()

    return True

run_test = UT_RTL_Report

if __name__ == "__main__":
    # auto-test
    arg_template = ML_EntityArgTemplate(
        default_entity_name = "ut_rtl_report",
        default_output_file = "ut_rtl_report.vhd",
        default_arg = UT_RTL_Report.get_default_args()
    )

    # argument extraction
    args = parse_arg_index_list = arg_template.arg_extraction()

    ut_rtl_report = UT_RTL_Report(args)

    ut_rtl_report.gen_implementation()
