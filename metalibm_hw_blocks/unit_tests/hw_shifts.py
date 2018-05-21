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
# created:          May 17th, 2018
# last-modified:    May 17th, 2018
#
# Author(s): Nicolas Brunie <nbrunie@kalray.eu>
###############################################################################
""" Sub-component instantiation unit test """

import sollya

from sollya import parse as sollya_parse

from metalibm_core.core.ml_operations import (
    Comparison, Addition, Select, Constant, Conversion,
    Min, Max,
    Statement,
    BitLogicLeftShift, BitLogicRightShift
)
from metalibm_core.core.ml_hdl_operations import (
    Signal, PlaceHolder
)
from metalibm_core.core.ml_hdl_format import (
    ML_StdLogicVectorFormat
)
from metalibm_core.code_generation.code_constant import VHDL_Code
from metalibm_core.core.ml_formats import (
    ML_Int32, ML_Integer
)
from metalibm_core.code_generation.vhdl_backend import VHDLBackend
from metalibm_core.core.ml_entity import (
    ML_Entity, ML_EntityBasis, DefaultEntityArgTemplate
)
from metalibm_core.utility.ml_template import \
    ML_EntityArgTemplate
from metalibm_core.utility.log_report import Log
from metalibm_core.core.ml_hdl_format import fixed_point

from metalibm_functions.unit_tests.utils import TestRunner

from metalibm_core.opt.p_check_precision import Pass_CheckGeneric
from metalibm_core.core.passes import PassScheduler

from metalibm_hw_blocks.lzc import ML_LeadingZeroCounter

from metalibm_core.utility.rtl_debug_utils import (
    debug_std, debug_dec, debug_fixed
)


class UnitShiftTest(ML_EntityBasis, TestRunner):
    entity_name = "ut_hw_shifts"
    """ Adaptative Entity unit-test """
    @staticmethod
    def get_default_args(width=32, **kw):
        """ generate default argument template """
        return DefaultEntityArgTemplate(
            precision=ML_Int32,
            debug_flag=False,
            target=VHDLBackend(),
            output_file="ut_hw_shifts.vhd",
            entity_name="ut_hw_shifts",
            language=VHDL_Code,
            width=width,
            passes=[
                ("beforepipelining:size_datapath"),
                ("beforepipelining:rtl_legalize"),
                ("beforepipelining:unify_pipeline_stages"),
                ("beforepipelining:dump"),
                ],
        )

    def __init__(self, arg_template=None):
        """ Initialize """
        # building default arg_template if necessary
        arg_template = UnitShiftTest.get_default_args() if \
            arg_template is None else arg_template
        # initializing I/O precision
        self.width = arg_template.width
        precision = arg_template.precision
        io_precisions = [precision] * 2
        Log.report(
            Log.Info,
            "generating Adaptative Entity with width={}".format(self.width)
        )

        # initializing base class
        ML_EntityBasis.__init__(self,
                                base_name="adaptative_design",
                                arg_template=arg_template
                                )

        self.accuracy = arg_template.accuracy
        self.precision = arg_template.precision

        int_size = 3
        frac_size = 7

        self.input_precision = fixed_point(int_size, frac_size)
        self.output_precision = fixed_point(int_size, frac_size)


    def generate_scheme(self):
        """ main scheme generation """
        Log.report(Log.Info, "input_precision is {}".format(self.input_precision))
        Log.report(Log.Info, "output_precision is {}".format(self.output_precision))

        shift_amount_precision = fixed_point(3, 0, signed=False)

        # declaring main input variable
        var_x = self.implementation.add_input_signal("x", self.input_precision)
        var_y = self.implementation.add_input_signal("s", shift_amount_precision)

        cst_8 = Constant(9, precision=ML_Integer)
        cst_7 = Constant(7, precision=ML_Integer)

        cst_right_shifted_x = BitLogicRightShift(var_x, cst_8)
        cst_left_shifted_x = BitLogicLeftShift(var_x, cst_7)

        dyn_right_shifted_x = BitLogicRightShift(var_x, var_y)
        dyn_left_shifted_x = BitLogicLeftShift(var_x, var_y)

        result = cst_right_shifted_x + cst_left_shifted_x + dyn_right_shifted_x + dyn_left_shifted_x

        # output
        self.implementation.add_output_signal("vr_out", result)

        return [self.implementation]

    standard_test_cases = [
    ]

    def numeric_emulate(self, io_map):
        """ Meta-Function numeric emulation """
        vx = io_map["x"]
        result = {"vr_out": vx}
        return result


    @staticmethod
    def __call__(args):
        # just ignore args here and trust default constructor?
        # seems like a bad idea.
        ut_adaptative_entity = UnitShiftTest(args)
        ut_adaptative_entity.gen_implementation()
        return True

run_test = UnitShiftTest


if __name__ == "__main__":
        # auto-test
    main_arg_template = ML_EntityArgTemplate(
        default_entity_name="ut_hw_shifts",
        default_output_file="ut_hw_shifts.vhd",
        default_arg=UnitShiftTest.get_default_args()
    )
    main_arg_template.parser.add_argument(
        "--width", dest="width", type=int, default=32,
        help="set input width value (in bits)"
    )
    # argument extraction
    args = parse_arg_index_list = main_arg_template.arg_extraction()

    ut_hw_shifts = UnitShiftTest(args)

    ut_hw_shifts.gen_implementation()
