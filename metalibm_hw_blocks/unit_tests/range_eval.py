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
""" Node range evaluation unit test """

import sollya

from sollya import parse as sollya_parse
from sollya import Interval, inf, sup

from metalibm_core.core.ml_operations import (
    Comparison, Addition, Select, Constant, Conversion,
    Min, Max, Ceil, Floor, Trunc
)
from metalibm_core.code_generation.code_constant import VHDL_Code
from metalibm_core.core.ml_formats import (
    ML_Int32
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

from metalibm_core.opt.opt_utils import evaluate_range


from metalibm_core.utility.rtl_debug_utils import (
    debug_std, debug_dec, debug_fixed
)

def interval_max(a, b):
    """ determine the range of max(x, y) for x in a, y in b """
    return Interval(
        max(inf(a), inf(b)),
        max(sup(a), sup(b))
    )
def interval_min(a, b):
    """ determine the range of min(x, y) for x in a, y in b """
    return Interval(
        min(inf(a), inf(b)),
        min(sup(a), sup(b))
    )
def interval_union(a, b):
    """ Returns the over union of interval a and interval b """
    return Interval(min(inf(a), inf(b)), max(sup(a), sup(b)))


class RangeEvalEntity(ML_Entity("ml_range_eval_entity"), TestRunner):
    """ Range Eval Entity unit-test """
    @staticmethod
    def get_default_args(width=32, **kw):
        """ generate default argument template """
        return DefaultEntityArgTemplate(
            precision=ML_Int32,
            debug_flag=False,
            target=VHDLBackend(),
            output_file="my_range_eval_entity.vhd",
            entity_name="my_range_eval_entity",
            language=VHDL_Code,
            width=width,
            # passes=[("beforecodegen:size_datapath")],
        )

    def __init__(self, arg_template=None):
        """ Initialize """
        # building default arg_template if necessary
        arg_template = RangeEvalEntity.get_default_args() if \
            arg_template is None else arg_template
        # initializing I/O precision
        precision = arg_template.precision
        io_precisions = [precision] * 2
        self.width = 17

        # initializing base class
        ML_EntityBasis.__init__(self,
                                base_name="adaptative_design",
                                arg_template=arg_template
                                )

        self.accuracy = arg_template.accuracy
        self.precision = arg_template.precision

    def generate_scheme(self):
        """ main scheme generation """

        int_size = 3
        frac_size = self.width - int_size

        input_precision = fixed_point(int_size, frac_size)
        output_precision = fixed_point(int_size, frac_size)

        expected_interval = {}

        # declaring main input variable
        var_x = self.implementation.add_input_signal("x", input_precision)
        x_interval = Interval(-10.3,10.7)
        var_x.set_interval(x_interval)
        expected_interval[var_x] = x_interval

        var_y = self.implementation.add_input_signal("y", input_precision)
        y_interval = Interval(-17.9,17.2)
        var_y.set_interval(y_interval)
        expected_interval[var_y] = y_interval

        var_z = self.implementation.add_input_signal("z", input_precision)
        z_interval = Interval(-7.3,7.7)
        var_z.set_interval(z_interval)
        expected_interval[var_z] = z_interval

        cst = Constant(42.5, tag = "cst")
        expected_interval[cst] = Interval(42.5)

        conv_ceil = Ceil(var_x, tag = "ceil")
        expected_interval[conv_ceil] = sollya.ceil(x_interval)

        conv_floor = Floor(var_y, tag = "floor")
        expected_interval[conv_floor] = sollya.floor(y_interval)

        mult = var_z * var_x
        mult.set_tag("mult")
        mult_interval = z_interval * x_interval
        expected_interval[mult] = mult_interval

        large_add = (var_x + var_y) - mult
        large_add.set_attributes(tag = "large_add")
        large_add_interval = (x_interval + y_interval) - mult_interval
        expected_interval[large_add] = large_add_interval


        reduced_result = Max(0, Min(large_add, 13))
        reduced_result.set_tag("reduced_result")
        reduced_result_interval = interval_max(
            Interval(0),
            interval_min(
                large_add_interval,
                Interval(13)
            )
        )
        expected_interval[reduced_result] = reduced_result_interval

        select_result = Select(
            var_x > var_y,
            reduced_result,
            var_z,
            tag = "select_result"
        )
        select_interval = interval_union(reduced_result_interval, z_interval)
        expected_interval[select_result] = select_interval


        # checking interval evaluation
        for var in [cst, var_x, var_y, mult, large_add, reduced_result, select_result, conv_ceil, conv_floor]:
            interval = evaluate_range(var)
            expected = expected_interval[var]
            print("{}: {} vs expected {}".format(var.get_tag(), interval, expected))
            assert not interval is None
            assert interval == expected


        return [self.implementation]

    standard_test_cases = [
        ({"x": 2, "y": 2}, None),
        ({"x": 1, "y": 2}, None),
        ({"x": 0.5, "y": 2}, None),
        ({"x": -1, "y": -1}, None),
    ]

    def numeric_emulate(self, io_map):
        """ Meta-Function numeric emulation """
        raise NotImplementedError


    @staticmethod
    def __call__(args):
        # just ignore args here and trust default constructor?
        # seems like a bad idea.
        ut_range_eval_entity = RangeEvalEntity(args)
        ut_range_eval_entity.gen_implementation()

        return True

run_test = RangeEvalEntity


if __name__ == "__main__":
        # auto-test
    main_arg_template = ML_EntityArgTemplate(
        default_entity_name="new_range_eval",
        default_output_file="mt_eval.vhd",
        default_arg=RangeEvalEntity.get_default_args()
    )

    # argument extraction
    args = parse_arg_index_list = main_arg_template.arg_extraction()

    ml_range_eval = RangeEvalEntity(args)

    ml_range_eval.gen_implementation()
