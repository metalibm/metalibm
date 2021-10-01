# -*- coding: utf-8 -*-

""" Adaptative fixed-point size unit test """
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

import sollya

from sollya import parse as sollya_parse

from metalibm_core.core.ml_operations import (
    Comparison, Addition, Select, Constant, Conversion
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


from metalibm_core.utility.rtl_debug_utils import (
    debug_std, debug_dec, debug_fixed
)


class AdaptativeEntity(ML_Entity("ml_adaptative_entity"), TestRunner):
    """ Adaptative Entity unit-test """
    @staticmethod
    def get_default_args(width=32, **kw):
        """ generate default argument template """
        return DefaultEntityArgTemplate(
            precision=ML_Int32,
            debug_flag=False,
            target=VHDLBackend(),
            output_file="my_adapative_entity.vhd",
            entity_name="my_adaptative_entity",
            base_name="adaptative_size",
            language=VHDL_Code,
            width=width,
            passes=[("beforecodegen:size_datapath")],
        )

    def __init__(self, arg_template=None):
        """ Initialize """
        # building default arg_template if necessary
        arg_template = AdaptativeEntity.get_default_args() if \
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
        ML_EntityBasis.__init__(self, arg_template=arg_template)

        self.accuracy = arg_template.accuracy
        self.precision = arg_template.precision

    def generate_scheme(self):
        """ main scheme generation """
        Log.report(Log.Info, "width parameter is {}".format(self.width))
        int_size = 3
        frac_size = self.width - int_size

        input_precision = fixed_point(int_size, frac_size)
        output_precision = fixed_point(int_size, frac_size)

        # declaring main input variable
        var_x = self.implementation.add_input_signal("x", input_precision)
        var_y = self.implementation.add_input_signal("y", input_precision)
        var_x.set_attributes(debug = debug_fixed)
        var_y.set_attributes(debug = debug_fixed)

        test = (var_x > 1)
        test.set_attributes(tag = "test", debug = debug_std)

        large_add = (var_x + var_y)

        pre_result = Select(
            test,
            1,
            large_add,
            tag = "pre_result",
            debug = debug_fixed
        )

        result = Conversion(pre_result, precision=output_precision)

        self.implementation.add_output_signal("vr_out", result)

        return [self.implementation]

    standard_test_cases = [
        ({"x": 2, "y": 2}, None),
        ({"x": 1, "y": 2}, None),
        ({"x": 0.5, "y": 2}, None),
        ({"x": -1, "y": -1}, None),
    ]

    def numeric_emulate(self, io_map):
        """ Meta-Function numeric emulation """
        int_size = 3
        frac_size = self.width - int_size
        input_precision = fixed_point(int_size, frac_size)
        output_precision = fixed_point(int_size, frac_size)

        value_x = io_map["x"]
        value_y = io_map["y"]
        test = value_x > 1
        large_add = output_precision.truncate(value_x + value_y)
        result_value = 1 if test else large_add
        result = {
            "vr_out": result_value
        }
        print(io_map, result)
        return result


    @staticmethod
    def __call__(args):
        # just ignore args here and trust default constructor?
        # seems like a bad idea.
        ut_adaptative_entity = AdaptativeEntity(args)
        ut_adaptative_entity.gen_implementation()

        return True

run_test = AdaptativeEntity


if __name__ == "__main__":
        # auto-test
    main_arg_template = ML_EntityArgTemplate(
        default_entity_name="new_adapt_entity",
        default_output_file="mt_adapt_entity.vhd",
        default_arg=AdaptativeEntity.get_default_args()
    )
    main_arg_template.parser.add_argument(
        "--width", dest="width", type=int, default=32,
        help="set input width value (in bits)"
    )
    # argument extraction
    args = parse_arg_index_list = main_arg_template.arg_extraction()

    ml_adaptative = AdaptativeEntity(args)

    ml_adaptative.gen_implementation()
