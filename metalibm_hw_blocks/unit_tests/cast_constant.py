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
""" Adaptative fixed-point size unit test """

import sollya

from sollya import parse as sollya_parse
from sollya import Interval, inf, sup

from metalibm_core.core.ml_operations import (
    Comparison, Addition, Select, Constant, Conversion, Min, Max,
    Ceil, Floor, Trunc
)
from metalibm_core.core.advanced_operations import FixedPointPosition
from metalibm_core.core.ml_hdl_operations import BitSelection

from metalibm_core.code_generation.code_constant import VHDL_Code
from metalibm_core.core.ml_formats import (
    ML_Int32
)
from metalibm_core.code_generation.vhdl_backend import VHDLBackend
from metalibm_core.core.ml_entity import (
    ML_Entity, ML_EntityBasis, DefaultEntityArgTemplate
)
from metalibm_core.utility.ml_template import (
    ML_EntityArgTemplate, hdl_precision_parser
)
from metalibm_core.utility.log_report import Log
from metalibm_core.core.ml_hdl_format import fixed_point

from metalibm_functions.unit_tests.utils import TestRunner

from metalibm_core.opt.opt_utils import evaluate_range


from metalibm_core.utility.rtl_debug_utils import (
    debug_std, debug_dec, debug_fixed
)



class UT_CastConstant(ML_Entity("ml_ut_fixed_point_position"), TestRunner):
    """ Range Eval Entity unit-test """
    @staticmethod
    def get_default_args(width=32, **kw):
        """ generate default argument template """
        return DefaultEntityArgTemplate(
            precision=ML_Int32,
            debug_flag=False,
            target=VHDLBackend(),
            output_file="ut_cast_constant.vhd",
            entity_name="ut_cast_constant",
            language=VHDL_Code,
            width=width,
            passes=[("beforecodegen:size_datapath")],
        )

    def __init__(self, arg_template=None):
        """ Initialize """
        # building default arg_template if necessary
        arg_template = UT_CastConstant.get_default_args() if \
            arg_template is None else arg_template

        # initializing base class
        ML_EntityBasis.__init__(self,
                                base_name="ut_cast_constant",
                                arg_template=arg_template
                                )

        self.accuracy = arg_template.accuracy
        self.precision = arg_template.precision
        # extra width parameter
        self.width = arg_template.width

    def generate_scheme(self):
        """ main scheme generation """

        int_size = 3
        frac_size = self.width - int_size

        input_precision = hdl_precision_parser("FU%d.%d" % (int_size, frac_size))
        output_precision = hdl_precision_parser("FS%d.%d" % (int_size, frac_size))


        # declaring main input variable
        var_x = self.implementation.add_input_signal("x", input_precision)

        var_y = self.implementation.add_input_signal("y", input_precision)

        var_z = self.implementation.add_input_signal("z", input_precision)

        abstract_formulae = var_x

        anchor = FixedPointPosition(
            abstract_formulae,
            - 3,
            align=FixedPointPosition.FromPointToMSB,
            tag="anchor"
        )

        comp = abstract_formulae > anchor

        result = Select(
            comp,
            Conversion(var_x, precision=self.precision),
            Conversion(var_y, precision=self.precision)
        )


        self.implementation.add_output_signal("result", result)


        return [self.implementation]

    def numeric_emulate(self, io_map):
        """ Meta-Function numeric emulation """
        raise NotImplementedError


    @staticmethod
    def __call__(args):
        # just ignore args here and trust default constructor?
        # seems like a bad idea.
        ut_cast_constant = UT_CastConstant(args)
        ut_cast_constant.gen_implementation()

        return True

run_test = UT_CastConstant


if __name__ == "__main__":
    # auto-test
    main_arg_template = ML_EntityArgTemplate(
        default_entity_name="ut_cast_constant",
        default_output_file="ut_cast_constant.vhd",
        default_arg=UT_CastConstant.get_default_args()
    )
    main_arg_template.parser.add_argument(
        "--width", dest="width", type=int, default=32,
        help="set input width value (in bits)"
    )

    # argument extraction
    args = parse_arg_index_list = main_arg_template.arg_extraction()

    ml_range_eval = UT_CastConstant(args)

    ml_range_eval.gen_implementation()
