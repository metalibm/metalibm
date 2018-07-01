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

from metalibm_core.core.ml_operations import (
    Comparison, Addition, Select, Constant, Conversion,
    Min, Max, Equal,
    Statement
)
from metalibm_core.core.ml_hdl_operations import (
    StaticDelay, equal_to
)
from metalibm_core.code_generation.code_constant import VHDL_Code
from metalibm_core.core.ml_formats import (
    ML_Int32
)
from metalibm_core.core.ml_hdl_format import (
    ML_StdLogic
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


from metalibm_core.utility.rtl_debug_utils import (
    debug_std, debug_dec, debug_fixed
)


class UTStaticDelay(ML_Entity("ut_static_delay"), TestRunner):
    """ Adaptative Entity unit-test """
    @staticmethod
    def get_default_args(width=32, **kw):
        """ generate default argument template """
        return DefaultEntityArgTemplate(
            precision=ML_Int32,
            debug_flag=False,
            target=VHDLBackend(),
            output_file="my_static_delay.vhd",
            entity_name="my_static_delay",
            language=VHDL_Code,
            width=width,
            pipelined=True,
            passes=[
                ("beforepipelining:dump_with_stages"),
                ("beforepipelining:size_datapath"),
                ("beforepipelining:dump_with_stages"),
                ("beforepipelining:rtl_legalize"),
                ("beforepipelining:dump_with_stages"),
                ("beforepipelining:unify_pipeline_stages"),
                ("beforepipelining:dump_with_stages"),
                ],
        )

    def __init__(self, arg_template=None):
        """ Initialize """
        # building default arg_template if necessary
        arg_template = UTStaticDelay.get_default_args() if \
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

        def check_function(optree):
            """ Check that every node (except Statements) have a defined
                init_stage attributes """
            if isinstance(optree, Statement):
                return True
            else:
                init_stage = optree.attributes.get_dyn_attribute("init_stage")
                if init_stage is None:
                    raise Exception("Check of init_state definition failed")
                else:
                    return True

        Log.report(Log.Info, "registering pass to check results")
        check_pass = Pass_CheckGeneric(
            self.backend,
            check_function,
            "checking pass"
        )
        #self.get_pass_scheduler().register_pass(
        #    check_pass, pass_slot = PassScheduler.JustBeforeCodeGen
        #)

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

        untimed_i = self.implementation.add_input_signal("untimed_i", ML_StdLogic)

        sub = var_x - var_y
        c = Constant(0)

        self.implementation.start_new_stage()

        #pre_result = Select(
        #    c > sub,
        #    c,
        #    sub
        #)
        pre_result = Max(0, sub)

        self.implementation.start_new_stage()

        self.implementation.start_new_stage()

        self.implementation.start_new_stage()

        result = Conversion(pre_result + var_x, precision=output_precision)

        result = Select(
            Equal(
                StaticDelay(untimed_i, 0),
                StaticDelay(untimed_i, 2, relative=False),
            ),
            result,
            Conversion(var_x, precision=output_precision)
        )

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
        raise NotImplementedError


    @staticmethod
    def __call__(args):
        # just ignore args here and trust default constructor?
        # seems like a bad idea.
        ut_static_delay = UTStaticDelay(args)
        ut_static_delay.gen_implementation()

        return True

run_test = UTStaticDelay


if __name__ == "__main__":
        # auto-test
    main_arg_template = ML_EntityArgTemplate(
        default_entity_name="ut_unify_pipeline_bench_entity",
        default_output_file="ut_unify_pipeline_bench_entity.vhd",
        default_arg=UTStaticDelay.get_default_args()
    )
    main_arg_template.parser.add_argument(
        "--width", dest="width", type=int, default=32,
        help="set input width value (in bits)"
    )
    # argument extraction
    args = parse_arg_index_list = main_arg_template.arg_extraction()

    ut_unify_pipeline_bench = UTStaticDelay(args)

    ut_unify_pipeline_bench.gen_implementation()
