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

from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import ML_Int32
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_entity import ML_EntityBasis

import metalibm_core.code_generation.vhdl_backend as vhdl_backend
from metalibm_core.code_generation.code_constant import VHDL_Code

from metalibm_core.utility.ml_template import (
    DefaultEntityArgTemplate, ML_EntityArgTemplate
)
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import *

from metalibm_core.utility.rtl_debug_utils import debug_dec


from metalibm_core.core.ml_hdl_format import (
    ML_StdLogicVectorFormat, fixed_point)
from metalibm_core.core.ml_hdl_operations import Signal

class CompoundAdder(ML_EntityBasis):
    entity_name = "compound_adder"
    @staticmethod
    def get_default_args(entity_name="my_compound_adder", **kw):
        DEFAULT_VALUES = {
             "precision": fixed_point(34, 0, signed=False),
             "debug_flag": False,
             "target": vhdl_backend.VHDLBackend(),
             "output_file": "my_compound_adder.vhd",
             "io_formats": {"x": fixed_point(32, 0, signed=False), "y": fixed_point(32, 0, signed=False)},
             "passes": ["beforepipelining:size_datapath", "beforepipelining:rtl_legalize", "beforepipelining:unify_pipeline_stages"],
             "entity_name": entity_name,
             "language": VHDL_Code,
        }
        DEFAULT_VALUES.update(kw)
        return DefaultEntityArgTemplate(
            **DEFAULT_VALUES,
        )

    def __init__(self, arg_template=None):
        # building default arg_template if necessary
        arg_template = CompoundAdder.get_default_args() if arg_template is None else arg_template

        # initializing base class
        ML_EntityBasis.__init__(self,
          base_name = "compound_adder",
          arg_template = arg_template
        )

        self.accuracy  = arg_template.accuracy
        self.precision = arg_template.precision

    def numeric_emulate(self, io_map):
        vx = io_map["x"]
        vy = io_map["y"]
        result = {}
        result["add_r"] = vx + vy
        result["addp1_r"] = vx + vy + 1
        return result


    def generate_interfaces(self):
        # declaring main input variable
        vx = self.implementation.add_input_signal("x", self.get_io_format("x"))
        vy = self.implementation.add_input_signal("y", self.get_io_format("y"))
        # declaring main output
        precision = self.precision
        dummy_add_r = Signal("add_r", precision=precision, var_type=Variable.Local)
        dummy_addp1_r = Signal("add_process", precision=precision, var_type=Variable.Local)
        self.implementation.add_output_signal("add_r", dummy_add_r)
        self.implementation.add_output_signal("addp1_r", dummy_addp1_r)
        return vx, vy


    def generate_scheme(self, skip_interface_gen=False):
        # retrieving I/Os
        vx, vy = self.generate_interfaces()

        dummy_add_r = Signal("add_r", precision=self.precision, var_type=Variable.Local)
        dummy_addp1_r = Signal("add_process", precision=self.precision, var_type=Variable.Local)

        self.implementation.set_output_signal("add_r", Conversion(vx + vy, precision=self.precision))
        self.implementation.set_output_signal("addp1_r", Conversion(vx + vy + 1, precision=self.precision))

        return [self.implementation]


Log.report(Log.Info, "installing CompoundAdder legalizer in vhdl backend")
#vhdl_backend.handle_LZC_legalizer.optree_modifier = vhdl_legalize_count_leading_zeros

if __name__ == "__main__":
    # auto-test
    arg_template = ML_EntityArgTemplate(
        default_entity_name="compound_adder",
        default_output_file="compound_adder.vhd",
        default_arg=CompoundAdder.get_default_args())

    # argument extraction
    args = parse_arg_index_list = arg_template.arg_extraction()

    compound_adder           = CompoundAdder(args)

    compound_adder.gen_implementation()
