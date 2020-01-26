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
    ML_StdLogicVectorFormat, fixed_point, ML_StdLogic)
from metalibm_core.core.ml_hdl_operations import (
    Concatenation, multi_Concatenation,
    Signal, SubSignalSelection, BitSelection)

from metalibm_core.opt.opt_utils import logical_reduce

def vector_concatenation(*args):
    """ Concatenate a list of signals of arbitrary length into
        a single ML_StdLogicVectorFormat node whose correct precision set """
    num_args = len(args)
    if num_args == 1:
        return args[0]
    else:
        half_num = int(num_args / 2)
        lhs = vector_concatenation(*args[:half_num])
        rhs = vector_concatenation(*args[half_num:])
        return Concatenation(
            lhs,
            rhs,
            precision=ML_StdLogicVectorFormat(
                lhs.get_precision().get_bit_size() + rhs.get_precision().get_bit_size()
            )
        )

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
             "lower_limit": 1,
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
        self.lower_limit = arg_template.lower_limit

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

        # carry-select
        def rec_add(op_x, op_y, level=0, lower_limit=1):
            print("calling rec_add")
            n_x = op_x.get_precision().get_bit_size()
            n_y = op_y.get_precision().get_bit_size()
            n = max(n_x, n_y)
            if n <= lower_limit:
                if n == 1:
                    # end of recursion
                    def LSB(op): return BitSelection(op, 0)
                    bit_x = LSB(op_x)
                    bit_y = LSB(op_y)
                    return (
                        # add
                        Conversion(BitLogicXor(bit_x, bit_y, precision=ML_StdLogic), precision=ML_StdLogicVectorFormat(1)),
                        # add + 1
                        Conversion(BitLogicNegate(BitLogicXor(bit_x, bit_y, precision=ML_StdLogic), precision=ML_StdLogic), precision=ML_StdLogicVectorFormat(1)),
                        # generate
                        BitLogicAnd(bit_x, bit_y, precision=ML_StdLogic),
                        # propagate
                        BitLogicXor(bit_x, bit_y, precision=ML_StdLogic)
                    )
                else:
                    numeric_x = TypeCast(op_x, precision=fixed_point(n_x, 0, signed=False))
                    numeric_y = TypeCast(op_y, precision=fixed_point(n_y, 0, signed=False))
                    pre_add = numeric_x + numeric_y
                    pre_addp1 = pre_add + 1
                    add = SubSignalSelection(pre_add, 0, n - 1)
                    addp1 = SubSignalSelection(pre_addp1, 0, n - 1)
                    generate = BitSelection(pre_add, n)
                    # TODO/FIXME: padd when op_x's size does not match op_y' size
                    bitwise_xor = BitLogicXor(op_x, op_y, precision=ML_StdLogicVectorFormat(n))
                    op_list = [BitSelection(bitwise_xor, i) for i in range(n)]
                    propagate = logical_reduce(op_list, op_ctor=BitLogicAnd, precision=ML_StdLogic)

                    return add, addp1, generate, propagate

            half_n = int(n / 2)
            def subdivide(op, split_index_list):
                """ subdivide op node in len(split_index_list) + 1 slices
                    [0 to split_index_list[0]],
                    [split_index_list[0] + 1: split_index_list[1]], .... """
                n = op.get_precision().get_bit_size()
                sub_list = []
                lo_index = 0
                hi_index = None
                for s in split_index_list:
                    if lo_index >= n:
                        break
                    hi_index = min(s, n -1)
                    local_slice = SubSignalSelection(op, lo_index, hi_index) #, precision=fixed_point(hi_index - lo_index + 1, 0, signed=False))
                    sub_list.append(local_slice)
                    # key invariant to allow adding multiple subdivision together:
                    # the LSB weight must be uniform
                    lo_index = s + 1
                if hi_index < n - 1:
                    local_slice = SubSignalSelection(op, lo_index, n - 1)
                    sub_list.append(local_slice)
                # padding list
                while len(sub_list) < len(split_index_list) + 1:
                    sub_list.append(Constant(0))
                return sub_list

            x_slices = subdivide(op_x, [half_n - 1])
            y_slices = subdivide(op_y, [half_n - 1])


            # list of (x + y), (x + y + 1), generate, propagate
            add_slices = [rec_add(sub_x, sub_y, level=level+1, lower_limit=lower_limit) for sub_x, sub_y in zip(x_slices, y_slices)]

            NUM_SLICES = len(add_slices)

            def tree_reduce(gen_list, prop_list):
                return LogicalOr(gen_list[-1], LogicalAnd(prop_list[-1], tree_reduce(gen_list[:-1], prop_list[:-1])))

            add_list = [op[0] for op in add_slices]
            addp1_list = [op[1] for op in add_slices]
            generate_list = [op[2] for op in add_slices]
            propagate_list = [op[3] for op in add_slices]

            carry_propagate = [propagate_list[0]]
            carry_generate = [generate_list[0]]
            add_result = [add_list[0]]
            addp1_result = [addp1_list[0]]
            for i in range(1, NUM_SLICES):
                def sub_name(prefix, index):
                    return "%s_%d_%d" % (prefix, level, index)
                carry_propagate.append(BitLogicAnd(propagate_list[i], carry_propagate[i-1], tag=sub_name("carry_propagate", i), precision=ML_StdLogic))
                carry_generate.append(BitLogicOr(generate_list[i], BitLogicAnd(propagate_list[i], carry_generate[i-1], precision=ML_StdLogic), tag=sub_name("carry_generate", i), precision=ML_StdLogic))
                add_result.append(Select(carry_generate[i-1], addp1_list[i], add_list[i], tag=sub_name("add_result", i), precision=addp1_list[i].get_precision()))
                addp1_result.append(Select(BitLogicOr(carry_propagate[i-1], carry_generate[i-1], precision=ML_StdLogic), addp1_list[i], add_list[i], tag=sub_name("addp1_result", i), precision=addp1_list[i].get_precision()))
            add_result_full = vector_concatenation(
                    #Conversion(carry_generate[-1], precision=ML_StdLogicVectorFormat(1)),
                    *tuple(add_result[::-1]))
            addp1_result_full = vector_concatenation(
                    #Conversion(BitLogicOr(carry_generate[-1], carry_propagate[-1], precision=ML_StdLogic), precision=ML_StdLogicVectorFormat(1)),
                    *tuple(addp1_result[::-1]))
            return add_result_full, addp1_result_full, carry_generate[-1], carry_propagate[-1]

        pre_add, pre_addp1, last_generate, last_propagate = rec_add(vx, vy, lower_limit=self.lower_limit)
        # concatenating final carry
        add = vector_concatenation(Conversion(last_generate, precision=ML_StdLogicVectorFormat(1)), pre_add)
        addp1 = vector_concatenation(
            Conversion(BitLogicOr(last_generate, last_propagate, precision=ML_StdLogic), precision=ML_StdLogicVectorFormat(1)),
            pre_addp1)
        dummy_add_r = Signal("add_r", precision=self.precision, var_type=Variable.Local)
        dummy_addp1_r = Signal("add_process", precision=self.precision, var_type=Variable.Local)

        cvt_add = Conversion(TypeCast(add, precision=fixed_point(add.get_precision().get_bit_size(), 0, signed=False)), precision=self.precision)
        cvt_addp1 = Conversion(TypeCast(addp1, precision=fixed_point(addp1.get_precision().get_bit_size(), 0, signed=False)), precision=self.precision)

        self.implementation.set_output_signal("add_r", cvt_add)
        self.implementation.set_output_signal("addp1_r", cvt_addp1)

        return [self.implementation]

    standard_test_cases = [
        #({"x": 0xa, "y": 0x6}, None),
        #({"x": 0xa1, "y": 0x6d}, None),
        #({"x": 0x244f40a, "y": 0x8a768c6d}, None),
        #({"x": 0x5529922f, "y": 0xd18e7c}, None),
    ]


Log.report(Log.Info, "installing CompoundAdder legalizer in vhdl backend")
#vhdl_backend.handle_LZC_legalizer.optree_modifier = vhdl_legalize_count_leading_zeros

if __name__ == "__main__":
    # auto-test
    arg_template = ML_EntityArgTemplate(
        default_entity_name="compound_adder",
        default_output_file="compound_adder.vhd",
        default_arg=CompoundAdder.get_default_args())

    arg_template.parser.add_argument(
        "--lower-limit", action="store",
        default=1, type=int,
        help="set the minimum sub-slice width before a standard unexpanded Adder is used")

    # argument extraction
    args = parse_arg_index_list = arg_template.arg_extraction()

    compound_adder           = CompoundAdder(args)

    compound_adder.gen_implementation()
