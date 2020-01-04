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

from sollya import floor, log2
from sollya import parse as sollya_parse

from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_table import ML_Table
import metalibm_core.code_generation.vhdl_backend as vhdl_backend
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_entity import (
    ML_Entity, ML_EntityBasis, DefaultEntityArgTemplate,
    get_input_assign, get_input_msg, get_output_value_msg,
    get_output_check_statement
)


from metalibm_core.utility.ml_template import *
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import *
from metalibm_core.utility.num_utils   import ulp
from metalibm_core.utility.gappa_utils import is_gappa_installed

from metalibm_core.utility.rtl_debug_utils import debug_dec, debug_std


from metalibm_core.core.ml_hdl_format import *
from metalibm_core.core.ml_hdl_operations import *

from metalibm_hw_blocks.lzc import ML_LeadingZeroCounter


class ML_LeadingZeroAnticipator(ML_Entity("ml_lza")):
    @staticmethod
    def get_default_args(width=32, signed=False):
        return DefaultEntityArgTemplate(
             precision = ML_Int32,
             debug_flag = False,
             target = vhdl_backend.VHDLBackend(),
             output_file = "my_lza.vhd",
             entity_name = "my_lza",
             language = VHDL_Code,
             width = width,
             signed = signed,
        )

    @staticmethod
    def generate_lza(lhs_input, rhs_input, lhs_signed=False, rhs_signed=False):
        """ Generate LZA sub-graph
            returning POSitive and NEGative leading zero counts,
            lhs and rhs are assumed to be right aligned (LSB have identical weights) """
        lhs_size = lhs_input.get_precision().get_bit_size()
        rhs_size = rhs_input.get_precision().get_bit_size()

        lhs_raw_format = ML_StdLogicVectorFormat(lhs_size)
        lhs_fixed_format = fixed_point(lhs_size, 0, signed=lhs_signed)
        rhs_raw_format = ML_StdLogicVectorFormat(rhs_size)
        rhs_fixed_format = fixed_point(rhs_size, 0, signed=rhs_signed)

        common_size = 1 + max(rhs_size, lhs_size)
        common_fixed_format = fixed_point(
            common_size, 0,
            signed=(lhs_signed or rhs_signed)
        )
        common_raw_format = ML_StdLogicVectorFormat(common_size)

        lhs = TypeCast(
            Conversion(
                TypeCast(
                    lhs_input,
                    precision=lhs_fixed_format
                ),
                precision=common_fixed_format
            ),
            precision=common_raw_format
        )
        rhs = TypeCast(
            Conversion(
                TypeCast(
                    rhs_input,
                    precision=rhs_fixed_format
                ),
                precision=common_fixed_format
            ),
            precision=common_raw_format
        )


        # design based on "1 GHz Leading Zero Anticipator Using Independent
        #                 Sign-Bit Determination Logic"
        # by K. T. Lee and K. J. Nowka

        propagate = BitLogicXor(
            lhs, rhs,
            tag="propagate",
            precision=common_raw_format
        )
        kill = BitLogicAnd(
            BitLogicNegate(lhs, precision=common_raw_format),
            BitLogicNegate(rhs, precision=common_raw_format),
            tag="kill",
            precision=common_raw_format
        )
        generate_s = BitLogicAnd(
            lhs, rhs,
            tag="generate_s",
            precision=common_raw_format
        )

        pos_signal = BitLogicNegate(
            BitLogicXor(
                SubSignalSelection(propagate, 1, common_size - 1),
                SubSignalSelection(kill, 0, common_size - 2),
                precision=ML_StdLogicVectorFormat(common_size - 1)
            ),
            tag="pos_signal",
            debug=debug_std,
            precision=ML_StdLogicVectorFormat(common_size-1)
        )
        neg_signal = BitLogicNegate(
            BitLogicXor(
                SubSignalSelection(propagate, 1, common_size - 1),
                SubSignalSelection(generate_s, 0, common_size - 2),
                precision=ML_StdLogicVectorFormat(common_size - 1)
            ),
            tag="neg_signal",
            debug=debug_std,
            precision=ML_StdLogicVectorFormat(common_size - 1)
        )

        lzc_width = int(floor(log2(common_size-1))) + 1
        lzc_format = ML_StdLogicVectorFormat(lzc_width)

        pos_lzc = CountLeadingZeros(
            pos_signal,
            tag="pos_lzc",
            precision=lzc_format
        )
        neg_lzc = CountLeadingZeros(
            neg_signal,
            tag="neg_lzc",
            precision=lzc_format
        )
        return pos_lzc, neg_lzc

    def __init__(self, arg_template = None):
        # building default arg_template if necessary
        arg_template = ML_LeadingZeroAnticipator.get_default_args() if arg_template is None else arg_template
        # initializing I/O precision
        self.width = arg_template.width
        precision = arg_template.precision
        io_precisions = [precision] * 2
        Log.report(Log.Info, "generating LZC with width={}".format(self.width))

        # initializing base class
        ML_EntityBasis.__init__(self,
          base_name = "ml_lza",
          arg_template = arg_template
        )

        self.accuracy  = arg_template.accuracy
        self.precision = arg_template.precision
        self.signed = arg_template.signed

    def generate_scheme(self):
        fixed_precision = fixed_point(self.width, 0, signed=self.signed)
        # declaring main input variable
        vx = self.implementation.add_input_signal("x", fixed_precision)
        vy = self.implementation.add_input_signal("y", fixed_precision)

        ext_width = self.width + 1

        fixed_precision_ext = fixed_point(ext_width, 0, signed=self.signed)
        input_precision = ML_StdLogicVectorFormat(ext_width)

        pos_lzc, neg_lzc = ML_LeadingZeroAnticipator.generate_lza(
            vx, vy, lhs_signed=self.signed, rhs_signed=self.signed
        )

        self.implementation.add_output_signal("pos_lzc_o", pos_lzc)
        self.implementation.add_output_signal("neg_lzc_o", neg_lzc)

        return [self.implementation]

    def numeric_emulate(self, io_map):
        """ emulate leading zero anticipation """
        def count_leading_zero(v, w):
            """ generic leading zero count """
            tmp = v
            lzc = -1
            for i in range(w):
                if int(tmp) & 2**(w - 1 - i):
                    return i
            return w
        vx = io_map["x"]
        vy = io_map["y"]
        pre_op = abs(vx + vy)
        result = {}
        result["final_lzc"] = count_leading_zero(pre_op, self.width+1)
        return result

    def implement_test_case(self, io_map, input_values, output_signals, output_values, time_step):
        """ Implement the test case check and assertion whose I/Os values
            are described in input_values and output_values dict """
        test_statement = Statement()
        # string message describing expected input values
        # and dumping actual results
        input_msg = ""

        # Adding input setting
        for input_tag in input_values:
          input_signal = io_map[input_tag]
          # FIXME: correct value generation depending on signal precision
          input_value = input_values[input_tag]
          test_statement.add(get_input_assign(input_signal, input_value))
          input_msg += get_input_msg(input_tag, input_signal, input_value)
        test_statement.add(Wait(time_step * self.stage_num))

        final_lzc_value = output_values["final_lzc"] 
        vx = input_values["x"]
        vy = input_values["y"]

        pos_lzc = output_signals["pos_lzc_o"]
        neg_lzc = output_signals["neg_lzc_o"]
        if vx + vy >= 0:
            # positive result case
            main_lzc = pos_lzc
            output_tag = "pos_lzc_o"
        else:
            # negative result case
            main_lzc = neg_lzc
            output_tag = "neg_lzc_o"

        value_msg = get_output_value_msg(main_lzc, final_lzc_value)

        test_pass_cond = LogicalOr(
            Comparison(
                main_lzc,
                Constant(final_lzc_value, precision=main_lzc.get_precision()),
                precision=ML_Bool,
                specifier=Comparison.Equal
            ),
            Comparison(
                main_lzc,
                Constant(final_lzc_value - 1, precision=main_lzc.get_precision()),
                precision=ML_Bool,
                specifier=Comparison.Equal
            ),
            precision=ML_Bool
        )
        check_statement = ConditionBlock(
            LogicalNot(
                test_pass_cond,
                precision = ML_Bool
            ),
            Report(
                Concatenation(
                    " result for {}: ".format(output_tag),
                    Conversion(
                        TypeCast(
                            main_lzc,
                            precision = ML_StdLogicVectorFormat(
                                main_lzc.get_precision().get_bit_size()
                            )
                         ),
                        precision = ML_String
                        ),
                    precision = ML_String
                )
            )
        )

        test_statement.add(check_statement)
        assert_statement = Assert(
          test_pass_cond,
          "\"unexpected value for inputs {input_msg}, output {output_tag}, expecting {value_msg}, got: \"".format(
              input_msg=input_msg,
              output_tag=output_tag,
              value_msg=value_msg),
          severity = Assert.Failure
        )
        test_statement.add(assert_statement)

        return test_statement


    standard_test_cases =[
    ({
        "x": 0,
        "y": 1,
    }, None),
    ({
        "x": 0,
        "y": 0,
    }, None),
    ({
        "x": -2,
        "y": 1,
    }, None),
    ]


if __name__ == "__main__":
    # auto-test
    arg_template = ML_EntityArgTemplate(
        default_entity_name="new_lza",
        default_output_file="ml_lza.vhd",
        default_arg=ML_LeadingZeroAnticipator.get_default_args())
    arg_template.parser.add_argument(
        "--width", dest="width", type=int, default=32, help="set input width value (in bits)")
    arg_template.parser.add_argument(
        "--signed", dest="signed", action="store_const", const=True, default=False, help="enable signed input for LZA")

    # argument extraction
    args = arg_template.arg_extraction()

    ml_lza = ML_LeadingZeroAnticipator(args)

    ml_lza.gen_implementation()
