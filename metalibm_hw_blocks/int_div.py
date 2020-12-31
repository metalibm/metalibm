# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/kalray/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2020 Kalray
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
# created:            Jun  18th, 2020
# last-modified:      Jul  18th, 2020
# Author(s): Nicolas Brunie <nbrunie@kalray.eu>
###############################################################################
import sys

import sollya

from sollya import Interval, floor, round
S2 = sollya.SollyaObject(2)

from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_entity import (
    ML_Entity, ML_EntityBasis, DefaultEntityArgTemplate,
    RawLogicVectorRandomGen)


from metalibm_core.utility.ml_template import *
from metalibm_core.utility.log_report    import Log


from metalibm_core.core.ml_hdl_format import *
from metalibm_core.core.ml_hdl_operations import *
from metalibm_core.core.advanced_operations import FixedPointPosition
from metalibm_core.core.random_gen import FPRandomGen

from metalibm_core.opt.opt_utils import logical_or_reduce


from metalibm_core.utility.rtl_debug_utils import (
    debug_fixed, debug_dec, debug_std, debug_dec_unsigned, debug_cst_dec
)

# required to setup lzc as implementation for CountLeadingZeros
import metalibm_hw_blocks.lzc

FIX32 = fixed_point(32, 0, signed=False)

class MetaIntDiv(ML_EntityBasis):
    entity_name = "int_div"
    """ Implement the post-processing operator for
        a quantized neural network layer:

        quantized_input * scale + offset_input
    """
    def __init__(self, arg_template=DefaultEntityArgTemplate):
        # initializing I/O precision

        # initializing base class
        ML_EntityBasis.__init__(self,
          arg_template = arg_template
        )

        self.precision = arg_template.precision
        self.div_by_zero_result = arg_template.div_by_zero_result

    ## Generate default arguments structure (before any user / test overload)
    @staticmethod
    def get_default_args(**kw):
        default_arg_map = {
            "io_formats": {
                "dividend": FIX32,
                "divisor": FIX32,
                "result": FIX32
            },
            "pipelined": False,
            "output_file": "int_div.vhd",
            "entity_name": "int_div",
            "div_by_zero_result": FIX32.get_max_value(),
            "auto_test_range": {"dividend": Interval(0, 2**32-1), "divisor": Interval(0, 2**32-1)},
            "language": VHDL_Code,
            "passes": ["beforecodegen:size_datapath", "beforecodegen:rtl_legalize"],
        }
        default_arg_map.update(**kw)
        return DefaultEntityArgTemplate(**default_arg_map)

    def generate_scheme(self):
        return self.generate_scheme_basic()

    def generate_scheme_basic(self):
        """ Simple single-digit iteration algorithm """
        # quantized_input * scale + offset_input
        divisor_format = self.get_io_format("divisor")
        dividend_format = self.get_io_format("dividend")
        result_format = self.get_io_format("result")

        RESULT_SIZE = result_format.get_bit_size()

        dividend = self.implementation.add_input_variable("dividend", dividend_format)
        divisor = self.implementation.add_input_variable("divisor", divisor_format)

        lzc_dividend = CountLeadingZeros(dividend)
        lzc_divisor = CountLeadingZeros(divisor)
        lzc_delta = lzc_divisor - lzc_dividend
        normalized_divisor = BitLogicLeftShift(divisor, lzc_delta, tag="normalized_divisor")
        r = dividend
        q = Constant(0, precision=result_format)
        for i in range(RESULT_SIZE): # TODO/FIXME width
            end_of_loop = lzc_delta < 0
            test = LogicalAnd(r >= normalized_divisor, LogicalNot(end_of_loop))
            new_digit = Select(test, 1, 0, tag="new_digit_step_{}".format(i))
            q = Select(end_of_loop, q, Conversion(BitLogicLeftShift(q, 1) + new_digit, precision=result_format), precision=result_format)
            r = Conversion(r - Select(test, normalized_divisor, 0), precision=dividend_format)
            normalized_divisor = BitLogicRightShift(normalized_divisor, 1)
            lzc_delta = lzc_delta - 1
            q.set_attributes(tag="q_step_{}".format(i))
            r.set_attributes(tag="r_step_{}".format(i))

        result = Select(Equal(divisor, 0), Constant(self.div_by_zero_result, precision=result_format), q, precision=result_format)

        self.implementation.add_output_signal("result", result)
        return [self.implementation]

    def generate_scheme_multi_digit(self):
        """ More advanced (though pretty basic) multi-digit iteration
            algorithm """
        # quantized_input * scale + offset_input
        divisor_format = self.get_io_format("divisor")
        dividend_format = self.get_io_format("dividend")
        result_format = self.get_io_format("result")

        RESULT_SIZE = result_format.get_bit_size()

        dividend = self.implementation.add_input_variable("dividend", dividend_format)
        divisor = self.implementation.add_input_variable("divisor", divisor_format)

        # we use N most significant bits from dividend and
        #        M most siginificant bits from divisor to address
        #        the quotient candidate table
        N = 4
        M = 4

        lzc_dividend = CountLeadingZeros(dividend)
        lzc_divisor = CountLeadingZeros(divisor)
        lzc_delta = lzc_divisor - lzc_dividend
        normalized_divisor = BitLogicLeftShift(divisor, lzc_divisor - (N - 1), tag="normalized_divisor")
        dividend_width = dividend_format.get_bit_size() 
        divisor_width = divisor_format.get_bit_size()

        index_size = N + M
        # TODO/FIXME: check integer-size part
        storage_format = fixed_point(N, 0, signed=False)

        table_candidate_quotient = ML_NewTable(
            dimensions=[2**index_size],
            storage_precision=storage_format,
            tag="table_candidate_quotient")
        for i in range(2**N):
            # specific value for j == 0
            table_candidate_quotient[i * 2**M] = 0
            for j in range(1, 2**M):
                index = i * 2**M + j
                # TODO/FIXME: math.ceil we use conversion to native float format
                #             (should be double-precision / binary64) on most platform
                #             which would entail a conversion error if N and M are too large
                quotient = int(math.ceil((i * 2**(M-1)) / j))
                table_candidate_quotient[index] = quotient

        normalized_dividend = BitLogicLeftShift(dividend, lzc_dividend, tag="normalized_dividend")
        remainder = dividend
        q = Constant(0, precision=result_format)
        HighPart_divisor = SubSignalSelection(divisor, divisor_width - 1 - (N - 1) - (M - 1), divisor_width - 1 - (N - 1))  
        for i in range(dividend_format.get_bit_size() // N): # TODO/FIXME width
            end_of_loop = lzc_delta < N
            HighPart_dividend = SubSignalSelection(remainder, dividend_width - 1 - N + 1, dividend_width - 1)

            table_index = Concatenation(HighPart_dividend, HighPart_divisor)
            candidate_quotient = TableLoad(table_candidate_quotient, table_index, precision=storage_format, tag="candidate_quotient")
            pre_remainder = remainder - candidate_quotient * divisor
            # correction
            # NOTES: correction could be replaced by redundant representation
            # for quotient, associated with final transformation into
            # canonical representation
            candidate_quotient = Select(pre_remainder < 0, candidate_quotient - 1, candidate_quotient)
            quotient = Concatenate(quotient, Conversion(candidate_quotient, precision=fixed_point(N, 0, signed=False)))
            pre_remainder = Select(pre_remainder < 0, pre_remainder + divisor, pre_remainder)

            remainder = TypeCast(BitLogicLeftShift(pre_remainder, N), precision=dividend_format) 

        # baby step
        raise NotImplementedError
        self.implementation.add_output_signal("result", result)
        return [self.implementation]

    def numeric_emulate(self, io_map):
        divisor = io_map["divisor"]
        dividend = io_map["dividend"]

        result = {}
        if divisor == 0:
            result["result"] = self.div_by_zero_result
        else:
            result["result"] = int(sollya.floor(dividend / divisor))
        return result

    standard_test_cases = [
        ({'dividend': 0xd1558b05, "divisor": 0}, None),
        ({'dividend': 0xd1558b05, "divisor": 0xb3c32dbc}, None),
        ({'dividend': 4294967281, 'divisor': 2578561850}, None),
        ({'dividend': 3146334737, 'divisor': 45}, None),
        ({"dividend": 1, "divisor": 1}, None),
        ({"dividend": 1024, "divisor": 16}, None),
        ({"dividend": 3*1337, "divisor": 3}, None),
    ]


if __name__ == "__main__":
    # auto-test
    arg_template = ML_EntityArgTemplate(
        default_entity_name="dequantizer", default_output_file="dequantizer.vhd",
        default_arg=MetaIntDiv.get_default_args())
    arg_template.parser.add_argument("--div-by-zero-result", dest="div_by_zero_result",
        type=int, default=0, help="define result returned for division by zero")
    # argument extraction
    args = arg_template.arg_extraction()

    ml_hw_adder      = MetaIntDiv(args)

    ml_hw_adder.gen_implementation()
