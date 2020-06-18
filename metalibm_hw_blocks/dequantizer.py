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
# last-modified:      Mar  7th, 2018
# Author(s): Nicolas Brunie <nbrunie@kalray.eu>
###############################################################################
import sys

import sollya

from sollya import Interval, floor, round
S2 = sollya.SollyaObject(2)

from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_entity import ML_Entity, ML_EntityBasis, DefaultEntityArgTemplate


from metalibm_core.utility.ml_template import *
from metalibm_core.utility.log_report    import Log


from metalibm_core.core.ml_hdl_format import *
from metalibm_core.core.ml_hdl_operations import *
from metalibm_core.core.advanced_operations import FixedPointPosition

from metalibm_core.opt.opt_utils import logical_or_reduce


from metalibm_core.utility.rtl_debug_utils import (
    debug_fixed, debug_dec, debug_std, debug_dec_unsigned, debug_cst_dec
)

def fixed_exponent(op):
    e = op.get_precision().get_base_format().get_exponent_size()
    pre_exp_precision = ML_StdLogicVectorFormat(e)
    pre_exp = RawExponentExtraction(op, precision=pre_exp_precision)
    return TypeCast(
        pre_exp, precision=fixed_point(e, 0, signed=False),
    )
def fixed_mantissa(op):
    """ Extract the mantissa of op and returns it as an exact fixed-point
        integer (no fractionnal part) """
    m = op.get_precision().get_base_format().get_mantissa_size()
    pre_mant_precision = ML_StdLogicVectorFormat(m)
    pre_mant = MantissaExtraction(op, precision=pre_mant_precision)
    return TypeCast(
        pre_mant, precision=fixed_point(m, 0, signed=False)
    )

class Dequantizer(ML_Entity("dequantizer")):
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

    ## Generate default arguments structure (before any user / test overload)
    @staticmethod
    def get_default_args(**kw):
        default_arg_map = {
            "io_formats": {
                "scale": HdlVirtualFormat(ML_Binary32),
                "quantized_input": ML_Int32,
                "offset_input": ML_Int32,
                "result": ML_Int32
            },
            "pipelined": False,
            "output_file": "dequantizer.vhd",
            "entity_name": "dequantizer",
            "language": VHDL_Code,
            "passes": ["beforecodegen:size_datapath", "beforecodegen:rtl_legalize"],
        }
        default_arg_map.update(**kw)
        return DefaultEntityArgTemplate(**default_arg_map)

    def generate_scheme(self):
        scale_format = self.get_io_format("scale")
        quantized_input_format = self.get_io_format("quantized_input")
        offset_input_format = self.get_io_format("offset_input")

        scale = self.implementation.add_input_variable("scale", scale_format)
        quantized_input = self.implementation.add_input_variable("quantized_input", quantized_input_format)
        offset_input = self.implementation.add_input_variable("offset", offset_input_format)

        support_format = self.precision.get_support_format()
        base_format = self.precision.get_base_format()

        exp_precision = fixed_point(base_format.get_exponent_size(), 0, signed=False)

        p = base_format.get_field_size()
        n = support_format.get_bit_size()

        scale_exp = fixed_exponent(scale).modify_attributes(tag="scale_exp", debug=debug_fixed)
        scale_mant = fixed_mantissa(scale)

        unscaled_field = scale_mant * quantized_input
        # p - 1 (precision without implicit one, or length of mantissa fractionnal part)
        pm1 = scale.get_precision().get_base_format().get_mantissa_size() - 1
        # MAX_SHIFT computed such that no bit is lost (and kept for proper rounding)
        #           an extra +1 is added to ensure correct bit is used as round bit
        PRODUCT_SIZE = scale_mant.get_precision().get_bit_size() + quantized_input.get_precision().get_bit_size()
        MAX_SHIFT = PRODUCT_SIZE + 1
        shift_amount = Min(scale_exp - pm1, MAX_SHIFT)
        pre_shift_field = Conversion(unscaled_field, precision=fixed_point(PRODUCT_SIZE, MAX_SHIFT))
        scaled_field = BitLogicRightShift(pre_shift_field, shift_amount)

        #truncated_field = Conversion(scaled_field, precision=offset_input_format)
        #offseted_field = truncated_field + offset_input
        offseted_field = scaled_field + Conversion(offset_input, precision=fixed_point(offset_input_format.get_bit_size(), 0))

        round_bit = BitSelection(offseted_field, FixedPointPosition(offseted_field, -1, align=FixedPointPosition.FromPointToLSB))
        sticky_bit = NotEqual(SubSignalSelection(offseted_field, 0, FixedPointPosition(offseted_field, -2, align=FixedPointPosition.FromPointToLSB)), 0)

        # TODO: implement rounding

        result_format = self.get_io_format("result")
        result = Conversion(offseted_field, precision=result_format)

        self.implementation.add_output_signal("result", result)
        return [self.implementation]

    def numeric_emulate(self, io_map):
        qinput = io_map["quantized_input"]
        scale = io_map["scale"]
        offset = io_map["offset"]
        result = {}
        result["result"] = int(scale * qinput + offset)
        return result

    standard_test_cases = [
    ]


if __name__ == "__main__":
    # auto-test
    arg_template = ML_EntityArgTemplate(
        default_entity_name="dequantizer", default_output_file="dequantizer.vhd",
        default_arg=Dequantizer.get_default_args()
    )
    # argument extraction
    args = arg_template.arg_extraction()

    ml_hw_adder      = Dequantizer(args)

    ml_hw_adder.gen_implementation()
