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

def fixed_exponent(op):
    e = op.get_precision().get_base_format().get_exponent_size()
    pre_exp_precision = ML_StdLogicVectorFormat(e)
    pre_exp = RawExponentExtraction(op, precision=pre_exp_precision)
    return TypeCast(
        pre_exp, precision=fixed_point(e, 0, signed=False),
    )
def fixed_normalized_mantissa(op):
    """ Extract the mantissa of op and returns it as an exact fixed-point
        integer (no fractionnal part) """
    m = op.get_precision().get_base_format().get_mantissa_size()
    pre_mant_precision = ML_StdLogicVectorFormat(m)
    pre_mant = MantissaExtraction(op, precision=pre_mant_precision)
    return TypeCast(
        pre_mant, precision=fixed_point(1, m-1, signed=False)
    )

FIX32 = fixed_point(32, 0, signed=True)
ROUNDING_MODE_FORMAT = ML_StdLogicVectorFormat(3)

ROUND_RNE = Constant(0, precision=ROUNDING_MODE_FORMAT)
ROUND_RU = Constant(1, precision=ROUNDING_MODE_FORMAT)
ROUND_RD = Constant(2, precision=ROUNDING_MODE_FORMAT)
ROUND_RZ = Constant(3, precision=ROUNDING_MODE_FORMAT)
ROUND_RAZ = Constant(4, precision=ROUNDING_MODE_FORMAT)

ROUND_RNE = 0
ROUND_RU = 1
ROUND_RD = 2
ROUND_RZ = 3
ROUND_RAZ = 4

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
                "quantized_input": FIX32,
                "offset_input": FIX32,
                "result": FIX32
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
        # quantized_input * scale + offset_input
        scale_format = self.get_io_format("scale")
        quantized_input_format = self.get_io_format("quantized_input")
        offset_input_format = self.get_io_format("offset_input")
        result_format = self.get_io_format("result")

        RESULT_SIZE = result_format.get_bit_size()

        scale = self.implementation.add_input_variable("scale", scale_format)
        quantized_input = self.implementation.add_input_variable("quantized_input", quantized_input_format)
        offset_input = self.implementation.add_input_variable("offset", offset_input_format)
        rounding_mode = self.implementation.add_input_variable("rounding_mode", ML_StdLogicVectorFormat(3))

        support_format = self.precision.get_support_format()
        base_format = self.precision.get_base_format()

        exp_precision = fixed_point(base_format.get_exponent_size(), 0, signed=False)

        p = base_format.get_field_size()
        n = support_format.get_bit_size()

        biased_scale_exp = fixed_exponent(scale)#.modify_attributes(tag="scale_exp", debug=debug_fixed)
        scale_exp = biased_scale_exp + scale_format.get_base_format().get_bias()
        scale_exp.set_attributes(tag="scale_exp", debug=debug_fixed)
        scale_sign = ExtractSign(scale, precision=ML_StdLogic, tag="scale_sign")
        unsigned_scale_mant = fixed_normalized_mantissa(scale)

        signed_mant_format = fixed_point(2, scale.get_precision().get_base_format().get_mantissa_size()-1, signed=True)
        scale_mant = Select(scale_sign, -unsigned_scale_mant, unsigned_scale_mant, tag="scale_mant", precision=signed_mant_format, debug=debug_fixed)


        # unscaled field is in fixed-point normalized format
        unscaled_field = scale_mant * quantized_input
        unscaled_field.set_attributes(tag="unscaled_field", debug=debug_fixed)
        # p - 1 (precision without implicit one, or length of mantissa fractionnal part)
        pm1 = scale.get_precision().get_base_format().get_mantissa_size() - 1
        # PRODUCT_SIZE is the width of the unscaled scale * input "mantissa" product
        PRODUCT_SIZE = scale_mant.get_precision().get_bit_size() + quantized_input.get_precision().get_bit_size()
        # MAX_SHIFT computed such that no bit is lost (and kept for proper rounding)
        #           an extra +1 is added to ensure correct bit is used as round bit
        MAX_SHIFT = RESULT_SIZE + 1 + PRODUCT_SIZE + 1
        # TODO/FIXME: manage case where shift_amount < 0 (should be forced to 0)
        shift_amount = Max(Min(-scale_exp + RESULT_SIZE + 1, MAX_SHIFT, tag="shift_amount", debug=debug_fixed), 0)
        # unscaled_field is widended (padded with "0" right")
        # TODO/FIXME manage fixed-point format signedness
        extended_unscaled_field = Conversion(unscaled_field, precision=fixed_point(PRODUCT_SIZE, MAX_SHIFT))
        # widened unscaled_field is casted to set 0-padding as fractionnary part
        # TODO/FIXME manage fixed-point format signedness
        pre_shift_field = TypeCast(extended_unscaled_field, precision=fixed_point(MAX_SHIFT - 1, PRODUCT_SIZE + 1), tag="pre_shift_field", debug=debug_std)
        scaled_field = BitArithmeticRightShift(pre_shift_field, shift_amount, tag="scaled_field", debug=debug_std)

        #truncated_field = Conversion(scaled_field, precision=offset_input_format)
        #offseted_field = truncated_field + offset_input
        offseted_field = scaled_field + Conversion(offset_input, precision=fixed_point(offset_input_format.get_bit_size(), 0), tag="extended_offset", debug=debug_std)
        offseted_field.set_attributes(tag="offseted_field", debug=debug_std)

        round_bit = Equal(
            BitSelection(offseted_field, FixedPointPosition(offseted_field, -1, align=FixedPointPosition.FromPointToLSB), tag="round_bit"),
            Constant(0, precision=ML_StdLogic))
        sticky_bit = NotEqual(SubSignalSelection(offseted_field, 0, FixedPointPosition(offseted_field, -2, align=FixedPointPosition.FromPointToLSB), tag="sticky_bitfield", precision=None), 0, tag="sticky_bit")

        offseted_field_negative = offseted_field < 0
        offseted_field_parity_bit = BitSelection(offseted_field, FixedPointPosition(offseted_field, 0, align=FixedPointPosition.FromPointToLSB), tag="parity_bit")

        offseted_field_even = Equal(offseted_field_parity_bit, Constant(0, precision=ML_StdLogic), tag="offseted_field_even")

        # TODO: implement rounding
        # increment if round-up and (round_bit or sticky_bit)
        #           if round-rz and (result negative) and (round_bit or sticky_bit)
        #           if round-rne and (round_bit and (sticky_bit or (not result even)))
        #           if round-raz and (round_bit or sticky_bit and (result positive))
        round_up = Equal(rounding_mode, ROUND_RU)
        round_rz = Equal(rounding_mode, ROUND_RZ)
        round_rne = Equal(rounding_mode, ROUND_RNE)
        round_down = Equal(rounding_mode, ROUND_RD)
        round_raz = Equal(rounding_mode, ROUND_RAZ)
        round_increment = Select(
            logical_or_reduce([
                LogicalAnd(round_up, LogicalOr(round_bit, sticky_bit)),
                LogicalAnd(round_rz, LogicalAnd(offseted_field_negative, LogicalOr(round_bit, sticky_bit))),
                LogicalAnd(round_rne, LogicalAnd(round_bit, LogicalOr(sticky_bit, offseted_field_even))),
                LogicalAnd(round_raz, LogicalAnd(LogicalOr(round_bit, sticky_bit), LogicalNot(offseted_field_negative)))
            ]),
            1,
            0,
            precision=fixed_point(1, 0, signed=False),
            tag="round_increment")

        rounded_field = offseted_field + round_increment
        result_format = self.get_io_format("result")

        # detecting overflow / underflow
        MAX_BOUND = self.get_io_format("result").get_max_value()
        MIN_BOUND = self.get_io_format("result").get_min_value()
        bounded_result = Max(MIN_BOUND, Min(rounded_field, MAX_BOUND))

        result = Conversion(bounded_result, precision=result_format)

        self.implementation.add_output_signal("result", result)
        return [self.implementation]

    def init_test_generator(self, io_map, test_range):
        """ specialization of random input generators """
        ML_EntityBasis.init_test_generator(self, io_map)
        # patching generator for rounding_mode to limit value
        if not "scale" in test_range:
            self.input_generators["scale"] = FPRandomGen(self.get_io_format("scale").get_base_format(), weight_map={FPRandomGen.Category.Normal: 1.0})
        self.input_generators["rounding_mode"] = RawLogicVectorRandomGen(3, 0, max([ROUND_RNE, ROUND_RU, ROUND_RD, ROUND_RZ, ROUND_RAZ]))

    def numeric_emulate(self, io_map):
        qinput = io_map["quantized_input"]
        scale = io_map["scale"]
        offset = io_map["offset"]
        rounding_mode = io_map["rounding_mode"]

        def round_away_from_zero(value):
            return int((sollya.ceil if value > 0 else sollya.floor)(value))
        def round_nearest_tie_away_from_zero(value):
            rounded_value = int(sollya.nearestint(value))
            # detect ties
            tie = (value == rounded_value + 0.5 or value == rounded_value - 0.5)
            away_offset = 1 if value > 0 else -1 # in tie cases value != 0
            if tie:
                assert value != 0
                if value > 0:
                    return int(sollya.floor(value)) + 1
                else:
                    return int(sollya.ceil(value)) - 1
            else:
                return rounded_value

        ROUND_FUNCTION = {
            ROUND_RNE: lambda value: int(sollya.nearestint(value)),
            ROUND_RU: lambda value: int(sollya.ceil(value)),
            ROUND_RD: lambda value: int(sollya.floor(value)),
            ROUND_RZ: lambda value: int((sollya.floor if value > 0 else sollya.ceil)(value)),
            ROUND_RAZ: round_away_from_zero,
        }
        result = {}
        # TODO/FIXME: support rounding mode
        unbounded_result = ROUND_FUNCTION[rounding_mode](sollya.SollyaObject(scale) * qinput + offset)
        # threshold clamp
        MAX_BOUND = int(self.get_io_format("result").get_max_value())
        MIN_BOUND = int(self.get_io_format("result").get_min_value())
        result["result"] = max(min(MAX_BOUND, unbounded_result), MIN_BOUND)
        return result

    standard_test_cases = [
       # debug
       ({'scale': -9.3562177440117458422781203005040841215683797976732e-28, 'quantized_input': -59, 'offset': 1476644306, 'rounding_mode': 4}, None),
       ({'scale': -3.2018401007923587122374383797585380222573657934822e-30, 'quantized_input': 1382946913, 'offset': -49, 'rounding_mode': 3}, None),
       ({'scale': 49366.34765625, 'quantized_input': -889172583, 'offset': 63342963, 'rounding_mode': 0}, None),
       ({"quantized_input": -17, "scale": sollya.parse("0xe174fea2"), "offset": 0x6c732c7a, "rounding_mode": 3}, None),
       # scale overflow
       ({"quantized_input": 17, "scale": sollya.SollyaObject(2)**100, "offset": 12, "rounding_mode": 0}, None),
       # dummy tests
       ({"quantized_input": 0, "scale": 0, "offset": 0, "rounding_mode": 0}, None),
       ({"quantized_input": 0, "scale": 0, "offset": 1, "rounding_mode": 0}, None),
       ({"quantized_input": 0, "scale": 0, "offset": 17, "rounding_mode": 0}, None),
       ({"quantized_input": 0, "scale": 0, "offset": -17, "rounding_mode": 0}, None),

       ({"quantized_input": 17, "scale": 1.0, "offset": 0, "rounding_mode": 0}, None),
       #({"quantized_input": 17, "scale": -1.0, "offset": 0, "rounding_mode": 0}, None),
       ({"quantized_input": -17, "scale": 1.0, "offset": 0, "rounding_mode": 0}, None),
       #({"quantized_input": -17, "scale": -1.0, "offset": 0, "rounding_mode": 0}, None),

       ({"quantized_input": 17, "scale": 1.0, "offset": 42, "rounding_mode": 0}, None),
       #({"quantized_input": 17, "scale": -1.0, "offset": 42, "rounding_mode": 0}, None),
       ({"quantized_input": -17, "scale": 1.0, "offset": 42, "rounding_mode": 0}, None),
       #({"quantized_input": -17, "scale": -1.0, "offset": 42, "rounding_mode": 0}, None),

       ({"quantized_input": 17, "scale": 1.125, "offset": 42, "rounding_mode": 0}, None),
       #({"quantized_input": 17, "scale": -1.0, "offset": 42, "rounding_mode": 0}, None),
       ({"quantized_input": -17, "scale": 17.0, "offset": 42, "rounding_mode": 0}, None),
       #({"quantized_input": -17, "scale": -1.0, "offset": 42, "rounding_mode": 0}, None),

       # rounding
       ({"quantized_input": 17, "scale": 0.625, "offset": 1337, "rounding_mode": 0}, None),

       # TODO: cancellation tests
       # TODO: overflow tests
       ({"quantized_input": 2**31-1, "scale": 4.0, "offset": 42, "rounding_mode": 0}, None),
       # TODO: other tests
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
