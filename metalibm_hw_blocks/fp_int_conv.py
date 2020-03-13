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
# last-modified:      Mar  7th, 2018
# Author(s): Nicolas Brunie <nbrunie@kalray.eu>
###############################################################################
import sys

import sollya

from sollya import Interval, floor, round, log2
S2 = sollya.SollyaObject(2)

from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.code_generation.vhdl_backend import VHDLBackend
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_entity import ML_Entity, ML_EntityBasis, DefaultEntityArgTemplate


from metalibm_core.utility.ml_template import *
from metalibm_core.utility.log_report    import Log


from metalibm_core.core.ml_hdl_format import *
from metalibm_core.core.ml_hdl_operations import *


from metalibm_hw_blocks.lzc import ML_LeadingZeroCounter

from metalibm_core.utility.rtl_debug_utils import (
    debug_fixed, debug_dec, debug_std, debug_dec_unsigned, debug_cst_dec
)

class FP_Trunc(ML_Entity("fp_trunc")):
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
            "precision": HdlVirtualFormat(ML_Binary32),
            "pipelined": False,
            "output_file": "fp_trunc.vhd",
            "entity_name": "fp_trunc",
            "language": VHDL_Code,
            "passes": [("beforecodegen:size_datapath")],
        }
        default_arg_map.update(**kw)
        return DefaultEntityArgTemplate(**default_arg_map)

    def generate_scheme(self):
        vx = self.implementation.add_input_variable("vx", self.precision)
        support_format = self.precision.get_support_format()
        base_format = self.precision.get_base_format()

        exp_precision = fixed_point(base_format.get_exponent_size(), 0, signed=False)

        def fixed_exponent(op):
            e = op.get_precision().get_base_format().get_exponent_size()
            pre_exp_precision = ML_StdLogicVectorFormat(e)
            pre_exp = ExponentExtraction(op, precision=pre_exp_precision)
            return TypeCast(
                pre_exp, precision=fixed_point(e, 0, signed=False),
            )
        def fixed_mantissa(op):
            m = op.get_precision().get_base_format().get_mantissa_size()
            pre_mant_precision = ML_StdLogicVectorFormat(m)
            pre_mant = MantissaExtraction(op, precision=pre_mant_precision)
            return TypeCast(
                pre_mant, precision=fixed_point(m, 0, signed=False)
            )

        p = base_format.get_field_size()
        n = support_format.get_bit_size()

        vx_exp = fixed_exponent(vx).modify_attributes(tag="vx_exp", debug=debug_fixed)
        vx_mant = fixed_mantissa(vx)
        fixed_support_format = fixed_point(support_format.get_bit_size(), 0, signed=False)
        # shift amount to normalize mantissa into an integer
        int_norm_shift = Max(p - (vx_exp + base_format.get_bias()), 0, tag="int_norm_shift", debug=debug_fixed)
        pre_mant_mask = Constant(2**n-1, precision=fixed_support_format)
        mant_mask = TypeCast(
            BitLogicLeftShift(pre_mant_mask, int_norm_shift, precision=fixed_support_format),
            precision=support_format,
            tag="mant_mask",
            debug=debug_std
        )
        #mant_mask = BitLogicNegate(neg_mant_mask, precision=support_format, tag="mant_mask", debug=debug_std)

        normed_result = TypeCast(
            BitLogicAnd(
                TypeCast(vx, precision=support_format),
                mant_mask,
                precision=support_format
            ),
            precision=self.precision
        )

        vr_out = Select(
            # if exponent exceeds (precision - 1), then value
            # is equal to its integer part
            vx_exp + base_format.get_bias() > base_format.get_field_size(),
            vx,
            Select(
                vx_exp + base_format.get_bias() < 0,
                Constant(0, precision=self.precision),
                normed_result,
                precision=self.precision
            ),
            precision=self.precision
        )

        self.implementation.add_output_signal("vr_out", vr_out)
        return [self.implementation]

    def numeric_emulate(self, io_map):
        vx = io_map["vx"]
        result = {}
        base_format = self.precision.get_base_format()
        result["vr_out"] = (sollya.floor if vx > 0 else sollya.ceil)(vx)
        return result

    standard_test_cases = [
        ({"vx": ML_Binary32.get_value_from_integer_coding("0x48bef48d", base=16)}, None),
        ({"vx": 1.0}, None),
    ]


if __name__ == "__main__":
    # auto-test
    arg_template = ML_EntityArgTemplate(
        default_entity_name="fp_trunc", default_output_file="fp_trunc.vhd",
        default_arg=FP_Trunc.get_default_args()
    )
    # argument extraction
    args = arg_template.arg_extraction()

    ml_hw_adder      = FP_Trunc(args)

    ml_hw_adder.gen_implementation()
