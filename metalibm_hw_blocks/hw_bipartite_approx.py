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
# Created:          Feb 24th, 2018
# Last-modified:    Mar  7th, 2018
# Author(s): Nicolas Brunie <nbrunie@kalray.eu>
###############################################################################

import sys
import random
import inspect

import sollya

from sollya import Interval, ceil, floor, round
S2 = sollya.SollyaObject(2)
from sollya import parse as sollya_parse

from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.code_generation.vhdl_backend import VHDLBackend
from metalibm_core.core.ml_entity import (
    ML_Entity, ML_EntityBasis, DefaultEntityArgTemplate
)
from metalibm_core.core.ml_function import (
    ML_FunctionBasis
)
from metalibm_core.utility.ml_template import (
    DefaultArgTemplate
)

from metalibm_core.targets.common.fixed_point_backend import FixedPointBackend

from metalibm_core.core.random_gen import FixedPointRandomGen
from metalibm_core.core.advanced_operations import (
    FixedPointPosition
)
from metalibm_core.core.special_values import (
    FP_SpecialValue,
    is_nan,
    is_plus_infty, is_minus_infty, is_sv_omega,
    is_plus_zero, is_minus_zero,
    FP_QNaN, FP_PlusInfty,
)

from metalibm_core.utility.ml_template import (
    ML_EntityArgTemplate
)
from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import *

from metalibm_core.core.ml_hdl_format import *
from metalibm_core.core.ml_hdl_operations import *

from metalibm_core.utility.rtl_debug_utils import (
    debug_fixed, debug_dec, debug_std, debug_hex
)


from metalibm_core.opt.opt_utils import evaluate_range
from metalibm_core.opt.p_rtl_legalize import generate_bitfield_extraction

from metalibm_core.core.generic_bipartite_approx import GenericBipartiteApprox

# Rounding mode is one of:
# 00: Rounding to nearest, ties to Even
# 01: Rounding Up   (towards +infinity)
# 10: Rounding Down (towards -infinity)
# 11: Rounding Towards Zero
rnd_mode_format = ML_StdLogicVectorFormat(2)
rnd_rne = Constant(0, precision=rnd_mode_format, tag="rnd_rne")
rnd_ru  = Constant(1, precision=rnd_mode_format, tag="rnd_ru")
rnd_rd  = Constant(2, precision=rnd_mode_format, tag="rnd_rd")
rnd_rz  = Constant(3, precision=rnd_mode_format, tag="rnd_rz")
# because this module is imported before any scheme is generated, we
# need to make sure extra attributes required for pipelining (namely
#  init_stage and init_op) are properly initialized to a default value
for cst in [rnd_rne, rnd_ru, rnd_rd, rnd_rz]:
    cst.set_attributes(init_stage=None, init_op=None)

def get_virtual_cst(prec, value, language):
    """ Generate coding of constant <value> in language assuming
        format <prec> """
    return prec.get_support_format().get_cst(
        prec.get_base_format().get_integer_coding(value, language))

def rtl_get_fixed_slice(
        optree, hi, lo,
        align_hi=FixedPointPosition.FromLSBToLSB,
        align_lo=FixedPointPosition.FromLSBToLSB,
        **optree_args):
    """ return the slice of the fixed-point argument optree
        contained between bit-index hi and lo """
    return SubSignalSelection(
            optree,
            FixedPointPosition(
                optree,
                lo,
                align=align_lo,
                tag="lo position"
            ),
            FixedPointPosition(
                optree,
                hi,
                align=align_hi,
                tag="hi position"
            ),
            precision=None,
            **optree_args
        )
def sw_get_fixed_slice(
        optree, hi, lo,
        align_hi=FixedPointPosition.FromLSBToLSB,
        align_lo=FixedPointPosition.FromLSBToLSB,
        **optree_args):
    """ return the slice of the fixed-point argument optree
        contained between bit-index hi and lo """
    assert align_lo in [FixedPointPosition.FromLSBToLSB, FixedPointPosition.FromMSBToLSB]
    assert align_hi in [FixedPointPosition.FromLSBToLSB, FixedPointPosition.FromMSBToLSB]
    optree_format = optree.get_precision()
    optree_size = optree_format.get_integer_size() + optree_format.get_frac_size()
    # FIXME: does not take into accoun fixed-point value alignment
    #        within support format
    real_lo = lo if align_lo == FixedPointPosition.FromLSBToLSB else optree_size - 1 - lo  
    real_hi = hi if align_hi == FixedPointPosition.FromLSBToLSB else optree_size - 1 - hi
    result = generate_bitfield_extraction(ML_Custom_FixedPoint_Format(real_hi - real_lo + 1, 0, signed=False), optree, real_lo, real_hi)
    result.set_attributes(**optree_args)
    return result
    # shift = real_lo
    # mask_size = real_hi-real_lo + 1
    # print("mask_size={}, shift={}".format(mask_size, shift))

    # return BitLogicAnd(
    #     BitLogicRightShift(optree, Constant(shift, precision=ML_Int32)),
    #     Constant((2**mask_size - 1) * 2**-optree_format.get_frac_size(), precision=optree_format)
    #     , **optree_args)

def get_fixed_type_from_interval(interval, precision):
    """ generate a fixed-point format which can encode
        @p interval without overflow, and which spans
        @p precision bits """
    lo = inf(interval)
    hi = sup(interval)
    signed = True if lo < 0 else False
    msb_index = int(floor(sollya.log2(max(abs(lo), abs(hi))))) + 1
    extra_digit = 1 if signed else 0
    return fixed_point(msb_index + extra_digit, -(msb_index - precision), signed=signed)


class BipartiteApprox(ML_Entity("bipartite_approx"), GenericBipartiteApprox):
    def __init__(self,
             arg_template = DefaultEntityArgTemplate,
             ):

        # initializing base class
        ML_EntityBasis.__init__(self,
            arg_template = arg_template
        )
        self.pipelined = arg_template.pipelined
        # function to be approximated
        self.function = arg_template.function
        # interval on which the approximation must be valid
        self.interval = arg_template.interval

        self.disable_sub_testing = arg_template.disable_sub_testing
        self.disable_sv_testing = arg_template.disable_sv_testing

        self.alpha = arg_template.alpha
        self.beta = arg_template.beta
        self.gamma = arg_template.gamma
        self.guard_bits = arg_template.guard_bits

    def fixed_point_format_builder(self, int_size, frac_size, signed=True):
        return fixed_point(int_size, frac_size, signed=signed)

    def concatenate_values(self, *args, **kwargs):
        return Concatenation(
            *args,
            **kwargs
        )

    def get_debug_utils(self):
        """ return debug_fixed, debug_std """
        return debug_fixed, debug_std

    ## default argument template generation
    @staticmethod
    def get_default_args(**kw):
        """ generate default argument structure for BipartiteApprox """
        default_dict = {
            "target": VHDLBackend(),
            "output_file": "my_bipartite_approx.vhd",
            "entity_name": "my_bipartie_approx",
            "language": VHDL_Code,
            "function": lambda x: 1.0 / x,
            "interval": Interval(1, 2),
            "pipelined": False,
            "precision": fixed_point(1, 15, signed=False),
            "disable_sub_testing": False,
            "disable_sv_testing": False,
            "alpha": 6,
            "beta": 5,
            "gamma": 5,
            "guard_bits": 3,
            "passes": ["beforepipelining:size_datapath", "beforepipelining:rtl_legalize", "beforepipelining:unify_pipeline_stages"],
        }
        default_dict.update(kw)
        return DefaultEntityArgTemplate(
            **default_dict
        )

    def get_fixed_slice(
            self,
            optree, hi, lo,
            align_hi=FixedPointPosition.FromLSBToLSB,
            align_lo=FixedPointPosition.FromLSBToLSB,
            **optree_args):
        return rtl_get_fixed_slice(optree, hi, lo, align_hi, align_lo, **optree_args)

    def generate_scheme(self):
        ## convert @p value from an input floating-point precision
        #  @p in_precision to an output support format @p out_precision
        io_precision = self.precision

        # declaring main input variable
        vx = self.implementation.add_input_signal("x", io_precision)
        # rounding mode input
        rnd_mode = self.implementation.add_input_signal("rnd_mode", rnd_mode_format)

        vr_out = self.generate_bipartite_approx_module(vx )

        self.implementation.add_output_signal("vr_out", vr_out)


        return [self.implementation]

    def init_test_generator(self):
        """ Initialize test case generator """
        self.input_generator = FixedPointRandomGen(
            int_size=self.precision.get_integer_size(),
            frac_size=self.precision.get_frac_size(),
            signed=self.precision.signed
        )

    def generate_test_case(self, input_signals, io_map, index, test_range = None):
        """ specific test case generation for K1C TCA BLAU """
        rnd_mode = 2 # random.randrange(4)

        hi = sup(self.auto_test_range)
        lo = inf(self.auto_test_range)
        nb_step = int((hi - lo) * S2**self.precision.get_frac_size())
        x_value = lo + (hi - lo) * random.randrange(nb_step) / nb_step
        # self.input_generator.get_new_value()

        input_values = {
            "rnd_mode": rnd_mode,
            "x": x_value,
        }
        return input_values

    def numeric_emulate(self, io_map):
        vx = io_map["x"]
        rnd_mode_i = io_map["rnd_mode"]
        rnd_mode = {
            0: sollya.RN,
            1: sollya.RU,
            2: sollya.RD,
            3: sollya.RZ
        }[rnd_mode_i]
        result = {}
        result["vr_out"] = sollya.round(
            self.function(vx),
            self.precision.get_frac_size(),
            rnd_mode
        )
        return result


    #standard_test_cases = [({"x": 1.0, "y": (S2**-11 + S2**-17)}, None)]
    standard_test_cases = [
        ({"x": 1.0, "rnd_mode": 0}, None),
        ({"x": 1.5, "rnd_mode": 0}, None),
    ]



class SoftBipartiteApprox(ML_FunctionBasis, GenericBipartiteApprox):
    function_name = "bipartite_approx"
    def __init__(self,
             args = DefaultArgTemplate,
             ):

        # initializing base class
        ML_FunctionBasis.__init__(self,
            args=args
        )
        # function to be approximated
        self.function = args.function
        # interval on which the approximation must be valid
        self.interval = args.interval

        self.disable_sub_testing = args.disable_sub_testing
        self.disable_sv_testing = args.disable_sv_testing

        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.guard_bits = args.guard_bits

    def get_debug_utils(self):
        """ return debug_fixed, debug_std """
        return debug_multi, debug_multi

    ## default argument template generation
    @staticmethod
    def get_default_args(**kw):
        """ generate default argument structure for BipartiteApprox """
        default_dict = {
            "target": FixedPointBackend(),
            "output_file": "my_bipartite_approx.c",
            "entity_name": "my_bipartie_approx",
            "language": C_Code,
            "function": lambda x: 1.0 / x,
            "interval": Interval(1, 2),
            "pipelined": False,
            "precision": fixed_point(1, 15, signed=False),
            "disable_sub_testing": False,
            "disable_sv_testing": False,
            "alpha": 6,
            "beta": 5,
            "gamma": 5,
            "guard_bits": 3,
            "passes": ["start:size_datapath"],
        }
        default_dict.update(kw)
        return DefaultArgTemplate(
            **default_dict
        )

    def get_fixed_slice(
            self,
            optree, hi, lo,
            align_hi=FixedPointPosition.FromLSBToLSB,
            align_lo=FixedPointPosition.FromLSBToLSB,
            **optree_args):
        return sw_get_fixed_slice(optree, hi, lo, align_hi, align_lo, **optree_args)

    def generate_scheme(self):
        ## convert @p value from an input floating-point precision
        #  @p in_precision to an output support format @p out_precision
        io_precision = self.precision

        # declaring main input variable
        vx = self.implementation.add_input_variable("x", io_precision)

        vr_out = self.generate_bipartite_approx_module(vx)

        scheme = Statement(
            Return(vr_out)
        )

        return scheme

    def init_test_generator(self):
        """ Initialize test case generator """
        self.input_generator = FixedPointRandomGen(
            int_size=self.precision.get_integer_size(),
            frac_size=self.precision.get_frac_size(),
            signed=self.precision.signed
        )

    def generate_test_case(self, input_signals, io_map, index, test_range = None):
        """ specific test case generation for K1C TCA BLAU """
        rnd_mode = 2 # random.randrange(4)

        hi = sup(self.auto_test_range)
        lo = inf(self.auto_test_range)
        nb_step = int((hi - lo) * S2**self.precision.get_frac_size())
        x_value = lo + (hi - lo) * random.randrange(nb_step) / nb_step
        # self.input_generator.get_new_value()

        input_values = {
            "rnd_mode": rnd_mode,
            "x": x_value,
        }
        return input_values

    def numeric_emulate(self, io_map):
        vx = io_map["x"]
        rnd_mode_i = io_map["rnd_mode"]
        rnd_mode = {
            0: sollya.RN,
            1: sollya.RU,
            2: sollya.RD,
            3: sollya.RZ
        }[rnd_mode_i]
        result = {}
        result["vr_out"] = sollya.round(
            self.function(vx),
            self.precision.get_frac_size(),
            rnd_mode
        )
        print("numeric_emulate, ", io_map, result)
        return result


    #standard_test_cases = [({"x": 1.0, "y": (S2**-11 + S2**-17)}, None)]
    standard_test_cases = [
        ({"x": 1.0, "rnd_mode": 0}, None),
        ({"x": 1.5, "rnd_mode": 0}, None),
    ]


def global_eval(code):
    # FIXME: expose the full global context
    return eval(code, globals())

if __name__ == "__main__":
    default_arg = BipartiteApprox.get_default_args()
    # auto-test
    arg_template = ML_EntityArgTemplate(
        default_entity_name="new_bipartite_approx",
        default_output_file="bipartite_approx.vhd",
        default_arg=default_arg )
    arg_template.parser.add_argument(
        "--function", dest="function", action="store",
        type=global_eval,
        default=(lambda x: 1/x),
        help="function to be approximated")
    arg_template.parser.add_argument(
        "--interval", dest="interval", action="store",
        default=Interval(1, 2),
        type=global_eval,
        help="approximation interval")
    arg_template.parser.add_argument(
        "--disable-sub-test", dest="disable_sub_testing", action="store_const",
        const=True, default=False,
        help="disabling generation of subnormal input during testing")
    arg_template.parser.add_argument(
        "--disable-sv-test", dest="disable_sv_testing", action="store_const",
        const=True, default=False,
        help="disabling generation of special values input during testing")
    arg_template.parser.add_argument(
        "--alpha", dest="alpha", action="store", type=int,
        default=default_arg.alpha, help="size of most significant tabel index (used for linear slope tabulation)")
    arg_template.parser.add_argument(
        "--beta", dest="beta", action="store", type=int,
        default=default_arg.beta, help="size of medium significant index (used for initial value table index LSB)")
    arg_template.parser.add_argument(
        "--gamma", dest="gamma", action="store", type=int,
        default=default_arg.gamma, help="size of least significant table index (used for linear offset tabulation)")
    arg_template.parser.add_argument(
        "--guard-bits", dest="guard_bits", action="store", type=int,
        default=default_arg.guard_bits, help="")


    # argument extraction
    args = parse_arg_index_list = arg_template.arg_extraction()

    ml_hw_div      = BipartiteApprox(args)

    ml_hw_div.gen_implementation()
