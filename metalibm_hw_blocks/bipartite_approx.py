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

def get_fixed_slice(
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

class BipartiteApprox(ML_Entity("bipartite_approx")):
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

    def generate_scheme(self):
        ## convert @p value from an input floating-point precision
        #  @p in_precision to an output support format @p out_precision
        io_precision = self.precision

        # declaring main input variable
        vx = self.implementation.add_input_signal("x", io_precision)
        # rounding mode input
        rnd_mode = self.implementation.add_input_signal("rnd_mode", rnd_mode_format)

        # size of most significant table index (for linear slope tabulation)
        alpha = self.alpha # 6
        # size of medium significant table index (for initial value table index LSB)
        beta = self.beta # 5
        # size of least significant table index (for linear offset tabulation)
        gamma = self.gamma # 5

        guard_bits = self.guard_bits # 3

        vx.set_interval(self.interval)

        range_hi = sollya.sup(self.interval)
        range_lo = sollya.inf(self.interval)
        f_hi = self.function(range_hi)
        f_lo = self.function(range_lo)
        # fixed by format used for reduced_x
        range_size = range_hi - range_lo
        range_size_log2 = int(sollya.log2(range_size))
        assert 2**range_size_log2 == range_size

        reduced_x = Conversion(
            BitLogicRightShift(
                vx - range_lo,
                range_size_log2
            ),
            precision=fixed_point(0,alpha+beta+gamma,signed=False),
            tag="reduced_x",
            debug=debug_fixed
        )


        alpha_index = get_fixed_slice(
            reduced_x, 0, alpha-1,
            align_hi=FixedPointPosition.FromMSBToLSB,
            align_lo=FixedPointPosition.FromMSBToLSB,
            tag="alpha_index",
            debug=debug_std
        )
        gamma_index = get_fixed_slice(
            reduced_x, gamma-1, 0,
            align_hi=FixedPointPosition.FromLSBToLSB,
            align_lo=FixedPointPosition.FromLSBToLSB,
            tag="gamma_index",
            debug=debug_std
        )

        beta_index = get_fixed_slice(
            reduced_x, alpha, gamma,
            align_hi=FixedPointPosition.FromMSBToLSB,
            align_lo=FixedPointPosition.FromLSBToLSB,
            tag="beta_index",
            debug=debug_std
        )

        # Assuming monotonic function
        f_absmax = max(abs(f_hi), abs(f_lo))
        f_absmin = min(abs(f_hi), abs(f_lo))

        f_msb = int(sollya.ceil(sollya.log2(f_absmax))) + 1
        f_lsb = int(sollya.floor(sollya.log2(f_absmin)))
        storage_lsb = f_lsb - io_precision.get_bit_size() - guard_bits

        f_int_size = f_msb
        f_frac_size = -storage_lsb

        storage_format = fixed_point(
            f_int_size, f_frac_size, signed=False
        )
        Log.report(Log.Info, "storage_format is {}".format(storage_format))


        # table of initial value index
        tiv_index = Concatenation(
            alpha_index, beta_index,
            tag = "tiv_index",
            debug=debug_std
        )
        # table of offset value index
        to_index = Concatenation(
            alpha_index, gamma_index,
            tag = "to_index",
            debug=debug_std
        )

        tiv_index_size = alpha + beta
        to_index_size = alpha + gamma

        Log.report(Log.Info, "initial table structures")
        table_iv = ML_NewTable(
            dimensions=[2**tiv_index_size],
            storage_precision=storage_format,
            tag="tiv"
        )
        table_offset = ML_NewTable(
            dimensions=[2**to_index_size],
            storage_precision=storage_format,
            tag="to"
        )

        slope_table = [None] * (2**alpha)
        slope_delta = 1.0 / sollya.SollyaObject(2**alpha)
        delta_u = range_size * slope_delta * 2**-15
        Log.report(Log.Info, "computing slope value")
        for i in range(2**alpha):
            # slope is computed at the middle of range_size interval
            slope_x = range_lo + (i+0.5) * range_size * slope_delta
            # TODO: gross approximation of derivatives
            f_xpu = self.function(slope_x + delta_u / 2)
            f_xmu = self.function(slope_x - delta_u / 2)
            slope = (f_xpu - f_xmu) / delta_u
            slope_table[i] = slope

        range_rcp_steps = 1.0 / sollya.SollyaObject(2**tiv_index_size)
        Log.report(Log.Info, "computing value for initial-value table")
        for i in range(2**tiv_index_size):
            slope_index = i / 2**beta
            iv_x = range_lo + i * range_rcp_steps * range_size
            offset_x = 0.5 * range_rcp_steps * range_size
            # initial value is computed so that the piecewise linear
            # approximation intersects the function at iv_x + offset_x
            iv_y = self.function(iv_x + offset_x) - offset_x * slope_table[int(slope_index)]
            initial_value = storage_format.round_sollya_object(iv_y)
            table_iv[i] = initial_value

        # determining table of initial value interval
        tiv_min = table_iv[0]
        tiv_max = table_iv[0]
        for i in range(1, 2**tiv_index_size):
            tiv_min = min(tiv_min, table_iv[i])
            tiv_max = max(tiv_max, table_iv[i])
        table_iv.set_interval(Interval(tiv_min, tiv_max))


        offset_step = range_size / S2**(alpha+beta+gamma)
        for i in range(2**alpha):
            Log.report(Log.Info, "computing offset value for sub-table {}".format(i))
            for j in range(2**gamma):
                to_i = i * 2**gamma + j
                offset = slope_table[i] * j * offset_step
                table_offset[to_i] = offset


        # determining table of offset interval
        to_min = table_offset[0]
        to_max = table_offset[0]
        for i in range(1, 2**(alpha+gamma)):
            to_min = min(to_min, table_offset[i])
            to_max = max(to_max, table_offset[i])
        offset_interval = Interval(to_min, to_max)
        table_offset.set_interval(offset_interval)

        initial_value = TableLoad(
            table_iv, tiv_index,
            precision=storage_format,
            tag="initial_value",
            debug=debug_fixed
        )

        offset_precision = get_fixed_type_from_interval(
            offset_interval, 16
        )
        Log.report(Log.Verbose, "offset_precision is {} ({} bits)".format(offset_precision, offset_precision.get_bit_size()))
        table_offset.get_precision().storage_precision = offset_precision

        # rounding table value
        for i in range(1, 2**(alpha+gamma)):
            table_offset[i] = offset_precision.round_sollya_object(table_offset[i])

        offset_value = TableLoad(
            table_offset, to_index,
            precision=offset_precision,
            tag="offset_value",
            debug=debug_fixed
        )

        Log.report(Log.Verbose,
            "initial_value's interval: {}, offset_value's interval: {}".format(
            evaluate_range(initial_value),
            evaluate_range(offset_value)
        ))

        final_add = initial_value + offset_value
        round_bit = final_add # + FixedPointPosition(final_add, io_precision.get_bit_size(), align=FixedPointPosition.FromMSBToLSB)

        vr_out = Conversion(
            initial_value + offset_value,
            precision=io_precision,
            tag="vr_out",
            debug=debug_fixed
        )

        self.implementation.add_output_signal("vr_out", vr_out)

        # Approximation error evaluation
        approx_error = 0.0
        for i in range(2**alpha):
            for j in range(2**beta):
                tiv_i = (i * 2**beta + j)
                # = range_lo + tiv_i * range_rcp_steps * range_size
                iv = table_iv[tiv_i]
                for k in range(2**gamma):
                    to_i = i * 2**gamma + k
                    offset = table_offset[to_i] 
                    approx_value = offset + iv
                    table_x = range_lo + range_size * ((i * 2**beta + j) * 2**gamma + k) / S2**(alpha+beta+gamma)
                    local_error = abs(1 / (table_x) - approx_value)
                    approx_error = max(approx_error, local_error) 
        error_log2 = float(sollya.log2(approx_error))
        Log.report(Log.Verbose, "approx_error is {}, error_log2 is {}".format(float(approx_error), error_log2))

        # table size
        table_iv_size = 2**(alpha+beta)
        table_offset_size = 2**(alpha+gamma)
        Log.report(Log.Verbose, "tables' size are {} entries".format(table_iv_size + table_offset_size)) 

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
