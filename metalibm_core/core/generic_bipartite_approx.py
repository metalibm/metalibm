# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/metalibm/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2025 Nicolas Brunie
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
# Created:          Oct 14th, 2025
# Last-modified:    Oct 14th, 2025
# Author(s): Nicolas Brunie
###############################################################################

import sollya

from sollya import Interval 
S2 = sollya.SollyaObject(2)

from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.core.advanced_operations import FixedPointPosition

from metalibm_core.utility.log_report  import Log
from metalibm_core.utility.debug_utils import *

from metalibm_core.opt.opt_utils import evaluate_range


class GenericBipartiteApprox:
    """ generic bipartite approximation class """

    def fixed_point_format_builder(self, int_size, frac_size, signed=True):
        raise NotImplementedError

    def get_fixed_type_from_interval(self, interval, precision):
        """ generate a fixed-point format which can encode
            @p interval without overflow, and which spans
            @p precision bits """
        lo = inf(interval)
        hi = sup(interval)
        signed = True if lo < 0 else False
        msb_index = int(sollya.floor(sollya.log2(max(abs(lo), abs(hi))))) + 1
        extra_digit = 1 if signed else 0
        return self.fixed_point_format_builder(msb_index + extra_digit, -(msb_index - precision), signed=signed)

    def concatenate_values(self, *args, **kwargs):
        raise NotImplementedError

    def get_fixed_slice(
            self,
            optree, hi, lo,
            align_hi=FixedPointPosition.FromLSBToLSB,
            align_lo=FixedPointPosition.FromLSBToLSB,
            **optree_args):
        raise NotImplementedError

    def get_debug_utils(self):
        """ return debug_fixed, debug_std """
        raise NotImplementedError

    def generate_bipartite_approx_module(self, vx):
        """
            vx input value
        """
        debug_fixed, debug_std = self.get_debug_utils()
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
        print(f"range_lo={range_lo}, range_hi={range_hi}, range_size={range_size}, range_size_log2={range_size_log2}")
        # assert 2**range_size_log2 == range_size

        reduced_x = Conversion(
            BitLogicRightShift(
                vx - range_lo,
                range_size_log2
            ),
            precision=self.fixed_point_format_builder(0,alpha+beta+gamma,signed=False),
            tag="reduced_x",
            debug=debug_fixed
        )


        alpha_index = self.get_fixed_slice(
            reduced_x, 0, alpha-1,
            align_hi=FixedPointPosition.FromMSBToLSB,
            align_lo=FixedPointPosition.FromMSBToLSB,
            tag="alpha_index",
            debug=debug_std
        )
        gamma_index = self.get_fixed_slice(
            reduced_x, gamma-1, 0,
            align_hi=FixedPointPosition.FromLSBToLSB,
            align_lo=FixedPointPosition.FromLSBToLSB,
            tag="gamma_index",
            debug=debug_std
        )

        beta_index = self.get_fixed_slice(
            reduced_x, alpha, gamma,
            align_hi=FixedPointPosition.FromMSBToLSB,
            align_lo=FixedPointPosition.FromLSBToLSB,
            tag="beta_index",
            debug=debug_std
        )

        # Assuming monotonic function
        f_absmax = max(abs(f_hi), abs(f_lo))
        f_absmin = min(abs(f_hi), abs(f_lo))

        print(f"f_lo={float(f_lo)}, f_hi={float(f_hi)}, f_absmin={float(f_absmin)}, f_absmax={float(f_absmax)}")
        f_msb = int(sollya.ceil(sollya.log2(f_absmax))) + 1
        f_lsb = int(sollya.floor(sollya.log2(f_absmin)))
        storage_lsb = f_lsb - self.precision.get_bit_size() - guard_bits

        f_int_size = f_msb
        f_frac_size = -storage_lsb

        storage_format = self.fixed_point_format_builder(
            f_int_size, f_frac_size, signed=False
        )
        Log.report(Log.Info, "storage_format is {}".format(storage_format))

        # table of initial value index
        tiv_index = self.concatenate_values(
            alpha_index, beta_index,
            tag = "tiv_index",
            debug=debug_std
        )
        # table of offset value index
        to_index = self.concatenate_values(
            alpha_index, gamma_index,
            tag = "to_index",
            debug=debug_std
        )

        tiv_index_size = int(alpha + beta)
        to_index_size = int(alpha + gamma)

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
        #tiv_min = table_iv[0]
        #tiv_max = table_iv[0]
        #for i in range(1, 2**tiv_index_size):
        #    tiv_min = min(tiv_min, table_iv[i])
        #    tiv_max = max(tiv_max, table_iv[i])
        tiv_min = min(table_iv)
        tiv_max = max(table_iv)
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

        offset_precision = self.get_fixed_type_from_interval(
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

        result = Conversion(
            initial_value + offset_value,
            precision=self.precision,
            tag="vr_out",
            debug=debug_fixed
        )
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
                    local_error = abs(self.function(table_x) - approx_value)
                    approx_error = max(approx_error, local_error)
        error_log2 = float(sollya.log2(approx_error))
        Log.report(Log.Verbose, "approx_error is {}, error_log2 is {}".format(float(approx_error), error_log2))

        # table size
        table_iv_size = 2**(alpha+beta)
        table_offset_size = 2**(alpha+gamma)
        Log.report(Log.Verbose, "tables' size are {} entries".format(table_iv_size + table_offset_size)) 
        return result
