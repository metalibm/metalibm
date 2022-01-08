# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/metalibm/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2022 Nicolas Brunie
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
# created:              Jan  8th, 2022
# last-modified:        Jan  8th, 2022
#
# author(s):    Nicolas Brunie (metalibm POINT dev AT gmail com)
# desciprition: Approximation Table for RISC-V Vector backend
###############################################################################
# RISC-V approximation table(s) constructed from
# https://github.com/riscv/riscv-v-spec/blob/master/vfrec7.adoc

from metalibm_core.code_generation.generator_utility import type_strict_match
from metalibm_core.core.ml_operations import ReciprocalSeed
import sollya

from metalibm_core.core.ml_table import ML_ApproxTable
from metalibm_core.core.ml_formats import ML_Binary32


rvv_vfrec7_table = ML_ApproxTable(
    dimensions = [2**7], 
    index_size=7,
    storage_precision = ML_Binary32,
    init_data = [ 1.0 + v / sollya.SollyaObject(128) for v in 
            [127, 125, 123, 121, 119, 117, 116, 114,
             112, 110, 109, 107, 105, 104, 102, 100,
             99, 97, 96, 94, 93, 91, 90, 88, 87, 85,
             84, 83, 81, 80, 79, 77, 76, 75, 74, 72,
             71, 70, 69, 68, 66, 65, 64, 63, 62, 61,
             60, 59, 58, 57, 56, 55, 54, 53, 52, 51,
             50, 49, 48, 47, 46, 45, 44, 43, 42, 41,
             40, 40, 39, 38, 37, 36, 35, 35, 34, 33,
             32, 31, 31, 30, 29, 28, 28, 27, 26, 25,
             25, 24, 23, 23, 22, 21, 21, 20, 19, 19,
             18, 17, 17, 16, 15, 15, 14, 14, 13, 12,
             12, 11, 11, 10, 9, 9, 8, 8, 7, 7, 6, 5,
             5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 
            ]
    ]
)


rvv_approx_table_map = {
    None: { # language
        ReciprocalSeed: {
            None: {
                lambda optree: True: {
                    type_strict_match(ML_Binary32, ML_Binary32): rvv_vfrec7_table,
                },
            },
        },
    },
}
