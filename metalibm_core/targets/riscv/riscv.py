# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/metalibm/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2021 Nicolas Brunie
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
# created:          Sep  5th, 2021
# last-modified:    Sep  5th, 2021
#
# Author(s):        Nicolas Brunie <metalibmdev@gmail.com>
###############################################################################

from metalibm_core.core.target import UniqueTargetDecorator

from metalibm_core.code_generation.abstract_backend import LOG_BACKEND_INIT
from metalibm_core.code_generation.code_constant import C_Code

from metalibm_core.utility.log_report import Log

from metalibm_core.targets.common.vector_backend import VectorBackend



@UniqueTargetDecorator
class RISCV_RV64(VectorBackend):
    target_name = "rv64g"

    code_generation_table = {
        C_Code: {},
    }

    def __init__(self):
        super().__init__()

# debug message
Log.report(LOG_BACKEND_INIT, "initializing RISC-V targets")
