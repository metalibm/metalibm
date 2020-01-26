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
#
# Author(s): Nicolas Brunie <nbrunie@kalray.eu>,
#            Hugues de Lassus <hugues.de-lassus@univ-perp.fr>
# Description: optimization pass to promote a scalar/vector DAG into AVX
#              registers
###############################################################################

from metalibm_core.targets.intel.x86_processor import *

from metalibm_core.utility.log_report import Log

from metalibm_core.core.ml_formats import *
from metalibm_core.core.passes import OptreeOptimization, Pass, LOG_PASS_INFO

from metalibm_core.opt.p_vector_promotion import Pass_Vector_Promotion


class Pass_M256_Promotion(Pass_Vector_Promotion):
  """ Vector register promotion from generic vector formats
      to x86's m256 based vector formats """
  pass_tag = "m256_promotion"
  ## Translation table between standard formats
  #  and __m256-based register formats
  trans_table = {
    v4float64:   ML_AVX_m256_v4float64,
    v8float32:   ML_AVX_m256_v8float32,
    v4int64:     ML_AVX_m256_v4int64,
    v8int32:     ML_AVX_m256_v8int32,
    v4uint64:    ML_AVX_m256_v4uint64,
    v8uint32:    ML_AVX_m256_v8uint32,
    v8bool:      ML_AVX_m256_v8bool,
    v4lbool:     ML_AVX_m256_v4lbool,
  }

  def get_translation_table(self):
    return self.trans_table

  def __init__(self, target):
    Pass_Vector_Promotion.__init__(self, target)
    self.set_descriptor("AVX promotion pass")



Log.report(LOG_PASS_INFO, "Registering m256_conversion pass")
# register pass
Pass.register(Pass_M256_Promotion)
