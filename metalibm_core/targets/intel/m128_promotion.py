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
#            Hugues de Lassus <hugues.de-lassus@univ-perp.fr>,
# Description: optimization pass to promote a scalar/vector DAG
#              into SSE registers
###############################################################################


from metalibm_core.targets.intel.x86_processor import *

from metalibm_core.utility.log_report import Log

from metalibm_core.core.ml_formats import *
from metalibm_core.core.passes import OptreeOptimization, Pass, LOG_PASS_INFO

from metalibm_core.opt.p_vector_promotion import Pass_Vector_Promotion


## _m128 register promotion
class Pass_M128_Promotion(Pass_Vector_Promotion):
  """ Vector register promotion from generic vector formats
      to x86's m128 based vector formats """
  pass_tag = "m128_promotion"
  ## Translation table between standard formats
  #  and __m128-based register formats
  trans_table = {
    ML_Binary32: ML_SSE_m128_v1float32,
    ML_Binary64: ML_SSE_m128_v1float64,
    v2float64:   ML_SSE_m128_v2float64,
    v4float32:   ML_SSE_m128_v4float32,
    v4bool:      ML_SSE_m128_v4bool,
    v2lbool:     ML_SSE_m128_v2lbool,
    v2int64:     ML_SSE_m128_v2int64,
    v4int32:     ML_SSE_m128_v4int32,
    v2uint64:    ML_SSE_m128_v2uint64,
    v4uint32:    ML_SSE_m128_v4uint32,
    ML_Int32:    ML_SSE_m128_v1int32,
  }

  def get_translation_table(self):
    return self.trans_table

  def __init__(self, target):
    Pass_Vector_Promotion.__init__(self, target)
    self.set_descriptor("SSE promotion pass")



Log.report(LOG_PASS_INFO, "Registering m128_conversion pass")
# register pass
Pass.register(Pass_M128_Promotion)
