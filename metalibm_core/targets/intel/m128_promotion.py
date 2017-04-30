# -*- coding: utf-8 -*-
# optimization pass to promote a scalar/vector DAG into SSE/AVX  registers

from metalibm_core.targets.intel.x86_processor import *

from metalibm_core.core.ml_formats import *
from metalibm_core.core.passes import OptreeOptimization, Pass

from metalibm_core.opt.p_vector_promotion import Pass_Vector_Promotion


## _m128 register promotion
class Pass_M128_Promotion(Pass_Vector_Promotion):
  pass_tag = "m128_promotion"
  ## Translation table between standard formats
  #  and __m128-based register formats
  trans_table = {
    ML_Binary32: ML_SSE_m128_v1float32,
    ML_Binary64: ML_SSE_m128_v1float64,
    v2float64:   ML_SSE_m128_v2float64,
    v4float32:   ML_SSE_m128_v4float32,
    v4int32:     ML_SSE_m128_v4int32,
    ML_Int32:    ML_SSE_m128_v1int32,
  }

  def get_translation_table(self):
    return self.trans_table

  def __init__(self, target):
    Pass_Vector_Promotion.__init__(self, target)
    self.set_descriptor("SSE/AVX promotion pass")



print "Registering m128_conversion pass"
# register pass
Pass.register(Pass_M128_Promotion)
