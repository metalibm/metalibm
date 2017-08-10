# -*- coding: utf-8 -*-
# optimization pass to promote a scalar/vector DAG into AVX registers

from metalibm_core.targets.intel.x86_processor import *

from metalibm_core.utility.log_report import Log

from metalibm_core.core.ml_formats import *
from metalibm_core.core.passes import OptreeOptimization, Pass, LOG_PASS_INFO

from metalibm_core.opt.p_vector_promotion import Pass_Vector_Promotion


## _m256 register promotion
class Pass_M256_Promotion(Pass_Vector_Promotion):
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
  }

  def get_translation_table(self):
    return self.trans_table

  def __init__(self, target):
    Pass_Vector_Promotion.__init__(self, target)
    self.set_descriptor("AVX promotion pass")



Log.report(LOG_PASS_INFO, "Registering m256_conversion pass")
# register pass
Pass.register(Pass_M256_Promotion)
