# -*- coding: utf-8 -*-

import commands


import metalibm_functions.ml_log10
import metalibm_functions.ml_log1p
import metalibm_functions.ml_log2
import metalibm_functions.ml_log

function_list = [
  metalibm_functions.ml_log2.ML_Log2,
  metalibm_functions.ml_log10.ML_Log10,
  metalibm_functions.ml_log1p.ML_Log1p,
  metalibm_functions.ml_log.ML_Log,
]

def test_function(function_ctor, options = []):
  function_name = function_ctor.get_name()
  try: 
    fct = function_ctor()
  except:
    return False
  return True



success = True
result_map = {}
for function in function_list:
  test_result = test_function(function)
  result_map[function] = test_result 
  success &= test_result

for function in result_map:
  function_name = function.get_name()
  if not result_map[function]:
    print "%s \033[31;1m FAILED \033[0;m" % function_name
  else:
    print "%s SUCCESS" % function_name

if success:
  print "OVERALL SUCCESS"
else:
  print "OVERALL FAILURE"
