# -*- coding: utf-8 -*-

import commands
import argparse


import metalibm_functions.ml_log10
import metalibm_functions.ml_log1p
import metalibm_functions.ml_log2
import metalibm_functions.ml_log

# old scheme
old_scheme_function_list = [
  metalibm_functions.ml_log1p.ML_Log1p,
  metalibm_functions.ml_log.ML_Log,
]

# new scheme (ML_Function)
new_scheme_function_list = [
  metalibm_functions.ml_log2.ML_Log2,
  metalibm_functions.ml_log10.ML_Log10,
]

def old_scheme_test(function_ctor, options = []):
  function_name = function_ctor.get_name()
  try: 
    fct = function_ctor()
  except:
    return False
  return True

def new_scheme_test(function_ctor, options = []):
  function_name = function_ctor.get_name()
  fct = function_ctor()
  fct.gen_implementation()
  #try: 
  #except:
  #  return False
  return True


test_list = [(function, old_scheme_test) for function in old_scheme_function_list]
test_list += [(function, new_scheme_test) for function in new_scheme_function_list]




success = True
result_map = {}

for function, test_function in test_list:
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
