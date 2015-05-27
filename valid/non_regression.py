# -*- coding: utf-8 -*-

import commands

def test_function(script_name, options = []):
  status, out = commands.getstatusoutput("python metalibm_functions/%s.py %s" % (script_name, " ".join(options)))
  if status != 0:
    print out
    print "%s \033[31;1m FAILED \033[0;m" % script_name
  else:
    print "%s SUCCESS" % script_name
  return (status == 0)

function_list = [
  "ml_log",
  "ml_log2",
  "ml_log1p",
  "ml_log10",
]

success = True
for function in function_list:
  success &= test_function(function)

if success:
  print "OVERALL SUCCESS"
else:
  print "OVERALL FAILURE"
