# -*- coding: utf-8 -*-

from metalibm_core.core.ml_function import DefaultArgTemplate
from metalibm_core.core.ml_formats import ML_Binary32

## Runner wrapper for unit tests
class TestRunner:
  def __init__(self):
    pass

  ## Generate default argument template
  #  may be overloaded
  @staticmethod
  def get_default_args(**kw):
    arg_template = DefaultArgTemplate(
      precision = ML_Binary32,
      output_file = "ut_out.c",
      function_name = "ut_test"
    )
    return arg_template

  ## overloading 
  def __call__(self):
    raise NotImplementedError
