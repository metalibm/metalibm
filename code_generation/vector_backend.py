# -*- coding: utf-8 -*-

###############################################################################
# This file is part of Kalray's Metalibm tool
# Copyright (2016)
# All rights reserved
# created:          Feb 2nd, 2016
# last-modified:    Feb 2nd, 2016
#
# description: implement a vector backend for Metalibm
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from ..utility.log_report import *
from .generator_utility import *
from .complex_generator import *
from ..core.ml_formats import *
from ..core.ml_operations import *
from ..utility.common import Callable
from .generic_processor import GenericProcessor

from metalibm_core.core.target import TargetRegister



vector_c_code_generation_table = {
  Addition: {
    None: {
      lambda True: True: {
      },
    },
  },
  Subtraction: {
    None: {
      lambda True: True: {
      },
    },
  },
  Multiplication: {
    None: {
      lambda True: True: {
      },
    },
  },
}

vector_gappa_code_generation_table = {
}

class VectorBackend(GenericProcessor):
  target_name = "vector"
  TargetRegister.register_new_target(target_name, lambda _: VectorBackend)

  code_generation_table = {
    C_Code: vector_c_code_generation_table, 
  }
