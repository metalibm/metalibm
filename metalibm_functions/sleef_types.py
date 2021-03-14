# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/metalibm/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2021 Kalray
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

###############################################################################
# created:          Mar 14th, 2021
# last-modified:    Mar 14th, 2021
#
# author(s): Nicolas Brunie
###############################################################################
# Description: extra types and operation mapping to allow generation of
#              metalibm benchmarks for the sleef library (sleef.org)
###############################################################################
import sollya

from metalibm_core.core.ml_formats import (
    ML_Compound_Format, v4float64, ML_String, ML_Binary64)
from metalibm_core.core.display_utils import DisplayFormat
from metalibm_core.core.target import TargetRegister
from metalibm_core.core.ml_operations import BuildFromComponent, ComponentSelection

import metalibm_core.utility.ml_template as template_module
import metalibm_core.targets.common.vector_backend as vector_backend

from metalibm_core.code_generation.abstract_backend import GenericBackend
from metalibm_core.code_generation.code_constant import C_Code
from metalibm_core.code_generation.generator_utility import (
    TemplateOperatorFormat, type_strict_match)
from metalibm_core.code_generation.complex_generator import DynamicOperator


DISPLAY_SLEEF_VDOUBLE_2 = DisplayFormat(format_string="{{.x=[%a, %a, %a, %a], .y=[%a, %a, %a, %a]}}", pre_process_fct= lambda v: ", ".join(["%s.x[%d]" % (v, i) for i in range(4)] + ["%s.y[%d]" % (v, i) for i in range(4)]))
DISPLAY_SLEEF_DOUBLE_2  = DisplayFormat(format_string="{{.x=%a, .y=%a}}", pre_process_fct= lambda v: "%s.x, %s.y " % (v, v))

# defining a custom types
Sleef_SLEEF_VECTOR_DOUBLE_2 = ML_Compound_Format("Sleef_SLEEF_VECTOR_DOUBLE_2",
                                                 ["x", "y"],
                                                 [v4float64, v4float64],
                                                 None,
                                                 DISPLAY_SLEEF_VDOUBLE_2,
                                                 sollya.error,
                                                 header=["sleef.h"])

Sleef_SLEEF_DOUBLE_2 = ML_Compound_Format("Sleef_SLEEF_DOUBLE_2",
                  ["x", "y"],
                  [ML_Binary64, ML_Binary64],
                  None,
                  DISPLAY_SLEEF_DOUBLE_2,
				  sollya.error,
				  header=["sleef.h"])

# registering it as a parsable type
template_module.precision_map["sleef_svd2"] = Sleef_SLEEF_VECTOR_DOUBLE_2
template_module.precision_map["sleef_sd2"] = Sleef_SLEEF_DOUBLE_2


# extending vector target
vector_backend.VectorBackend.code_generation_table[C_Code][BuildFromComponent][None][lambda optree: True] = {
    type_strict_match(Sleef_SLEEF_VECTOR_DOUBLE_2, v4float64, v4float64):
        TemplateOperatorFormat("((Sleef_SLEEF_VECTOR_DOUBLE_2) {{.x={0}, .y={1}}})", arity=2),
    type_strict_match(Sleef_SLEEF_DOUBLE_2, ML_Binary64, ML_Binary64):
        TemplateOperatorFormat("((Sleef_SLEEF_DOUBLE_2) {{.x={0}, .y={1}}})", arity=2),
}
vector_backend.VectorBackend.code_generation_table[C_Code][ComponentSelection][ComponentSelection.NamedField] = {
    lambda optree: True:  {
        type_strict_match(v4float64, Sleef_SLEEF_VECTOR_DOUBLE_2, ML_String):
            DynamicOperator(lambda op: TemplateOperatorFormat("{0}.%s" % op.get_input(1).value, arity=2)),
        type_strict_match(ML_Binary64, Sleef_SLEEF_DOUBLE_2, ML_String):
            DynamicOperator(lambda op: TemplateOperatorFormat("{0}.%s" % op.get_input(1).value, arity=2)),
    },
}

# Upgrading pre-generated support tables
vector_target = vector_backend.VectorBackend.get_target_instance() # template_module.target_map["vector"]
vector_target.simplified_rec_op_map[C_Code] = vector_target.generate_supported_op_map(language=C_Code)

# adding callbacks to manage types I/Os
