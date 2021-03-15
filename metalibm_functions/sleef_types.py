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
    ML_Compound_Format, ML_CompoundVectorFormat, ML_VectorFormat,
    v4float64, v8float32,
    ML_String, ML_Binary64, ML_Binary32, VECTOR_TYPE_MAP)
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

DISPLAY_SLEEF_VFLOAT_2 = DisplayFormat(format_string="{{.x=[%a, %a, %a, %a, %a, %a, %a, %a], .y=[%a, %a, %a, %a, %a, %a, %a, %a]}}", pre_process_fct= lambda v: ", ".join(["%s.x[%d]" % (v, i) for i in range(8)] + ["%s.y[%d]" % (v, i) for i in range(8)]))
DISPLAY_SLEEF_FLOAT_2  = DisplayFormat(format_string="{{.x=%a, .y=%a}}", pre_process_fct= lambda v: "%s.x, %s.y " % (v, v))

DISPLAY_SLEEF_VDOUBLE_2 = DisplayFormat(format_string="{{.x=[%a, %a, %a, %a], .y=[%a, %a, %a, %a]}}", pre_process_fct= lambda v: ", ".join(["%s.x[%d]" % (v, i) for i in range(4)] + ["%s.y[%d]" % (v, i) for i in range(4)]))
DISPLAY_SLEEF_DOUBLE_2  = DisplayFormat(format_string="{{.x=%a, .y=%a}}", pre_process_fct= lambda v: "%s.x, %s.y " % (v, v))

class SleefCompoundVectorFormat(ML_CompoundVectorFormat):
    """ specific Compound(struct) format class for Sleef vector compounded
        vector types """
    def __init__(self, c_format_name, vector_size,
                   field_list, field_format_list,
                   scalar_format, sollya_precision=None,
                   cst_callback=None,
                   display_format="",
                   header=None):
        ML_VectorFormat.__init__(self, scalar_format, vector_size, c_format_name, header=header)
        # header must be re-submitted as argument to avoid being
        # over written by this new constructor call
        ML_Compound_Format.__init__(self, c_format_name, field_list, field_format_list, "", display_format, sollya_precision, header=header)
        self.cst_callback = cst_callback
        self.limb_num = 2

    def get_limb_precision(self, index):
        return self.field_format_list[index]

    def get_cst_default(self, cst_value, language = C_Code):
        elt_value_list = [self.scalar_format.get_cst(cst_value[i], language = language) for i in range(self.vector_size)]
        field_str_list = [".%s" % field_name for field_name in self.scalar_format.c_field_list]
        cst_value_array = [[None for i in range(self.vector_size)] for j in range(self.limb_num)]
        for lane_id in range(self.vector_size):
            tmp_cst = cst_value[lane_id]
            if tmp_cst != 0:
                Log.report(Log.Error, "only zero cst_value is supported in SleefCompoundVectorFormat.get_cst_default, not {}", tmp_cst)
            assert tmp_cst == 0
            for limb_id in range(self.limb_num):
                field_format = self.field_format_list[limb_id].get_scalar_format()
                field_value = sollya.round(tmp_cst, field_format.sollya_object, sollya.RN)
                cst_value_array[limb_id][lane_id] = field_value
        if language is C_Code:
            return "{" + ",".join("{} = {}".format(field_str_list[limb_id], self.get_limb_precision(limb_id).get_cst(cst_value_array[limb_id], language=language)) for limb_id in range(self.limb_num)) + "}"
        else:
            Log.report(Log.Error, "unsupported language in ML_MultiPrecision_VectorFormat.get_cst: %s" % (language))

class SleefCompoundFormat(ML_Compound_Format): pass
# defining custom types
Sleef_SLEEF_DOUBLE_2 = SleefCompoundFormat("Sleef_SLEEF_DOUBLE_2",
                  ["x", "y"],
                  [ML_Binary64, ML_Binary64],
                  None,
                  DISPLAY_SLEEF_DOUBLE_2,
				  sollya.binary64,
				  header="sleef.h")
Sleef_SLEEF_FLOAT_2 = SleefCompoundFormat("Sleef_SLEEF_FLOAT_2",
                  ["x", "y"],
                  [ML_Binary32, ML_Binary32],
                  None,
                  DISPLAY_SLEEF_FLOAT_2,
				  sollya.binary32,
				  header="sleef.h")


Sleef_SLEEF_VECTOR_DOUBLE_2 = SleefCompoundVectorFormat("Sleef_SLEEF_VECTOR_DOUBLE_2",
                                                         4,
                                                         ["x", "y"],
                                                         [v4float64, v4float64],
                                                         Sleef_SLEEF_DOUBLE_2,
                                                         sollya_precision=sollya.binary64,
                                                         cst_callback=None,
                                                         display_format=DISPLAY_SLEEF_VDOUBLE_2,
                                                         header="sleef.h")

Sleef_SLEEF_VECTOR_FLOAT_2 = SleefCompoundVectorFormat("Sleef_SLEEF_VECTOR_FLOAT_2",
                                                         8,
                                                         ["x", "y"],
                                                         [v8float32, v8float32],
                                                         Sleef_SLEEF_FLOAT_2,
                                                         sollya_precision=sollya.binary32,
                                                         cst_callback=None,
                                                         display_format=DISPLAY_SLEEF_VFLOAT_2,
                                                         header="sleef.h")


# registering sleef type for vectorisation
VECTOR_TYPE_MAP[Sleef_SLEEF_DOUBLE_2] = {4: Sleef_SLEEF_VECTOR_DOUBLE_2}
VECTOR_TYPE_MAP[Sleef_SLEEF_FLOAT_2] = {8: Sleef_SLEEF_VECTOR_FLOAT_2}

# registering it as a parsable type
template_module.precision_map["sleef_svf2"] = Sleef_SLEEF_VECTOR_FLOAT_2
template_module.precision_map["sleef_sf2"] = Sleef_SLEEF_FLOAT_2
template_module.precision_map["sleef_svd2"] = Sleef_SLEEF_VECTOR_DOUBLE_2
template_module.precision_map["sleef_sd2"] = Sleef_SLEEF_DOUBLE_2


# extending vector target
vector_backend.VectorBackend.code_generation_table[C_Code][BuildFromComponent][None][lambda optree: True] = {
    type_strict_match(Sleef_SLEEF_VECTOR_FLOAT_2, v8float32, v8float32):
        TemplateOperatorFormat("((Sleef_SLEEF_VECTOR_FLOAT_2) {{.x={0}, .y={1}}})", arity=2),
    type_strict_match(Sleef_SLEEF_FLOAT_2, ML_Binary32, ML_Binary32):
        TemplateOperatorFormat("((Sleef_SLEEF_FLOAT_2) {{.x={0}, .y={1}}})", arity=2),
    type_strict_match(Sleef_SLEEF_VECTOR_DOUBLE_2, v4float64, v4float64):
        TemplateOperatorFormat("((Sleef_SLEEF_VECTOR_DOUBLE_2) {{.x={0}, .y={1}}})", arity=2),
    type_strict_match(Sleef_SLEEF_DOUBLE_2, ML_Binary64, ML_Binary64):
        TemplateOperatorFormat("((Sleef_SLEEF_DOUBLE_2) {{.x={0}, .y={1}}})", arity=2),
}
vector_backend.VectorBackend.code_generation_table[C_Code][ComponentSelection][ComponentSelection.NamedField] = {
    lambda optree: True:  {
        type_strict_match(v8float32, Sleef_SLEEF_VECTOR_FLOAT_2, ML_String):
            DynamicOperator(lambda op: TemplateOperatorFormat("{0}.%s" % op.get_input(1).value, arity=2)),
        type_strict_match(ML_Binary32, Sleef_SLEEF_FLOAT_2, ML_String):
            DynamicOperator(lambda op: TemplateOperatorFormat("{0}.%s" % op.get_input(1).value, arity=2)),
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
