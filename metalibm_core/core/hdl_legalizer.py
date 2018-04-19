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
# last-modified:    Apr 19th, 2018
# Author(s): Nicolas Brunie <nbrunie@kalray.eu>
###############################################################################

from metalibm_core.core.ml_operations import (
    Select, Comparison, ExponentExtraction, MantissaExtraction,
    Constant, TypeCast
)
from metalibm_core.core.ml_hdl_operations import (
    SubSignalSelection, Concatenation
)
from metalibm_core.core.ml_hdl_format import (
    ML_StdLogicVectorFormat, ML_StdLogic
)

# Optree generation function for MantissaExtraction
def mantissa_extraction_modifier_from_fields(op, field_op, exp_is_zero, tag="mant_extr"):
    """ Legalizing a MantissaExtraction node into a sub-graph
        of basic operation, assuming <field_op> bitfield and <exp_is_zero> flag
        are already available """

    op_precision = op.get_precision().get_base_format()

    implicit_digit = Select(
        exp_is_zero,
        Constant(0, precision=ML_StdLogic),
        Constant(1, precision=ML_StdLogic),
        precision=ML_StdLogic,
        tag=tag+"_implicit_digit",
    )
    result = Concatenation(
        implicit_digit,
        TypeCast(
            field_op,
            precision=ML_StdLogicVectorFormat(op_precision.get_field_size())
        ),
        precision=ML_StdLogicVectorFormat(op_precision.get_mantissa_size()),
    )
    return result

def mantissa_extraction_modifier(optree):
    """ Legalizing a MantissaExtraction node into a sub-graph
        of basic operation """
    init_stage = optree.attributes.get_dyn_attribute("init_stage")
    op = optree.get_input(0)
    tag=optree.get_tag() or "mant_extr"

    op_precision = op.get_precision().get_base_format()
    exp_prec = ML_StdLogicVectorFormat(op_precision.get_exponent_size())
    field_prec = ML_StdLogicVectorFormat(op_precision.get_field_size())

    exp_op = ExponentExtraction(
        op, precision=exp_prec, init_stage=init_stage,
        tag=tag + "_exp_extr"
    )
    field_op = SubSignalSelection(
        TypeCast(
            op,
            precision=op.get_precision().get_support_format(),
            init_stage=init_stage,
            tag=tag + "_field_cast"
        ), 0, op_precision.get_field_size() - 1, precision=field_prec,
        init_stage=init_stage,
        tag=tag + "_field"

    )
    exp_is_zero = Comparison(
        exp_op,
        Constant(
            op_precision.get_zero_exponent_value(),
            precision=exp_prec,
            init_stage=init_stage
        ),
        precision=ML_Bool,
        specifier=Comparison.Equal,
        init_stage=init_stage
    )

    result =  mantissa_extraction_modifier_from_fields(op, field_op, exp_is_zero)
    forward_attributes(optree, result)
    return result

