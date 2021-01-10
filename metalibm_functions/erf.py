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
#
# created:            Mar   11th, 2016
# last-modified:      Mar    3rd, 2019
#
# description: meta-implementation of error-function erf
#              erf(x) = 2 / pi * integral(0, x, e^(-t^2), dt)
###############################################################################

import sollya
import yaml

from sollya import (
    Interval, ceil, floor, round, inf, sup, log, exp, log1p,
    dirtyinfnorm,
    guessdegree
)

from metalibm_core.core.ml_function import ML_FunctionBasis, DefaultArgTemplate
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.polynomials import Polynomial, PolynomialSchemeEvaluator
from metalibm_core.core.ml_table import ML_NewTable, generic_mantissa_msb_index_fct
from metalibm_core.core.precisions import ML_Faithful
from metalibm_core.core.special_values import (
        FP_QNaN, FP_MinusInfty, FP_PlusInfty, FP_PlusZero
)
from metalibm_core.core.indexing import SubFPIndexing
from metalibm_core.core.approximation import (
    search_bound_threshold, generate_piecewise_poly_approx,
    load_piecewese_poly_params_from_axf,
    generate_piecewise_poly_approx_from_params
)
from metalibm_core.core.simple_scalar_function import ScalarUnaryFunction

from metalibm_core.code_generation.generic_processor import GenericProcessor

from metalibm_core.utility.ml_template import ML_NewArgTemplate
from metalibm_core.utility.debug_utils import debug_multi
from metalibm_core.utility.axf_utils import AXF_JSON_Exporter, AXF_JSON_Importer


# static constant for numerical value 2
S2 = sollya.SollyaObject(2)


class ML_Erf(ScalarUnaryFunction):
    """ Meta implementation of the error-function """
    function_name = "ml_erf"
    def __init__(self, args):
        super().__init__(args)
        self.dump_axf_approx = args.dump_axf_approx
        self.load_axf_approx = args.load_axf_approx

    @staticmethod
    def get_default_args(**kw):
        """ Return a structure containing the arguments for ML_Erf,
            builtin from a default argument mapping overloaded with @p kw """
        default_args_erf = {
                "output_file": "my_erf.c",
                "function_name": "my_erf",
                "precision": ML_Binary32,
                "accuracy": ML_Faithful,
                "load_axf_approx": False,
                "dump_axf_approx": False,
                "target": GenericProcessor.get_target_instance(),
                "passes": [
                    ("start:instantiate_abstract_prec"),
                    ("start:instantiate_prec"),
                    ("start:basic_legalization"),
                    ("start:expand_multi_precision")],
        }
        default_args_erf.update(kw)
        return DefaultArgTemplate(**default_args_erf)

    def generate_scalar_scheme(self, vx):
        abs_vx = Abs(vx, precision=self.precision)

        FCT_LIMIT = 1.0

        one_limit = search_bound_threshold(sollya.erf, FCT_LIMIT, 1.0, 10.0, self.precision)
        one_limit_exp = int(sollya.floor(sollya.log2(one_limit)))
        Log.report(Log.Debug, "erf(x) = 1.0 limit is {}, with exp={}", one_limit, one_limit_exp)

        upper_approx_bound = 10

        # empiral numbers
        eps_exp = {ML_Binary32: -3, ML_Binary64: -5}[self.precision]
        eps = S2**eps_exp

        Log.report(Log.Info, "building mathematical polynomial")
        approx_interval = Interval(0, eps)
        # fonction to approximate is erf(x) / x
        # it is an even function erf(x) / x = erf(-x) / (-x)
        approx_fct = sollya.erf(sollya.x) - (sollya.x)
        poly_degree = int(sup(guessdegree(approx_fct, approx_interval, S2**-(self.precision.get_field_size()+5)))) + 1

        poly_degree_list = list(range(1, poly_degree, 2))
        Log.report(Log.Debug, "poly_degree is {} and list {}", poly_degree, poly_degree_list)
        global_poly_object = Polynomial.build_from_approximation(approx_fct, poly_degree_list, [self.precision]*len(poly_degree_list), approx_interval, sollya.relative)
        Log.report(Log.Debug, "inform is {}", dirtyinfnorm(approx_fct - global_poly_object.get_sollya_object(), approx_interval))
        poly_object = global_poly_object.sub_poly(start_index=1, offset=1)

        ext_precision = {
            ML_Binary32: ML_SingleSingle,
            ML_Binary64: ML_DoubleDouble,
        }[self.precision]

        pre_poly = PolynomialSchemeEvaluator.generate_horner_scheme(
            poly_object, abs_vx, unified_precision=self.precision)

        result = FMA(pre_poly, abs_vx, abs_vx)
        result.set_attributes(tag="result", debug=debug_multi)


        eps_target = S2**-(self.precision.get_field_size() +5)
        def offset_div_function(fct):
            return lambda offset: fct(sollya.x + offset)

        # empiral numbers
        field_size = {
            ML_Binary32: 6,
            ML_Binary64: 8
        }[self.precision]

        near_indexing = SubFPIndexing(eps_exp, 0, 6, self.precision)
        medium_indexing = SubFPIndexing(1, one_limit_exp, 7, self.precision)

        if self.load_axf_approx:
            assert not self.dump_axf_approx

            # TODO: implement import of approximations from AXF files
            [near_axf_approx, medium_axf_approx] = AXF_JSON_Importer.from_file(self.load_axf_approx)

            near_approx_offset_table, near_approx_poly_max_degree, near_approx_poly_table, near_approx_max_error = load_piecewese_poly_params_from_axf(near_axf_approx, near_indexing)

            near_approx = generate_piecewise_poly_approx_from_params(near_approx_offset_table,
                                                         near_approx_poly_max_degree,
                                                         near_approx_poly_table,
                                                         near_indexing,
                                                         self.precision,
                                                         abs_vx)


            medium_approx_offset_table, medium_approx_poly_max_degree, medium_approx_poly_table, medium_approx_max_error = load_piecewese_poly_params_from_axf(medium_axf_approx, medium_indexing)

            medium_approx = generate_piecewise_poly_approx_from_params(medium_approx_offset_table,
                                                         medium_approx_poly_max_degree,
                                                         medium_approx_poly_table,
                                                         medium_indexing,
                                                         self.precision,
                                                         abs_vx)
        else:
            near_approx, axf_near_approx = generate_piecewise_poly_approx(offset_div_function(sollya.erf), near_indexing, eps_target, self.precision, abs_vx, axf_export=not self.dump_axf_approx is False)

            def offset_function(fct):
                return lambda offset: fct(sollya.x + offset)

            medium_approx, axf_medium_approx = generate_piecewise_poly_approx(offset_function(sollya.erf), medium_indexing, eps_target, self.precision, abs_vx, axf_export=not self.dump_axf_approx is False)

            if self.dump_axf_approx:
                axf_near_approx.tag = "erf-near"
                axf_medium_approx.tag = "erf-medium"
                #print(yaml.dump([axf_near_approx, axf_medium_approx]))
                AXF_JSON_Exporter.to_file(self.dump_axf_approx,
                                   [
                                    axf_near_approx.serialize_to_dict(),
                                    axf_medium_approx.serialize_to_dict()])

        near_approx.set_attributes(tag="near_approx", debug=debug_multi)
        medium_approx.set_attributes(tag="medium_approx", debug=debug_multi)

        # approximation for positive values
        scheme = ConditionBlock(
            abs_vx < eps,
            Return(result),
            ConditionBlock(
                abs_vx < near_indexing.max_bound,
                Return(near_approx),
                ConditionBlock(
                    abs_vx < medium_indexing.max_bound,
                    Return(medium_approx),
                    Return(Constant(1.0, precision=self.precision))
                )
            )
        )
        return scheme

    def numeric_emulate(self, input_value):
        return sollya.erf(input_value)

    standard_test_cases = [
        (sollya.parse("0x1.4c0d4e9f58p-8"),),
        (1.0, None),
        (4.0, None),
        (0.5, None),
        (1.5, None),
        (1024.0, None),
        (sollya.parse("0x1.13b2c6p-2"), None),
        (sollya.parse("0x1.2cb10ap-5"), None),
        (0.0, None),
        (sollya.parse("0x1.07e08ep+1"), None),
    ]



if __name__ == "__main__":
        # auto-test
        arg_template = ML_NewArgTemplate(default_arg=ML_Erf.get_default_args())

        arg_template.get_parser().add_argument(
             "--dump-axf-approx", default=False,
            action="store", help="dump approximations in AXF format")
        arg_template.get_parser().add_argument(
             "--load-axf-approx", default=False,
            action="store", help="load approximations from file in AXF format")

        args = arg_template.arg_extraction()
        ml_erf = ML_Erf(args)
        ml_erf.gen_implementation()
