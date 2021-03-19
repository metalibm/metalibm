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
# Description: wrapper around meta-external bench to support the sleef library
#             (sleef.org)
###############################################################################
from metalibm_functions.external_bench import (
    ML_ExternalBench, add_generic_cmd_args)

from metalibm_core.core.ml_operations import (
    BuildFromComponent, ComponentSelection, Addition)
from metalibm_core.core.ml_vectorizer import vectorize_format

from metalibm_core.utility.ml_template import (
    MultiAryArgTemplate, DefaultMultiAryArgTemplate,
    precision_parser)

# importing (installing) Sleef's type wrappers
import metalibm_functions.sleef_types as sleef_types

class SleefBench(ML_ExternalBench):
    """ Implementation of external bench function wrapper """
    function_name = "sleef_bench"
    def __init__(self, args=DefaultMultiAryArgTemplate):
        self.output_format = args.output_format
        super().__init__(args)
        # post-fixing output_format if undefined
        if self.output_format is None:
            self.output_format = self.precision

    def get_output_precision(self):
        return self.output_format

    def get_bench_storage_format(self):
        return vectorize_format(self.get_output_precision(), self.vector_size)

    def generate_chain_dep_op(self, result_format):
        """ extend generate_chain_dep_op to support sleef's types """
        if isinstance(result_format, sleef_types.SleefCompoundVectorFormat):
            field_format = {
                sleef_types.Sleef_SLEEF_VECTOR_FLOAT_2: sleef_types.SLEEF_VECTOR_FLOAT_field_format,
                sleef_types.Sleef_SLEEF_VECTOR_DOUBLE_2: sleef_types.SLEEF_VECTOR_DOUBLE_field_format
            }[result_format]
            def sleef_chain_op(local_acc, local_result):
                return BuildFromComponent(
                    Addition(
                        ComponentSelection(local_acc, "x", specifier=ComponentSelection.NamedField, precision=field_format),
                        ComponentSelection(local_result, "x", specifier=ComponentSelection.NamedField, precision=field_format),
                        precision=field_format
                    ),
                    Addition(
                        ComponentSelection(local_acc, "y", specifier=ComponentSelection.NamedField, precision=field_format),
                        ComponentSelection(local_result, "y", specifier=ComponentSelection.NamedField, precision=field_format),
                        precision=field_format
                    ),
                    precision=result_format
                )
            return sleef_chain_op
        elif isinstance(result_format, sleef_types.SleefCompoundFormat):
            def sleef_chain_op(local_acc, local_result):
                field_format = result_format.field_format_list[0]
                return BuildFromComponent(
                    Addition(
                        ComponentSelection(local_acc, "x", specifier=ComponentSelection.NamedField, precision=field_format),
                        ComponentSelection(local_result, "x", specifier=ComponentSelection.NamedField, precision=field_format),
                        precision=field_format
                    ),
                    Addition(
                        ComponentSelection(local_acc, "y", specifier=ComponentSelection.NamedField, precision=field_format),
                        ComponentSelection(local_result, "y", specifier=ComponentSelection.NamedField, precision=field_format),
                        precision=field_format
                    ),
                    precision=result_format
                )
            return sleef_chain_op
        return ML_ExternalBench.generate_chain_dep_op(self, result_format)

if __name__ == "__main__":
    # auto-test
    arg_template = MultiAryArgTemplate(default_arg=SleefBench.get_default_args())

    add_generic_cmd_args(arg_template)
    arg_template.get_parser().add_argument(
        "--output-format", dest="output_format", default=None,
        action="store", type=precision_parser,
        help="set a custom return format")

    # argument extraction
    args = arg_template.arg_extraction()
    sleef_bench = SleefBench(args)
    sleef_bench.gen_implementation()
