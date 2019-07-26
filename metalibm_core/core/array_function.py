# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/kalray/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2019 Kalray
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
# created:          Jul 26th, 2019
# last-modified:    Jul 26th, 2019
###############################################################################


import sollya
from sollya import Interval, inf, sup


from metalibm_core.code_generation.code_function import (
    CodeFunction, FunctionGroup
)
from metalibm_core.code_generation.generator_utility import (
    FunctionOperator, TemplateOperatorFormat,
    FO_Arg,
)
from metalibm_core.core.ml_operations import (
    Variable, Constant,
    Loop, ReferenceAssign, Statement,
    TableLoad, TableStore,
    FunctionObject,
    Return, ConditionBlock,
)
from metalibm_core.core.special_values import FP_QNaN
from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.core.random_gen import (
    FPRandomGen, MPFPRandomGen, FixedPointRandomGen
)
from metalibm_core.core.ml_function import (
    ML_FunctionBasis, DefaultArgTemplate
)
from metalibm_core.core.ml_formats import (
    ML_Int32, ML_UInt32, ML_Void,
    ML_FP_MultiElementFormat, ML_FP_Format, ML_Fixed_Format,
)

# TODO/FXIME: to be factorize with core.ml_function get_default_rng definition
def get_precision_rng(precision, inf_bound, sup_bound):
    """ build a random number generator for format @p precision
        which generates values within the range [inf_bound, sup_bound] """
    if isinstance(precision, ML_FP_MultiElementFormat):
        return MPFPRandomGen.from_interval(precision, inf_bound, sup_bound)
    elif isinstance(precision, ML_FP_Format):
        return FPRandomGen.from_interval(precision, inf_bound, sup_bound)
    elif isinstance(precision, ML_Fixed_Format):
        return FixedPointRandomGen.from_interval(precision, inf_bound, sup_bound)
    else:
        Log.report(Log.Error, "unsupported format {} in get_precision_rng", precision)

class ML_ArrayFunction(ML_FunctionBasis):
    """ generic function working on arbitrary length arrays """
    def element_numeric_emulate(self):
        """ single element emulation of function """
        raise NotImplementedError

    def generate_test_wrapper(self, test_num=10, index_range=[0, 100], test_range=Interval(-1.0, 1.0), debug=False):
        low_input = inf(test_range)
        high_input = sup(test_range)
        auto_test = CodeFunction("test_wrapper", output_format = ML_Int32)

        tested_function    = self.implementation.get_function_object()
        function_name      = self.implementation.get_name()

        failure_report_op       = FunctionOperator("report_failure")
        failure_report_function = FunctionObject("report_failure", [], ML_Void, failure_report_op)

        printf_success_op = FunctionOperator("printf", arg_map = {0: "\"test successful %s\\n\"" % function_name}, void_function = True) 
        printf_success_function = FunctionObject("printf", [], ML_Void, printf_success_op)

        test_total   = test_num

        sollya_precision = self.precision.get_sollya_object()
        interval_size = high_input - low_input

        NUM_INPUT_ARRAY = 1
        INPUT_INDEX_OFFSET = 1
        INPUT_ARRAY_MAX_SIZE = max(index_range)

        input_tables = [
          ML_NewTable(
            dimensions = [INPUT_ARRAY_MAX_SIZE],
            storage_precision=self.get_input_precision(INPUT_INDEX_OFFSET + i).get_data_precision(),
            tag = self.uniquify_name("input_table_arg%d" % i)
          ) for i in range(NUM_INPUT_ARRAY)
        ]
        ## output values required to check results are stored in output table
        num_output_value = self.accuracy.get_num_output_value()
        expected_table = ML_NewTable(dimensions = [INPUT_ARRAY_MAX_SIZE, num_output_value], storage_precision = self.precision, tag = self.uniquify_name("expected_table"))
        output_array = ML_NewTable(dimensions = [INPUT_ARRAY_MAX_SIZE], storage_precision=self.precision, tag=self.uniquify_name("output_array"))

        # general index for input/output tables
        table_index = 0

        test_case_list = []

        # TODO/FIXME: implement proper input range depending on input index
        # assuming a single input array
        input_precisions = [self.get_input_precision(1).get_data_precision()]
        rng_map = [get_precision_rng(precision, low_input, high_input) for precision in input_precisions]

        assert test_num == 1


        # random test cases
        for i in range(INPUT_ARRAY_MAX_SIZE):
            # resetting output array
            output_array[i] = FP_QNaN(self.precision)
            input_list = []
            for in_id in range(NUM_INPUT_ARRAY):
                # this random generator is limited to python float precision
                # (generally machine double precision)
                # TODO/FIXME: implement proper high precision generation
                # based on real input_precision (e.g. ML_DoubleDouble)
                input_precision = input_precisions[in_id]
                input_value = rng_map[in_id].get_new_value() # random.uniform(low_input, high_input)
                input_value = input_precision.round_sollya_object(input_value, sollya.RN)
                input_list.append(input_value)
            test_case_list.append(tuple(input_list))

        # generating output from the concatenated list
        # of all inputs
        for table_index, input_tuple in enumerate(test_case_list):
            # storing inputs
            for in_id in range(NUM_INPUT_ARRAY):
                input_tables[in_id][table_index] = input_tuple[in_id]
            # computing and storing output values
            output_values = self.accuracy.get_output_check_value(self, input_tuple)
            for o in range(num_output_value):
                expected_table[table_index][o] = output_values[o]

        # scalar implemetation test
        test_loop = self.get_array_test_wrapper(test_total, tested_function, input_tables, expected_table, output_array)

        # common test scheme between scalar and vector functions
        test_scheme = Statement(
          test_loop,
          printf_success_function(),
          Return(Constant(0, precision = ML_Int32))
        )
        auto_test.set_scheme(test_scheme)
        return FunctionGroup([auto_test])



    def get_printf_input_function(self):
        input_precisions = [self.get_input_precision(0).get_data_precision()]

        # build the complete format string from the input precisions
        input_display_formats = ", ".join(prec.get_display_format().format_string for prec in input_precisions)
        input_display_vars = ", ".join(prec.get_display_format().pre_process_fct("{%d}" % index) for index, prec in enumerate(input_precisions, 1))

        result_arg_id = 1 + len(input_precisions)
        # expected_arg_id = 1 + result_arg_id
        # build the format string for result/expected display
        result_display_format = self.precision.get_display_format().format_string
        result_display_vars = self.precision.get_display_format().pre_process_fct("{%d}" % result_arg_id)
        # expected_display_vars = self.precision.get_display_format().pre_process_fct("{%d}" % expected_arg_id)

        template = ("printf(\"error[%d]: {fct_name}({arg_display_format}),"
                    " result is {result_display_format} "
                    "vs expected \""
                    ", {{0}}, {arg_display_vars}, {result_display_vars}"
                    ")").format(
                        fct_name=self.function_name,
                        arg_display_format=input_display_formats,
                        arg_display_vars=input_display_vars,
                        result_display_format=result_display_format,
                        #expected_display_format=result_display_format,
                        result_display_vars=result_display_vars,
                        #expected_display_vars=expected_display_vars
                    )
        printf_op = TemplateOperatorFormat(template, void_function=True, arity=(result_arg_id+1)) 
        printf_input_function = FunctionObject("printf", [ML_Int32] + self.get_input_precisions() + [self.precision], ML_Void, printf_op)
        return printf_input_function

    ## generate a test loop for scalar tests
    #    @param test_num number of elementary tests to be executed
    #    @param tested_function FunctionObject to be tested
    #    @param input_table ML_NewTable object containing test inputs
    #    @param output_table ML_NewTable object containing test outputs
    #    @param printf_function FunctionObject to print error case
    def get_array_test_wrapper(self, test_num, tested_function, input_tables, expected_table, output_array, NUM_INPUT_ARRAY=1):
        vi = Variable("i", precision = ML_Int32, var_type = Variable.Local)
        test_num_cst = Constant(test_num, precision = ML_Int32, tag = "test_num")

        array_len = Variable("len", precision=ML_UInt32, var_type=Variable.Local)


        array_inputs    = tuple(input_tables[in_id] for in_id in range(NUM_INPUT_ARRAY))
        function_call = tested_function(*((output_array,) + array_inputs + (array_len,)))

        # internal array iterator index
        vj = Variable("j", precision=ML_UInt32, var_type=Variable.Local)


        local_inputs = tuple(TableLoad(input_tables[in_id], vj) for in_id in range(NUM_INPUT_ARRAY))
        expected_values = [TableLoad(expected_table, vj, i) for i in range(self.accuracy.get_num_output_value())]


        printf_input_function = self.get_printf_input_function()

        printf_error_template = "printf(\"max %s error is %s \\n\", %s)" % (
            self.function_name,
            self.precision.get_display_format().format_string,
            self.precision.get_display_format().pre_process_fct("{0}")
        )
        printf_error_op = TemplateOperatorFormat(printf_error_template, arity=1, void_function=True)

        printf_error_function = FunctionObject("printf", [self.precision], ML_Void, printf_error_op)

        printf_max_op = FunctionOperator("printf", arg_map = {0: "\"max %s error is reached at input number %s \\n \"" % (self.function_name, "%d"), 1: FO_Arg(0)}, void_function = True) 
        printf_max_function = FunctionObject("printf", [self.precision], ML_Void, printf_max_op)

        loop_increment = 1

        local_result = TableLoad(output_array, vj)

        if self.break_error:
                return_statement_break = Statement(
                        printf_input_function(*((vi,) + local_inputs + (local_result,))), 
                        self.accuracy.get_output_print_call(self.function_name, output_values)
                )
        else:
                return_statement_break = Statement(
                        printf_input_function(*((vi,) + local_inputs + (local_result,))), 
                        self.accuracy.get_output_print_call(self.function_name, expected_values),
                        Return(Constant(1, precision = ML_Int32))
                )

        check_array_loop = Loop(
            ReferenceAssign(vj, 0),
            vj < array_len,
            Statement(
                ConditionBlock(
                    self.accuracy.get_output_check_test(
                        local_result,
                        expected_values
                    ),
                    return_statement_break
                ),
                ReferenceAssign(vj, vj+1),
            )
        )

        test_loop = Loop(
            ReferenceAssign(vi, Constant(0, precision = ML_Int32)),
            vi < test_num_cst,
            Statement(
                ReferenceAssign(array_len, Constant(17)),
                function_call,
                check_array_loop,
                ReferenceAssign(vi, vi + loop_increment)
            ),
        )

        test_statement = Statement()

        # adding functional test_loop to test statement
        test_statement.add(test_loop)

        return test_statement

