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
    Division, Conversion, Subtraction,
)
from metalibm_core.core.special_values import FP_QNaN
from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.core.random_gen import get_precision_rng
from metalibm_core.core.ml_function import (
    ML_FunctionBasis, DefaultArgTemplate
)
from metalibm_core.core.ml_formats import (
    ML_Int32, ML_UInt32, ML_Void,
    ML_Binary64, ML_Int64,
    FormatAttributeWrapper,
)

def generate_1d_table(dim, storage_precision, tag, value_gen=lambda index: None, empty=False):
    """ generate a 1D ML_NewTable by using the given value generator @p value_gen """
    gen_table = ML_NewTable(
        dimensions = [dim],
        storage_precision=storage_precision,
        tag=tag,
        empty=empty
    )
    for i in range(dim):
        gen_table[i] = value_gen(i)
    return gen_table

def generate_2d_table(dim0, dim1, storage_precision, tag, value_gen=lambda index0: None):
    """ generate a 1D ML_NewTable by using the given value generator @p value_gen,
        values are generated one row at a time (rather than cell by cell) """
    gen_table = ML_NewTable(
        dimensions = [dim0, dim1],
        storage_precision=storage_precision,
        tag=tag
    )
    for i0 in range(dim0):
        row_values = value_gen(i0)
        for i1 in range(dim1):
            gen_table[i0][i1] = row_values[i1]
    return gen_table

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

        # TODO/FIXME: implement proper input range depending on input index
        # assuming a single input array
        input_precisions = [self.get_input_precision(1).get_data_precision()]
        rng_map = [get_precision_rng(precision, low_input, high_input) for precision in input_precisions]

        assert test_num == 1

        # generated table of inputs
        input_tables = [
            generate_1d_table(
                INPUT_ARRAY_MAX_SIZE,
                self.get_input_precision(INPUT_INDEX_OFFSET + table_id).get_data_precision(),
                self.uniquify_name("input_table_arg%d" % table_id),
                value_gen=(lambda _: input_precisions[table_id].round_sollya_object(rng_map[table_id].get_new_value(), sollya.RN))
            ) for table_id in range(NUM_INPUT_ARRAY)
        ]

        # generate output_array
        output_array = generate_1d_table(
            INPUT_ARRAY_MAX_SIZE,
            self.precision,
            self.uniquify_name("output_array"),
            value_gen=(lambda _: FP_QNaN(self.precision))
        )


        #check_array_loop = self.generate_array_check_loop(input_tables, expected_table, output_array, array_len)
            
        # accumulate element number
        acc_num = Variable("acc_num", precision=ML_Int64, var_type=Variable.local)

        test_loop = self.get_array_test_wrapper(
            test_total, tested_function, input_tables,
            output_array, acc_num,
            self.generate_array_check_loop,
            INPUT_ARRAY_MAX_SIZE)

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

    def generate_array_check_loop(self, input_tables, output_array, array_len, INPUT_ARRAY_MAX_SIZE):
        # internal array iterator index
        vj = Variable("j", precision=ML_UInt32, var_type=Variable.Local)

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


        NUM_INPUT_ARRAY = len(input_tables)

        ## output values required to check results are stored in output table
        num_output_value = self.accuracy.get_num_output_value()


        def expected_value_gen(row_id):
            """ generate a full row of expected values using inputs from input_tables"""
            output_values = self.accuracy.get_output_check_value(self, tuple(input_tables[table_index][row_id] for table_index in range(NUM_INPUT_ARRAY)))
            return output_values

        # generating expected value table
        expected_table = generate_2d_table(
            INPUT_ARRAY_MAX_SIZE, num_output_value,
            self.precision,
            "expected_table",
            value_gen=expected_value_gen
        )

        local_inputs = tuple(TableLoad(input_tables[in_id], vj) for in_id in range(NUM_INPUT_ARRAY))
        expected_values = [TableLoad(expected_table, vj, i) for i in range(self.accuracy.get_num_output_value())]
        local_result = TableLoad(output_array, vj)

        if self.break_error:
            return_statement_break = Statement(
                printf_input_function(*((vj,) + local_inputs + (local_result,))), 
                self.accuracy.get_output_print_call(self.function_name, output_values)
            )
        else:
            return_statement_break = Statement(
                printf_input_function(*((vj,) + local_inputs + (local_result,))), 
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
        return check_array_loop

    def get_array_test_wrapper(
            self, test_num, tested_function, input_tables, output_array,
            acc_num,
            post_statement_generator, INPUT_ARRAY_MAX_SIZE, NUM_INPUT_ARRAY=1):
        """ generate a test loop for scalar tests
             @param test_num number of elementary tests to be executed
             @param tested_function FunctionObject to be tested
             @param input_table ML_NewTable object containing test inputs
             @param output_table ML_NewTable object containing test outputs
             @param post_statement is a statement executed after the benched
                    function has been called (example to check result)
             @param printf_function FunctionObject to print error case
        """
        vi = Variable("i", precision = ML_Int32, var_type = Variable.Local)
        test_num_cst = Constant(test_num, precision = ML_Int32, tag = "test_num")

        array_len = Variable("len", precision=ML_UInt32, var_type=Variable.Local)

        array_inputs    = tuple(input_tables[in_id] for in_id in range(NUM_INPUT_ARRAY))
        function_call = tested_function(*((output_array,) + array_inputs + (array_len,)))

        post_statement = post_statement_generator(input_tables, output_array, array_len, INPUT_ARRAY_MAX_SIZE)

        loop_increment = 1

        test_loop = Loop(
            ReferenceAssign(vi, Constant(0, precision = ML_Int32)),
            vi < test_num_cst,
            Statement(
                ReferenceAssign(array_len, Constant(17)),
                function_call,
                post_statement,
                ReferenceAssign(acc_num, acc_num + Conversion(array_len, precision=acc_num.precision)),
                ReferenceAssign(vi, vi + loop_increment),
            ),
        )

        test_statement = Statement()

        # adding functional test_loop to test statement
        test_statement.add(test_loop)

        return test_statement

    ## Generate a test wrapper for the @p self function 
    #    @param test_num     number of test to perform
    #    @param test_range numeric range for test's inputs
    #    @param debug enable debug mode
    def generate_bench_wrapper(self, test_num=1, loop_num=100000, test_range=Interval(-1.0, 1.0), debug=False, index_range=[1, 117]):
        low_input = inf(test_range)
        high_input = sup(test_range)
        auto_test = CodeFunction("bench_wrapper", output_format=ML_Binary64)

        tested_function        = self.implementation.get_function_object()
        function_name            = self.implementation.get_name()

        failure_report_op             = FunctionOperator("report_failure")
        failure_report_function = FunctionObject("report_failure", [], ML_Void, failure_report_op)


        printf_success_op = FunctionOperator("printf", arg_map = {0: "\"test successful %s\\n\"" % function_name}, void_function = True) 
        printf_success_function = FunctionObject("printf", [], ML_Void, printf_success_op)

        output_precision = FormatAttributeWrapper(self.precision, ["volatile"])

        test_total = test_num

        NUM_INPUT_ARRAY = 1
        INPUT_INDEX_OFFSET = 1
        INPUT_ARRAY_MAX_SIZE = max(index_range)


        # TODO/FIXME: implement proper input range depending on input index
        # assuming a single input array
        input_precisions = [self.get_input_precision(1).get_data_precision()]
        rng_map = [get_precision_rng(precision, low_input, high_input) for precision in input_precisions]

        assert test_num == 1

        # generated table of inputs
        input_tables = [
            generate_1d_table(
                INPUT_ARRAY_MAX_SIZE,
                self.get_input_precision(INPUT_INDEX_OFFSET + table_id).get_data_precision(),
                self.uniquify_name("input_table_arg%d" % table_id),
                value_gen=(lambda _: input_precisions[table_id].round_sollya_object(rng_map[table_id].get_new_value(), sollya.RN))
            ) for table_id in range(NUM_INPUT_ARRAY)
        ]

        # generate output_array
        output_array = generate_1d_table(
            INPUT_ARRAY_MAX_SIZE,
            output_precision,
            self.uniquify_name("output_array"),
            #value_gen=(lambda _: FP_QNaN(self.precision))
            value_gen=(lambda _: None),
            empty=True
        )


        # accumulate element number
        acc_num = Variable("acc_num", precision=ML_Int64, var_type=Variable.Local)
            
        test_loop = self.get_array_test_wrapper(
            test_total, tested_function, input_tables,
            output_array,
            acc_num,
            lambda a, b, c, d: Statement(), INPUT_ARRAY_MAX_SIZE)

        timer = Variable("timer", precision = ML_Int64, var_type = Variable.Local)
        printf_timing_op = FunctionOperator(
                "printf",
                arg_map = {
                        0: "\"%s %%\"PRIi64\" elts computed in %%\"PRIi64\" nanoseconds => %%.3f CPE \\n\"" % function_name,
                        1: FO_Arg(0), 2: FO_Arg(1),
                        3: FO_Arg(2)
                }, void_function = True
        )
        printf_timing_function = FunctionObject("printf", [ML_Int64, ML_Int64, ML_Binary64], ML_Void, printf_timing_op)

        vj = Variable("j", precision=ML_Int32, var_type=Variable.Local)
        loop_num_cst = Constant(loop_num, precision=ML_Int32, tag="loop_num")
        loop_increment = 1

        # bench measure of clock per element
        cpe_measure = Division(
                Conversion(timer, precision=ML_Binary64),
                Conversion(acc_num, precision=ML_Binary64),
                precision=ML_Binary64,
                tag="cpe_measure",
        )

        # common test scheme between scalar and vector functions
        test_scheme = Statement(
            self.processor.get_init_timestamp(),
            ReferenceAssign(timer, self.processor.get_current_timestamp()),
            ReferenceAssign(acc_num, 0),
            Loop(
                    ReferenceAssign(vj, Constant(0, precision=ML_Int32)),
                    vj < loop_num_cst,
                    Statement(
                            test_loop,
                            ReferenceAssign(vj, vj + loop_increment)
                    )
            ),
            ReferenceAssign(timer,
                Subtraction(
                    self.processor.get_current_timestamp(),
                    timer,
                    precision = ML_Int64
                )
            ),
            printf_timing_function(
                Conversion(acc_num, precision=ML_Int64),
                timer,
                cpe_measure,
            ),
            Return(cpe_measure),
            # Return(Constant(0, precision = ML_Int32))
        )
        auto_test.set_scheme(test_scheme)
        return FunctionGroup([auto_test])
