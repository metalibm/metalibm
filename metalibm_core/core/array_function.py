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

import random
from metalibm_core.core.precisions import ML_FunctionPrecision
from metalibm_core.opt.p_expand_multi_precision import Pass_ExpandMultiPrecision
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
    LogicalAnd, LogicalNot, Max, Select, Test, Variable, Constant,
    Loop, ReferenceAssign, Statement,
    TableLoad,
    FunctionObject,
    Return, ConditionBlock,
    Division, Conversion, Subtraction, Addition,
    Equal
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

from metalibm_core.utility.ml_template import ML_NewArgTemplate

def generate_1d_table(dim, storage_precision, tag, value_gen=lambda index: None, empty=False, const=True):
    """ generate a 1D ML_NewTable by using the given value generator @p value_gen """
    gen_table = ML_NewTable(
        dimensions = [dim],
        storage_precision=storage_precision,
        tag=tag,
        const=const,
        empty=empty
    )
    for i in range(dim):
        gen_table[i] = value_gen(i)
    return gen_table

def generate_2d_table(dim0, dim1, storage_precision, tag, value_gen=(lambda index0: None), const=True):
    """ generate a 2D ML_NewTable by using the given value generator @p value_gen,
        values are generated one row at a time (rather than cell by cell) """
    gen_table = ML_NewTable(
        dimensions = [dim0, dim1],
        storage_precision=storage_precision,
        const=const,
        tag=tag
    )
    for i0 in range(dim0):
        row_values = value_gen(i0)
        for i1 in range(dim1):
            gen_table[i0][i1] = row_values[i1]
    return gen_table


def generate_2d_multi_table(size_offset_list, dim1, storage_precision, tag, value_gen=lambda table_index, sub_row_index: None):
    """ generate a 2D multi-array stored in a ML_NewTable. 
        The multi-array dimensions are defined by the (size, offset) pairs in size_offset_list
        for the first dimension and @p dim1 for the second dimension.
        Table value are obtained by using the given value generator @p value_gen,
        values are generated one row at a time (rather than cell by cell) """
    # table first dimension is the sum of each sub-array size
    dim0 = sum(size_offset_list[sub_id][0] for sub_id in range(size_offset_list.dimensions[0]))

    gen_table = ML_NewTable(
        dimensions = [dim0, dim1],
        storage_precision=storage_precision,
        tag=tag
    )
    for table_index, (size, offset) in enumerate(size_offset_list):
        for i0 in range(size):
            row_values = value_gen(table_index, i0)
            for i1 in range(dim1):
                gen_table[offset + i0][i1] = row_values[i1]
    return gen_table

class DefaultArrayFunctionArgTemplate(DefaultArgTemplate):
    test_index_range = [0, 10]

class ML_ArrayFunctionArgTemplate(ML_NewArgTemplate):
    def __init__(self, default_arg=DefaultArrayFunctionArgTemplate):
        ML_NewArgTemplate.__init__(self, default_arg)
        self.parser.add_argument(
            "--test-index-range", dest="test_index_range", action="store",
            type=(lambda s: list(int(v) for v in s.split(","))),
            default=default_arg.test_index_range,
            help="interval for test arrays size"
        )


class ML_ArrayFunction(ML_FunctionBasis):
    """ generic function working on arbitrary length arrays """
    def __init__(self, args=DefaultArrayFunctionArgTemplate):
        ML_FunctionBasis.__init__(self, args)
        self.test_index_range = args.test_index_range

    def element_numeric_emulate(self):
        """ single element emulation of function """
        raise NotImplementedError


    def generate_test_tables(self, test_num, outType: ML_FunctionPrecision, outAccuracy, test_ranges=[Interval(-1.0, 1.0)]):
        """ Generate inputs and output table to be shared between auto test
            and max_error tests
            
            Args:
                outType (ML_Format): function output type
                outAccuracy (ML_FunctionPrecision): function expected accuracy (unsued)
                test_ranges : range(s) to select inputs from
            """
        index_range = self.test_index_range
        test_total  = test_num + len(self.standard_test_cases) + (1 if len(self.value_test) else 0)

        # number of arrays expected as inputs for tested_function
        NUM_INPUT_ARRAY = 1
        # position of the first input array in tested_function operands (generally
        # equals to 1 as to 0-th input is often the destination array)
        INPUT_INDEX_OFFSET = 1

        # concatenating standard test array at the beginning of randomly
        # generated array
        TABLE_SIZE_VALUES = ([len(self.value_test)] if len(self.value_test) else []) + [len(std_table) for std_table in self.standard_test_cases] + [random.randrange(index_range[0], index_range[1] + 1) for i in range(test_num)]
        OFFSET_VALUES = [sum(TABLE_SIZE_VALUES[:i]) for i in range(test_total)]

        table_size_offset_array = generate_2d_table(
            test_total, 2,
            ML_UInt32,
            self.uniquify_name("table_size_array"),
            value_gen=(lambda row_id: (TABLE_SIZE_VALUES[row_id], OFFSET_VALUES[row_id]))
        )
        INPUT_ARRAY_SIZE = sum(TABLE_SIZE_VALUES)

        # TODO/FIXME: implement proper input range depending on input index
        # assuming a single input array
        input_precisions = [self.get_input_precision(1).get_data_precision()]
        rng_map = [get_precision_rng(precision, test_range) for precision, test_range in zip(input_precisions, test_ranges)]

        def inputValueGen(table_id):
            """ input value generator, select inputs from command-line value_test,
                from stantard_cases or randomly generated based on index """
            def helper(index):
                value = None
                if value is None and index < len(self.value_test):
                    value = self.value_test[index][table_id]
                index -= len(self.value_test)
                if value is None and index < len(self.standard_test_cases):
                    value = self.standard_test_cases[index]
                if value is None:
                    value = rng_map[table_id].get_new_value()
                return input_precisions[table_id].round_sollya_object(value, sollya.RN)
            return helper
            

        # generated table of inputs
        input_tables = [
            generate_1d_table(
                INPUT_ARRAY_SIZE,
                self.get_input_precision(INPUT_INDEX_OFFSET + table_id).get_data_precision(),
                self.uniquify_name("input_table_arg%d" % table_id),
                value_gen=inputValueGen(table_id)
            ) for table_id in range(NUM_INPUT_ARRAY)
        ]

        # generate the expected table for the whole multi-array
        expected_array = self.generate_expected_table(outType, outAccuracy, input_tables, table_size_offset_array)

        return test_total, (table_size_offset_array, input_tables), expected_array


    def generate_test_wrapper(self, test_total, input_tuple, expected_array):
        table_size_offset_array, input_tables = input_tuple

        auto_test = CodeFunction("test_wrapper", output_format = ML_Int32)

        tested_function    = self.implementation.get_function_object()
        function_name      = self.implementation.get_name()

        printf_success_op = FunctionOperator("printf", arg_map = {0: "\"test successful %s\\n\"" % function_name}, void_function = True, require_header=["stdio.h"]) 
        printf_success_function = FunctionObject("printf", [], ML_Void, printf_success_op)

        # accumulate element number
        acc_num = Variable("acc_num", precision=ML_Int64, var_type=Variable.Local)

        # the total number of input values can be evaluated as the sum of the elements of the
        # last row of table_size_offtset_array which contain the last offset and the the last array size
        INPUT_ARRAY_SIZE = table_size_offset_array[test_total - 1][0] + table_size_offset_array[test_total - 1][1]
        outType = self.get_output_precision()

        # generate output_array (empty table to store implementation results before compare)
        output_array = generate_1d_table(
            INPUT_ARRAY_SIZE,
            outType,
            self.uniquify_name("output_array"),
            const=False,
            value_gen=(lambda _: FP_QNaN(outType))
        )

        test_loop = self.get_array_test_wrapper(
            test_total, tested_function,
            table_size_offset_array,
            input_tables,
            expected_array,
            output_array,
            acc_num,
            self.generate_array_check_loop)

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

        template = ("printf(\"error[%u]: {fct_name}({arg_display_format}),"
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
        printf_op = TemplateOperatorFormat(template, void_function=True, arity=(result_arg_id+1), require_header=["stdio.h"]) 
        printf_input_function = FunctionObject("printf", [ML_UInt32] + input_precisions + [self.precision], ML_Void, printf_op)
        return printf_input_function


    def generate_expected_table(self, outPrecision, outAccuracy, input_tables, table_size_offset_array):
        """ Generate the complete table of expected results """
        ## output values required to check results are stored in expected_table
        num_output_value = outAccuracy.get_num_output_value()
        NUM_INPUT_ARRAY = len(input_tables)

        def expected_value_gen(table_id, table_row_id):
            """ generate a full row of expected values using inputs from input_tables"""
            table_offset = table_size_offset_array[table_id][1]
            row_id = table_offset + table_row_id
            output_values = outAccuracy.get_output_check_value(self.numeric_emulate(*tuple(input_tables[table_index][row_id] for table_index in range(NUM_INPUT_ARRAY))))
            return output_values

        # generating expected value table
        expected_table = generate_2d_multi_table(
            table_size_offset_array, num_output_value,
            outPrecision,
            "expected_table",
            value_gen=expected_value_gen
        )
        return expected_table


    def generate_array_check_loop(self, input_tables, expected_array, output_array, table_size_offset_array,
                                  array_offset, array_len, test_id):
        # internal array iterator index
        vj = Variable("j", precision=ML_UInt32, var_type=Variable.Local)

        printf_input_function = self.get_printf_input_function()

        NUM_INPUT_ARRAY = len(input_tables)

        # inputs for the (vj)-th entry of the sub-arrat
        local_inputs = tuple(TableLoad(input_tables[in_id], array_offset + vj) for in_id in range(NUM_INPUT_ARRAY))
        # expected values for the (vj)-th entry of the sub-arrat
        expected_values = [TableLoad(expected_array, array_offset + vj, i) for i in range(self.accuracy.get_num_output_value())]
        # local result for the (vj)-th entry of the sub-arrat
        local_result = TableLoad(output_array, array_offset + vj)

        if self.break_error:
            return_statement_break = Statement(
                printf_input_function(*((vj,) + local_inputs + (local_result,))), 
                self.accuracy.get_output_print_call(self.function_name, expected_values)
            )
        else:
            return_statement_break = Statement(
                printf_input_function(*((vj,) + local_inputs + (local_result,))), 
                self.accuracy.get_output_print_call(self.function_name, expected_values),
                Return(Constant(1, precision = ML_Int32))
            )

        # loop implementation to check sub-array array_offset
        # results validity
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
            self,
            test_num, tested_function,
            table_size_offset_array,
            input_tables, expected_array, output_array,
            acc_num,
            post_statement_generator,
            NUM_INPUT_ARRAY=1):
        """ generate a test loop for mult    def get_output_precision(self):
        return ML_Void
i-array tests
             @param test_num number of elementary array tests to be executed
             @param tested_function FunctionObject to be tested
             @param table_size_offset_array ML_NewTable object containing
                    (table-size, offset) pairs for multi-array testing
             @param input_table ML_NewTable containing multi-array test inputs
             @param output_table ML_NewTable containing multi-array test outputs
             @param post_statement_generator is generator used to generate
                    a statement executed at the end of the test of one of the
                    arrays of the multi-test. It expects 6 arguments:
                    (input_tables, output_array, table_size_offset_array,
                     array_offset, array_len, test_id)
             @param printf_function FunctionObject to print error case
        """
        test_id = Variable("test_id", precision = ML_Int32, var_type = Variable.Local)
        test_num_cst = Constant(test_num, precision = ML_Int32, tag = "test_num")

        array_len = Variable("len", precision=ML_UInt32, var_type=Variable.Local)

        array_offset = TableLoad(table_size_offset_array, test_id, 1)

        def pointer_add(table_addr, offset):
            pointer_format = table_addr.get_precision_as_pointer_format()
            return Addition(table_addr, offset, precision=pointer_format)

        array_inputs    = tuple(pointer_add(input_tables[in_id], array_offset) for in_id in range(NUM_INPUT_ARRAY))
        function_call = tested_function(
            *((pointer_add(output_array, array_offset),) + array_inputs + (array_len,)))


        post_statement = post_statement_generator(
                            input_tables, expected_array, output_array, table_size_offset_array,
                            array_offset, array_len, test_id)

        loop_increment = 1

        test_loop = Loop(
            ReferenceAssign(test_id, Constant(0, precision = ML_Int32)),
            test_id < test_num_cst,
            Statement(
                ReferenceAssign(array_len, TableLoad(table_size_offset_array, test_id, 0)),
                function_call,
                post_statement,
                ReferenceAssign(acc_num, acc_num + Conversion(array_len, precision=acc_num.precision)),
                ReferenceAssign(test_id, test_id + loop_increment),
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
    def generate_bench_wrapper(self, test_num=1, loop_num=100000, test_ranges=[Interval(-1.0, 1.0)], debug=False):
        # interval where the array lenght is chosen from (randomly)
        index_range = self.test_index_range

        # bench_wrapper result is always of type ML_Binary64, it is a CPE measure
        auto_test = CodeFunction("bench_wrapper", output_format=ML_Binary64)

        tested_function   = self.implementation.get_function_object()
        function_name     = self.implementation.get_name()

        # BUGFIX: adding the volatile attribute make the bench output array incompatible
        #         with the array function prototype which does not contain the volatile attribute
        # output_precision = FormatAttributeWrapper(self.precision, ["volatile"])
        output_precision = self.precision

        test_total = test_num

        # number of arrays expected as inputs for tested_function
        NUM_INPUT_ARRAY = 1
        # position of the input array in tested_function operands (generally
        # equals to 1 as to 0-th input is often the destination array)
        INPUT_INDEX_OFFSET = 1


        # concatenating standard test array at the beginning of randomly
        # generated array
        TABLE_SIZE_VALUES = [len(std_table) for std_table in self.standard_test_cases] + \
                            [random.randrange(index_range[0], index_range[1] + 1) for i in range(test_num)]
        OFFSET_VALUES = [sum(TABLE_SIZE_VALUES[:i]) for i in range(test_total)]

        table_size_offset_array = generate_2d_table(
            test_total, 2,
            ML_UInt32,
            self.uniquify_name("table_size_array"),
            value_gen=(lambda row_id: (TABLE_SIZE_VALUES[row_id], OFFSET_VALUES[row_id]))
        )

        INPUT_ARRAY_SIZE = sum(TABLE_SIZE_VALUES)


        # TODO/FIXME: implement proper input range depending on input index
        # assuming a single input array
        input_precisions = [self.get_input_precision(1).get_data_precision()]
        rng_map = [get_precision_rng(precision, Interval(inf(test_range), sup(test_range))) for precision, test_range in zip(input_precisions, test_ranges)]

        # generated table of inputs
        input_tables = [
            generate_1d_table(
                INPUT_ARRAY_SIZE,
                self.get_input_precision(INPUT_INDEX_OFFSET + table_id).get_data_precision(),
                self.uniquify_name("input_table_arg%d" % table_id),
                value_gen=(lambda _: input_precisions[table_id].round_sollya_object(rng_map[table_id].get_new_value(), sollya.RN))
            ) for table_id in range(NUM_INPUT_ARRAY)
        ]

        # generate the expected table for the whole multi-array
        expected_array = self.generate_expected_table(self.get_output_precision(), self.accuracy, input_tables, table_size_offset_array)

        # generate output_array
        output_array = generate_1d_table(
            INPUT_ARRAY_SIZE,
            output_precision,
            self.uniquify_name("output_array"),
            #value_gen=(lambda _: FP_QNaN(self.precision))
            value_gen=(lambda _: None),
            const=False,
            empty=True
        )


        # accumulate element number
        acc_num = Variable("acc_num", precision=ML_Int64, var_type=Variable.Local)

        def empty_post_statement_gen(input_tables, output_array,
                                     table_size_offset_array, array_offset,
                                     array_len, test_id):
            return Statement()

        test_loop = self.get_array_test_wrapper(
            test_total, tested_function,
            table_size_offset_array,
            input_tables, expected_array, output_array,
            acc_num,
            empty_post_statement_gen)

        timer = Variable("timer", precision = ML_Int64, var_type = Variable.Local)
        printf_timing_op = FunctionOperator(
                "printf",
                arg_map = {
                        0: "\"%s %%\"PRIi64\" elts computed in %%\"PRIi64\" nanoseconds => %%.3f CPE \\n\"" % function_name,
                        1: FO_Arg(0), 2: FO_Arg(1),
                        3: FO_Arg(2)
                },
                void_function = True,
                require_header=["stdio.h", "inttypes.h"]
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

    def get_main_fct_return_type(self):
        # Array-function does not return anything
        return ML_Void

    def generate_max_error_wrapper(self, tested_function, test_total,
                                   input_tables, output_table,
                                   errorPrecision, errorAccuracy):
        """ Generate max_errror eval function, manages
            both scalar and vector formats """
        max_error_main_statement = self.generate_vector_max_error_eval(tested_function, test_total, input_tables, output_table, errorPrecision, errorAccuracy) 
        max_error_function = CodeFunction("max_error_eval", output_format=errorPrecision) 
        max_error_function.set_scheme(max_error_main_statement)
        return max_error_function
  
    def generate_vector_max_error_eval(self, tested_function, test_num,
                                       input_bundle, output_table, outType, outAccuracy):
        """ generate the main Statement to evaluate the maximal error (both
            relative and absolute) for a vector function """
        offsetArray, input_tables = input_bundle
        errorPrecision = outType
        print(f"errorPrecision={errorPrecision}")
        max_error_relative = Variable("max_error_relative", precision=errorPrecision, var_type=Variable.Local)
        max_error_absolute = Variable("max_error_absolute", precision=errorPrecision, var_type=Variable.Local)
  
        printf_error_template = "printf(\"max %s error is absolute=%s reached at index %s, relative=%s reached at index %s\\n\", %s, %s, %s, %s)" % (
          self.function_name,
          errorPrecision.get_display_format(self.language).format_string,
          ML_UInt32.get_display_format(self.language).format_string,
          errorPrecision.get_display_format(self.language).format_string,
          ML_UInt32.get_display_format(self.language).format_string,
          errorPrecision.get_display_format(self.language).pre_process_fct("{0}"),
          ML_UInt32.get_display_format(self.language).pre_process_fct("{2}"),
          errorPrecision.get_display_format(self.language).pre_process_fct("{1}"),
          ML_UInt32.get_display_format(self.language).pre_process_fct("{3}")
        )
        printf_error_op = TemplateOperatorFormat(printf_error_template, arity=4, void_function=True, require_header=["stdio.h"])
        printf_error_function = FunctionObject("printf", [errorPrecision, errorPrecision, ML_UInt32, ML_UInt32], ML_Void, printf_error_op)
  
        local_inputs = [
          Variable(
            "vec_x_{}".format(i) ,
            precision=self.implementation.get_input_format(i),
            var_type=Variable.Local
          ) for i in range(self.arity)
        ]
        # table to store the function results, before comparing them to the expected values
        resultTable = ML_NewTable(
            dimensions=[test_num],
            const=False,
            empty=True,    # result table is initially empty
            storage_precision = self.get_output_precision(),
            tag=self.uniquify_name("max_error_result_table")
        )

        loop_increment = 1
        test_num_cst = Constant(test_num, precision=ML_UInt32)
        NUM_INPUT_ARRAY = len(input_tables)

        # computing results
        futRun = Statement(tested_function(resultTable, *tuple(input_tables), test_num_cst))

        # error evaluation loop
        vi = Variable("i", precision=ML_UInt32, var_type=Variable.Local)
        maxAbsErrIndex = Variable("maxAbsErrIndex", precision=ML_UInt32, var_type=Variable.Local)
        maxRelErrIndex = Variable("maxRelErrIndex", precision=ML_UInt32, var_type=Variable.Local)
        # inputs for the (vi)-th entry of the sub-arrat
        local_inputs = tuple(TableLoad(input_tables[in_id], vi, precision=self.precision, tag = "local_inputs") for in_id in range(NUM_INPUT_ARRAY))
        # expected values for the (vi)-th entry of the sub-arrat
        expected_values = [TableLoad(output_table, vi, i, precision=outType, tag="expected_%d" % i)  for i in range(outAccuracy.get_num_output_value())]
        # local result for the (vi)-th entry of the sub-arrat
        local_result = TableLoad(resultTable, vi, precision=self.get_output_precision(), tag="local_result")

        local_error_relative, localErrorValidty = outAccuracy.compute_error(local_result, expected_values, relative=True)
        local_error_absolute, _ = outAccuracy.compute_error(local_result, expected_values, relative=False)

        assignation_statement = Statement(local_result,
                                          local_error_relative,
                                          local_error_absolute)
  
        comp_statement = Statement()
        # TODO/FIXME: cleanup error value selection when
        # - error is NaN
        # - result is NaN
        # - one of the operand is NaN
        # TODO: in particular check compute_error behavior on NaNs
        newRelErr = Select(
                Test(local_error_relative, specifier=Test.IsNaN),
                # force max_error_relative if local_error_relative is equal to NaN
                # to ensure max error is never a NaN
                # max-error is only valid for non-special cases
                max_error_relative,
                Max(
                  local_error_relative,
                  max_error_relative,
                  precision=errorPrecision),
                  precision=errorPrecision, tag="newRelErr")
        newAbsErr = Select(
                Test(local_error_absolute, specifier=Test.IsNaN),
                max_error_absolute,
                Max(
                  local_error_absolute,
                  max_error_absolute,
                  precision=errorPrecision),
                precision=errorPrecision, tag="newAbsErr")
             
        def EqualAndNotNaN(lhs, rhs):
            return LogicalAnd(Equal(lhs, rhs),
                              LogicalAnd(LogicalNot(Test(lhs, specifier=Test.IsNaN)),
                                         LogicalNot(Test(rhs, specifier=Test.IsNaN))))
        comp_statement.add(ReferenceAssign(maxRelErrIndex, Select(EqualAndNotNaN(max_error_relative, newRelErr), maxRelErrIndex, vi)))
        comp_statement.add(ReferenceAssign(max_error_relative, newRelErr))
        comp_statement.add(ReferenceAssign(maxAbsErrIndex, Select(EqualAndNotNaN(max_error_absolute, newAbsErr), maxAbsErrIndex, vi)))
        comp_statement.add(ReferenceAssign(max_error_absolute, newAbsErr))
  
        error_loop = Loop(
          Statement(
            ReferenceAssign(vi, Constant(0, precision = ML_Int32)),
            ReferenceAssign(maxAbsErrIndex, Constant(0, precision = ML_Int32)),
            ReferenceAssign(maxRelErrIndex, Constant(0, precision = ML_Int32)),
          ),
          vi < test_num_cst,
          Statement(
            assignation_statement,
            comp_statement,
            ReferenceAssign(vi, vi + loop_increment)
          ),
        )
        main_statement = Statement(
            ReferenceAssign(max_error_absolute, Constant(-1, precision=errorPrecision)),
            ReferenceAssign(max_error_relative, Constant(-1, precision=errorPrecision)),
            futRun,
            error_loop,
            printf_error_function(max_error_absolute, max_error_relative, maxAbsErrIndex, maxRelErrIndex),
            Return(max_error_relative))
  
          
        # TODO/FIXME: optimize
        expandPass = Pass_ExpandMultiPrecision(self.processor)
        expandPass.execute(main_statement)
        assert not main_statement is None
  
        return main_statement
  