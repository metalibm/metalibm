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
# last-modified:    Mar  7th, 2018
###############################################################################
""" Unitary bench to infer micro-architecture
    performance measurement """

import random

from sollya import (
    S2, Interval, inf, sup
)

from metalibm_core.core.ml_function import (
    ML_Function, ML_FunctionBasis, DefaultArgTemplate
)

from metalibm_core.core.ml_operations import (
    Statement, ReferenceAssign, Constant, Loop, Variable,
    FunctionObject, Subtraction, Division, Conversion,
    Return
)

import metalibm_core.core.ml_operations as metaop

from metalibm_core.core.ml_formats import (
    ML_Binary64, ML_Binary32, ML_Int32, ML_Int64, ML_Void,
    FormatAttributeWrapper, ML_Fixed_Format, ML_FP_Format
)
from metalibm_core.code_generation.generator_utility import (
    FunctionOperator, FO_Arg
)

from metalibm_core.utility.ml_template import (
    ML_NewArgTemplate, precision_parser
)


class OpUnitBench(object):
    """ Operation Unitary Bench class """
    # @param op_class Operation class (to build new operation instance)
    #  @param op_arity Number of arguments expected by the operation
    #  @param init_interval Range of the first arguments to the operation
    #  @param renorm_function optree -> optree function renormalizing the
    #         result of a series of Operation to avoid overflow / underflow
    #  @param output_precision format of the operation output
    #  @param input_precisions list of input operand formats
    def __init__(
            self, op_class, bench_name, op_arity=2,
            init_interval=Interval(-0.5, 0.5),
            renorm_function=lambda x: x, output_precision=ML_Binary32,
            input_precisions=[ML_Binary32, ML_Binary32]
        ):
        """ OpUnitBench ctor """
        self.op_class = op_class
        self.op_arity = op_arity
        self.init_interval = init_interval
        self.renorm_function = renorm_function
        self.output_precision = output_precision
        self.input_precisions = input_precisions
        self.bench_name = bench_name


    # generate bench
    def generate_bench(self, processor, test_num=1000, unroll_factor=10):
        """ generate performance bench for self.op_class """
        initial_inputs = [
            Constant(
                random.uniform(
                    inf(self.init_interval),
                    sup(self.init_interval)
                ), precision=precision
            ) for i, precision in enumerate(self.input_precisions)
        ]

        var_inputs = [
            Variable(
                            "var_%d" % i,
                            precision = FormatAttributeWrapper(
                                precision, ["volatile"]
                            ),
                            var_type=Variable.Local
                        )
            for i, precision in enumerate(self.input_precisions)
        ]

        printf_timing_op = FunctionOperator(
            "printf",
            arg_map={
                0: "\"%s[%s] %%\"PRIi64\" elts computed "\
                   "in %%\"PRIi64\" cycles =>\\n     %%.3f CPE \\n\"" %
                (
                    self.bench_name,
                    self.output_precision.get_display_format().format_string
                ),
                1: FO_Arg(0),
                2: FO_Arg(1),
                3: FO_Arg(2),
                4: FO_Arg(3)
            }, void_function=True
        )
        printf_timing_function = FunctionObject(
            "printf",
            [self.output_precision, ML_Int64, ML_Int64, ML_Binary64],
            ML_Void, printf_timing_op
        )
        timer = Variable("timer", precision=ML_Int64, var_type=Variable.Local)

        void_function_op = FunctionOperator(
            "(void)", arity=1, void_function=True)
        void_function = FunctionObject(
            "(void)",
            [self.output_precision],
            ML_Void, void_function_op
        )

        # initialization of operation inputs
        init_assign = metaop.Statement()
        for var_input, init_value in zip(var_inputs, initial_inputs):
            init_assign.push(ReferenceAssign(var_input, init_value))

        # test loop
        loop_i = Variable("i", precision=ML_Int64, var_type=Variable.Local)
        test_num_cst = Constant(
            test_num / unroll_factor,
            precision=ML_Int64,
            tag="test_num"
        )

        # Goal build a chain of dependant operation to measure
        # elementary operation latency
        local_inputs = tuple(var_inputs)
        local_result = self.op_class(
            *local_inputs,
                        precision=self.output_precision,
                        unbreakable = True
                    )
        for i in range(unroll_factor - 1):
            local_inputs = tuple([local_result] + var_inputs[1:])
            local_result = self.op_class(
                *local_inputs, precision=self.output_precision, unbreakable = True
                        )
        # renormalisation
        local_result = self.renorm_function(local_result)

        # variable assignation to build dependency chain
        var_assign = Statement()
        var_assign.push(ReferenceAssign(var_inputs[0], local_result))
        final_value = var_inputs[0]

        # loop increment value
        loop_increment = 1

        test_loop = Loop(
            ReferenceAssign(loop_i, Constant(0, precision=ML_Int32)),
            loop_i < test_num_cst,
            Statement(
                var_assign,
                ReferenceAssign(loop_i, loop_i + loop_increment)
            ),
        )

        # bench scheme
        test_scheme = Statement(
            ReferenceAssign(timer, processor.get_current_timestamp()),
            init_assign,
            test_loop,

            ReferenceAssign(timer,
                            Subtraction(
                                processor.get_current_timestamp(),
                                timer,
                                precision=ML_Int64
                            )
                            ),
            # prevent intermediary variable simplification
            void_function(final_value),
            printf_timing_function(
                final_value,
                Constant(test_num, precision=ML_Int64),
                timer,
                Division(
                    Conversion(timer, precision=ML_Binary64),
                    Constant(test_num, precision=ML_Binary64),
                    precision=ML_Binary64
                )
            )
            # ,Return(Constant(0, precision = ML_Int32))
        )

        return test_scheme

def is_fp_format(precision):
  return isinstance(precision.get_base_format(), ML_FP_Format)
def is_int_format(precision):
  return isinstance(precision.get_base_format(), ML_Fixed_Format)

def predicate_is_fp_op(op_class, output_precision, input_precisions):
  return is_fp_format(output_precision)
def predicate_is_int_op(op_class, output_precision, input_precisions):
  return is_int_format(output_precision)

OPERATOR_BENCH_MAP = {
  metaop.Addition:
    {
    predicate_is_fp_op:
      lambda precision:
      OpUnitBench(
          metaop.Addition, "Addition %s" % precision, 2, Interval(-1, 1),
          output_precision=precision, input_precisions=[precision] * 2
      ),
    predicate_is_int_op:
      lambda precision:
      OpUnitBench(
          metaop.Addition, "Addition %s" % precision, 2, Interval(-1000, 1000),
          output_precision=precision, input_precisions=[precision] * 2
      ),
  },
  metaop.Subtraction: {
    predicate_is_fp_op:
      lambda precision:
      OpUnitBench(
          metaop.Subtraction, "Subtraction %s" % precision, 2, Interval(-1, 1),
          output_precision=precision, input_precisions=[precision] * 2
      ),
    predicate_is_int_op:
      lambda precision:
        OpUnitBench(
            metaop.Subtraction, "Subtraction %s" % precision, 2,
            Interval(-1000, 1000),
            output_precision=precision, input_precisions=[precision] * 2
        ),
  },
  metaop.Multiplication: {
    predicate_is_fp_op:
      lambda precision:
      OpUnitBench(
          metaop.Multiplication, "Multiplication %s" % precision, 2,
          Interval(0.9999, 1.0001),
          output_precision=precision, input_precisions=[precision] * 2
      ),
    predicate_is_int_op:
      lambda precision:
      OpUnitBench(
          metaop.Multiplication, "Multiplication %s" % precision, 2,
          Interval(-1000, 1000),
                  output_precision=precision,
          input_precisions=[precision] * 2
      ),
  },
  metaop.FusedMultiplyAdd: {
    predicate_is_fp_op:
      lambda precision:
      OpUnitBench(
          metaop.FusedMultiplyAdd, "FusedMultiplyAdd %s" % precision, 3,
          Interval(0.9999, 1.0001),
          output_precision=precision, input_precisions=[precision] * 3
      ),
  },
  metaop.Division: {
    predicate_is_fp_op:
      lambda precision:
      OpUnitBench(
          metaop.Division, "Division %s" % precision, 2,
					Interval(0.9999, 1.0001),
					output_precision = precision,
					input_precisions = [precision] * 2
			),
    predicate_is_int_op:
      lambda precision:
      OpUnitBench(
          metaop.Division, "Division %s" % precision, 2,
          Interval(
              - S2** (precision.get_bit_size() - 1),
                S2**(precision.get_bit_size() - 1)
          ),
          output_precision=precision,
          input_precisions=[precision] * 2
      ),
  },
}

class UnitBench(ML_Function("ml_unit_bench")):
    """ Implementation of unitary operation node bench """

    def __init__(self,
                 args=DefaultArgTemplate,
                 ):
        # initializing base class
        ML_FunctionBasis.__init__(self, args=args)

        # number of basic iteration
        self.test_num = args.test_num
        self.unroll_factor = args.unroll_factor
        # dict of operations to be benched
        self.operation_map = args.operation_map

    def get_execute_handle(self):
        return self.function_name

    @staticmethod
    def get_default_args(**kw):
        """ generate default argument structure for OpUnitBench """
        default_values = {
            "precision": ML_Int32,
            "output_file": "unit_bench.c",
            "function_name": "unit_bench",
        }
        default_values.update(kw)
        return DefaultArgTemplate(**default_values)

    def generate_scheme(self):
        """ generate an operation unitary bench test scheme
            (graph of operation implementing latency computation
             on a dependent sequence of self.op_class)"""
        unroll_factor = self.unroll_factor
        test_num = self.test_num

        bench_statement = metaop.Statement()
        # floating-point bench
        for op_class in self.operation_map:
          for output_precision in self.operation_map[op_class]:
            for predicate in OPERATOR_BENCH_MAP[op_class]:
              if predicate(op_class, output_precision, None):
                op_bench = OPERATOR_BENCH_MAP[op_class][predicate]
                bench_statement.add(op_bench(output_precision).generate_bench(
                    self.processor, test_num, unroll_factor))
        bench_statement.add(Return(0))

        return bench_statement


def operation_parser(s):
    """ Convert a string into an operation class

        Args:
            s (str): input string to be converted

        Returns:
            class: child of ML_Operation

        Examples:
            >>> operation_parse("add")
            Addition
    """
    return {
        "add": metaop.Addition,
        "mul": metaop.Multiplication,
        "sub": metaop.Subtraction,
        "div": metaop.Division,
        "fma": metaop.FusedMultiplyAdd
    }[s]

def operation_map_parser(s):
    """ Convert a operation map string description into the corresponding
        dictionnary

        Args:
            s (str): operation map string decription

        Returns:
            dict: map of (operation_class -> list of format)

        Example:
            >>> operation_map_parser("binary32,int32:add,mul;int64:div")
            {
                Addition: [ML_Binary32, ML_Int32],
                Multiplication: [ML_Binary64],
                Division: [ML_Int64]
            }
    """
    op_map = {}
    level0_list = s.split(";")
    for entry in level0_list:
        formats, ops = entry.split(':')
        formats = [precision_parser(f) for f in formats.split(',')]
        ops = [operation_parser(o) for o in ops.split(',')]
        for op in ops:
            if not op in op_map:
                op_map[op] = []
            for f in formats:
                op_map[op].append(f)
    return op_map

if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(
        default_arg=UnitBench.get_default_args())


    # argument extraction
    arg_template.get_parser().add_argument(
        "--test-num", dest="test_num", default=10000,
        action="store", type=int, help="number of basic iteration"
    )
    arg_template.get_parser().add_argument(
        "--unroll-factor", dest="unroll_factor",
        default=10, action="store", type=int, help="number of basic iteration"
    )

    # TODO: on-going
    arg_template.get_parser().add_argument(
       "--operations", dest="operation_map", default="binary64,binary32:add,mul",
        action="store", type=operation_map_parser, help="number of basic iteration"
    )

    ARGS = arg_template.arg_extraction()
    ML_UNIT_BENCH = UnitBench(ARGS)
    ML_UNIT_BENCH.gen_implementation()
