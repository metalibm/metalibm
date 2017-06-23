# -*- coding: utf-8 -*-

""" Unitary bench to infer micro-architecture
    performance measurement """

import random

import sollya

from sollya import (
    S2, Interval, ceil, floor, inf, sup, pi, log, exp, cos, sin,
    guessdegree, dirtyinfnorm
)

from metalibm_core.core.ml_function import (
    ML_Function, ML_FunctionBasis, DefaultArgTemplate
)

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_table import ML_NewTable
from metalibm_core.code_generation.generator_utility import (
    FunctionOperator, FO_Result, FO_Arg
)

from metalibm_core.utility.ml_template import (
    ML_NewArgTemplate, ArgDefault, precision_parser
)
from metalibm_core.utility.log_report import Log
from metalibm_core.utility.debug_utils import *
from metalibm_core.utility.num_utils import ulp


class OpUnitBench(object):
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
        self.op_class = op_class
        self.op_arity = op_arity
        self.init_interval = init_interval
        self.renorm_function = renorm_function
        self.output_precision = output_precision
        self.input_precisions = input_precisions
        self.bench_name = bench_name

    # generate bench
    def generate_bench(self, processor, test_num=1000, unroll_factor=10):
        initial_inputs = [
            Constant(
                random.uniform(
                    inf(self.init_interval),
                    sup(self.init_interval)
                ), precision=precision
            ) for i, precision in enumerate(self.input_precisions)
        ]

        var_inputs = [
            Variable("var_%d" % i, precision=precision, var_type=Variable.Local)
            for i, precision in enumerate(self.input_precisions)
        ]

        printf_timing_op = FunctionOperator(
            "printf",
            arg_map={
                0: "\"%s[%s] %%lld elts computed "\
                   "in %%lld cycles => %%.3f CPE \\n\"" %
                (
                    self.bench_name,
                    self.output_precision.get_display_format()
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
        init_assign = Statement()
        for var_input, init_value in zip(var_inputs, initial_inputs):
            init_assign.push(ReferenceAssign(var_input, init_value))

        # test loop
        vi = Variable("i", precision=ML_Int64, var_type=Variable.Local)
        test_num_cst = Constant(
            test_num / unroll_factor,
            precision=ML_Int64,
            tag="test_num"
        )

        # Goal build a chain of dependant operation to measure
        # elementary operation latency
        local_inputs = tuple(var_inputs)
        local_result = self.op_class(
            *local_inputs, precision=self.output_precision)
        input_list = var_inputs
        for i in xrange(unroll_factor - 1):
            local_inputs = tuple([local_result] + var_inputs[1:])
            local_result = self.op_class(
                *local_inputs, precision=self.output_precision)
        # renormalisation
        local_result = self.renorm_function(local_result)

        # variable assignation to build dependency chain
        var_assign = Statement()
        var_assign.push(ReferenceAssign(var_inputs[0], local_result))
        final_value = var_inputs[0]

        # loop increment value
        loop_increment = 1

        test_loop = Loop(
            ReferenceAssign(vi, Constant(0, precision=ML_Int32)),
            vi < test_num_cst,
            Statement(
                var_assign,
                ReferenceAssign(vi, vi + loop_increment)
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


OPERATOR_BENCH_LIST = [
    lambda precision:
    OpUnitBench(Addition, "Addition %s" % precision, 2, Interval(-1, 1),
                output_precision=precision, input_precisions=[precision] * 2),
    lambda precision:
    OpUnitBench(Subtraction, "Subtraction %s" % precision, 2, Interval(-1, 1),
                output_precision=precision, input_precisions=[precision] * 2),
    lambda precision:
    OpUnitBench(
        Multiplication, "Multiplication %s" % precision, 2,
        Interval(0.9999, 1.0001),
        output_precision=precision, input_precisions=[precision] * 2
    ),
    lambda precision:
    OpUnitBench(Division, "Division %s" %
                precision, 2, Interval(0.9999, 1.0001)),
]

INT_OPERATOR_BENCH_LIST = [
    lambda precision:
    OpUnitBench(Addition, "Addition %s" % precision, 2, Interval(-1000, 100),
                output_precision=precision, input_precisions=[precision] * 2),
    lambda precision:
    OpUnitBench(
        Subtraction, "Subtraction %s" % precision, 2, Interval(-1000, 1000),
        output_precision=precision, input_precisions=[precision] * 2
    ),
    lambda precision:
    OpUnitBench(
        Multiplication, "Multiplication %s" % precision, 2,
        Interval(-1000, 1000), output_precision=precision,
        input_precisions=[precision] * 2
    ),
    lambda precision:
    OpUnitBench(
        Division, "Division %s" % precision, 2,
        Interval(
            - S2** (precision.get_bit_size() - 1),
              S2**(precision.get_bit_size() - 1)
        )
    ),
]


class ML_UnitBench(ML_Function("ml_external_bench")):
    """ Implementation of external bench function wrapper """

    def __init__(self,
                 arg_template=DefaultArgTemplate,
                 output_file="bench.c",
                 function_name="main",
                 ):
        arity = 0

        # initializing base class
        ML_FunctionBasis.__init__(self,
                                  base_name="bench",
                                  function_name=function_name,
                                  output_file=output_file,
                                  arity=arity,
                                  arg_template=arg_template
                                  )
        # number of basic iteration
        self.test_num = arg_template.test_num
        self.unroll_factor = arg_template.unroll_factor


    @staticmethod
    def get_default_args(**kw):
        default_values = {
            "precision": ML_Int32,
        }
        default_values.update(kw)
        return DefaultArgTemplate(**default_values)

    def generate_scheme(self):
        operation = Multiplication
        function_name = "Addition"
        precision = ML_Binary32
        arity = 2
        unroll_factor = self.unroll_factor
        test_num = self.test_num

        bench_statement = Statement()
        # floating-point bench
        for precision in [ML_Binary32, ML_Binary64]:
            for op_bench in OPERATOR_BENCH_LIST:
                bench_statement.add(op_bench(precision).generate_bench(
                    self.processor, test_num, unroll_factor))
        # integer bench
        for precision in [ML_Int32, ML_Int64]:
            for op_bench in INT_OPERATOR_BENCH_LIST:
                bench_statement.add(op_bench(precision).generate_bench(
                    self.processor, test_num, unroll_factor))
        bench_statement.add(Return(0))

        return bench_statement



if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(
        default_function_name="main", default_output_file="bench.c",
        default_arg=ML_UnitBench.get_default_args())

    def precision_list_parser(s):
        return [precision_parser(p) for p in s.split(",")]

    # argument extraction
    arg_template.get_parser().add_argument(
        "--test-num", dest="test_num", default=10000,
        action="store", type=int, help="number of basic iteration"
    )
    arg_template.get_parser().add_argument(
        "--unroll-factor", dest="unroll_factor",
        default=10, action="store", type=int, help="number of basic iteration"
    )

    args = arg_template.arg_extraction()

    ml_unit_bench = ML_UnitBench(args)
    ml_unit_bench.gen_implementation()
