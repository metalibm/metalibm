# -*- coding: utf-8 -*-

###############################################################################
#
# Copyright (c) 2018 Kalray
#
###############################################################################
# last-modified:        Mar    7th, 2018
# Author(s): Nicolas Brunie <nbrunie@kalray.eu>
###############################################################################
import sys
import random
import math

from enum import Enum

import sollya
import operator

from sollya import Interval, floor, round, log2
from sollya import parse as sollya_parse
S2 = sollya.SollyaObject(2)

from metalibm_core.core.attributes import ML_Debug
from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *
from metalibm_core.core.ml_table import ML_Table
from metalibm_core.code_generation.vhdl_backend import VHDLBackend
from metalibm_core.core.polynomials import *
from metalibm_core.core.ml_entity import ML_Entity, ML_EntityBasis, DefaultEntityArgTemplate
from metalibm_core.code_generation.generator_utility import FunctionOperator, FO_Result, FO_Arg

from metalibm_core.core.advanced_operations import FixedPointPosition

from metalibm_core.core.random_gen import FPRandomGen
from metalibm_core.core.hdl_legalizer import (
        mantissa_extraction_modifier_from_fields, raw_fp_field_extraction
)

from metalibm_core.utility.ml_template import *
from metalibm_core.utility.log_report    import Log
from metalibm_core.utility.debug_utils import *
from metalibm_core.utility.num_utils     import ulp
from metalibm_core.utility.gappa_utils import is_gappa_installed

from metalibm_core.core.special_values import (
        is_number, FP_SpecialValue, FP_PlusInfty, FP_MinusInfty, FP_QNaN,
        FP_PlusZero, FP_MinusZero,
        FP_PlusOmega, FP_MinusOmega,
        is_zero, is_minus_zero, is_plus_zero,
        is_snan, is_infty, is_nan, is_qnan,
        is_numeric_value,
)

from metalibm_core.core.ml_hdl_operations import (
        equal_to, logical_reduce, logical_or_reduce, logical_and_reduce
)

from metalibm_core.core.ml_hdl_format import *
from metalibm_core.core.ml_hdl_operations import *

from metalibm_core.utility.rtl_debug_utils import (
        debug_fixed, debug_dec, debug_std, debug_dec_unsigned
)
from metalibm_core.utility.ml_template import hdl_precision_parser

from metalibm_core.targets.kalray.k1c_fp_utils import (
        rnd_mode_format, rnd_rne, rnd_ru, rnd_rd, rnd_rz
)

from metalibm_hw_blocks.rtl_blocks import zext, rzext
from metalibm_hw_blocks.lzc import ML_LeadingZeroCounter
from metalibm_hw_blocks.lza import ML_LeadingZeroAnticipator

class OpInput(object):
    def __init__(self, precision):
        self.precision = precision

class MultInput:
    def __init__(self, lhs_precision, rhs_precision):
        self.lhs_precision = lhs_precision
        self.rhs_precision = rhs_precision

    @staticmethod
    def parse(s):
        if "x" in s:
            lhs, rhs = s.split("x")
            return MultInput(
                hdl_precision_parser(lhs),
                hdl_precision_parser(rhs)
            )
        else:
            return OpInput(hdl_precision_parser(s))

def multiplication_descriptor_parser(arg_str):
    return [MultInput.parse(s) for s in arg_str.split("+")]

def comp_3to2(a, b, c):
    """ 3 digits to 2 digits compressor """
    #full = Addition(a, b, c, precision=ML_StdLogicVectorFormat(2))
    #carry = BitSelection(full, 1)
    #digit = BitSelection(full, 0)
    #return carry, digit
    s = BitLogicXor(a, BitLogicXor(b, c, precision=ML_StdLogic), precision=ML_StdLogic)
    c = BitLogicOr(
        BitLogicAnd(a, b, precision=ML_StdLogic),
        BitLogicOr(
            BitLogicAnd(a, c, precision=ML_StdLogic),
            BitLogicAnd(c, b, precision=ML_StdLogic),
            precision=ML_StdLogic
        ),
        precision=ML_StdLogic
    )
    return c, s

    a = TypeCast(a, precision=fixed_point(1, 0, signed=False))
    b = TypeCast(b, precision=fixed_point(1, 0, signed=False))
    c = TypeCast(c, precision=fixed_point(1, 0, signed=False))

    full = TypeCast(Conversion(a + b + c, precision=fixed_point(2, 0, signed=False)), precision=ML_StdLogicVectorFormat(2))
    carry = BitSelection(full, 1)
    digit = BitSelection(full, 0)
    return carry, digit

def comp_4to2(cin, a, b, c, d):
    """ 4:2 digit compressor """
    cout, s0 = comp_3to2(a, b, c)
    if cin is None:
        c1 = BitLogicAnd(d, s0, precision=ML_StdLogic)
        s1 = BitLogicXor(d, s0, precision=ML_StdLogic)
    else:
        c1, s1 = comp_3to2(cin, d, s0)
    return cout, c1, s1

def wallace_4to2_reduction(previous_bit_heap):
    """ BitHeap Wallace reduction using 4:2 compressors """
    next_bit_heap = BitHeap()
    carry_bit_heap = BitHeap()
    while previous_bit_heap.max_count() > 0:
        bit_list, w = previous_bit_heap.pop_lower_bits(4)
        if carry_bit_heap.bit_count(w) > 0:
            cin = carry_bit_heap.pop_bit(w)
        else:
            cin = None
        if len(bit_list) == 0:
            if cin:
                next_bit_heap.insert(w, cin)
        elif len(bit_list) == 1:
            next_bit_heap.insert_bit(w, bit_list[0])
            if cin:
                next_bit_heap.insert_bit(w, cin)
        elif len(bit_list) == 2:
            if cin is None:
                for b in bit_list:
                    next_bit_heap.insert_bit(w, b)
            else:
                b_wp1, b_w = comp_3to2(bit_list[0], bit_list[1], cin)
                next_bit_heap.insert_bit(w + 1, b_wp1)
                next_bit_heap.insert_bit(w, b_w)
                cin = None
        elif len(bit_list) == 3:
            if cin:
                next_bit_heap.insert_bit(w, cin)
            b_wp1, b_w = comp_3to2(bit_list[0], bit_list[1], bit_list[2])
            next_bit_heap.insert_bit(w + 1, b_wp1)
            next_bit_heap.insert_bit(w, b_w)
        else:
            assert len(bit_list) == 4
            cout, b_wp1, b_w = comp_4to2(cin, bit_list[0], bit_list[1], bit_list[2], bit_list[3])
            next_bit_heap.insert_bit(w + 1, b_wp1)
            next_bit_heap.insert_bit(w, b_w)
            carry_bit_heap.insert_bit(w + 1, cout)
    # flush carry-bit heap
    while carry_bit_heap.max_count() > 0:
        bit_list, w = carry_bit_heap.pop_lower_bits(1)
        next_bit_heap.insert_bit(w, bit_list[0])
    return next_bit_heap

def dadda_4to2_reduction(previous_bit_heap):
    """ BitHeap Wallace reduction using 4:2 compressors """
    next_bit_heap = BitHeap()
    carry_bit_heap = BitHeap()
    max_count = previous_bit_heap.max_count()
    new_count = int(math.ceil(max_count / 2.0))
    # each step reduce the height of the bit heap at at most
    # new_count. However it is not necessary to reduce over it
    while previous_bit_heap.max_count() > 0:
        bit_list, w = previous_bit_heap.pop_lower_bits(4)
        # if a carry frmo this weight exists, we must try to 
        # accumulate it
        if carry_bit_heap.bit_count(w) > 0:
            cin = carry_bit_heap.pop_bit(w)
        else:
            cin = None
        if len(bit_list) == 0:
            if cin:
                next_bit_heap.insert(w, cin)
        elif len(bit_list) == 1:
            next_bit_heap.insert_bit(w, bit_list[0])
            if cin:
                next_bit_heap.insert_bit(w, cin)
        elif (0 if cin is None else 1) + previous_bit_heap.bit_count(w) + len(bit_list) + next_bit_heap.bit_count(w) <= new_count:
            # drop every bit in next stage
            if not cin is None:
                next_bit_heap.insert_bit(w, cin)
            for b in bit_list:
                next_bit_heap.insert_bit(w, b)
        elif len(bit_list) == 2:
            if cin is None:
                for b in bit_list:
                    next_bit_heap.insert_bit(w, b)
            else:
                b_wp1, b_w = comp_3to2(bit_list[0], bit_list[1], cin)
                next_bit_heap.insert_bit(w + 1, b_wp1)
                next_bit_heap.insert_bit(w, b_w)
                cin = None
        elif len(bit_list) == 3:
            if cin:
                next_bit_heap.insert_bit(w, cin)
            b_wp1, b_w = comp_3to2(bit_list[0], bit_list[1], bit_list[2])
            next_bit_heap.insert_bit(w + 1, b_wp1)
            next_bit_heap.insert_bit(w, b_w)
        else:
            assert len(bit_list) == 4
            cout, b_wp1, b_w = comp_4to2(cin, bit_list[0], bit_list[1], bit_list[2], bit_list[3])
            next_bit_heap.insert_bit(w + 1, b_wp1)
            next_bit_heap.insert_bit(w, b_w)
            carry_bit_heap.insert_bit(w + 1, cout)
    # flush carry-bit heap
    while carry_bit_heap.max_count() > 0:
        bit_list, w = carry_bit_heap.pop_lower_bits(1)
        next_bit_heap.insert_bit(w, bit_list[0])
    return next_bit_heap

def wallace_reduction(previous_bit_heap):
    """ Partial Product Tree compression using Wallace Algorithm
        and 3:2 compressor """
    next_bit_heap = BitHeap()
    while previous_bit_heap.max_count() > 0:
        bit_list, w = previous_bit_heap.pop_lower_bits(3)
        if len(bit_list) <= 2:
            for b in bit_list:
                next_bit_heap.insert_bit(w, b)
        else:
            b_wp1, b_w = comp_3to2(bit_list[0], bit_list[1], bit_list[2])
            next_bit_heap.insert_bit(w + 1, b_wp1)
            next_bit_heap.insert_bit(w, b_w)
    return next_bit_heap

def dadda_reduction(previous_bit_heap):
    """ Dadda reduction for partial product tree using 3:2 compressors """
    next_bit_heap = BitHeap()
    max_count = previous_bit_heap.max_count()
    new_count = int(math.ceil((max_count / 3.0) * 2))
    while previous_bit_heap.max_count() > 0:
        bit_list, w = previous_bit_heap.pop_lower_bits(3)
        if len(bit_list) <= 2 or previous_bit_heap.bit_count(w) + len(bit_list) + next_bit_heap.bit_count(w) <= new_count:
            for b in bit_list:
                next_bit_heap.insert_bit(w, b)
        else:
            b_wp1, b_w = comp_3to2(bit_list[0], bit_list[1], bit_list[2])
            next_bit_heap.insert_bit(w + 1, b_wp1)
            next_bit_heap.insert_bit(w, b_w)
    return next_bit_heap

class ReductionMethod(Enum):
    Wallace = "wallace"
    Dadda = "dadda"
    Wallace_4to2 = "wallace_4to2"
    Dadda_4to2 = "dadda_4to2"

    def __str__(self):
        return self.value

REDUCTION_METHOD_MAP = {
    ReductionMethod.Wallace: wallace_reduction,
    ReductionMethod.Dadda: dadda_reduction,
    ReductionMethod.Wallace_4to2: wallace_4to2_reduction,
    ReductionMethod.Dadda_4to2: dadda_4to2_reduction,
}


class BitHeap:
    def __init__(self):
        self.heap = {}
        self.count = {}

    def insert_bit(self, index, value):
        #print "inserting bit {} with weight {}".format(value, index)
        if not index in self.heap:
            self.heap[index] = []
            self.count[index] = 0
        self.heap[index].append(value)
        self.count[index] += 1

    def max_count(self):
        """ @return the maximum value stored in self.count dict """
        if len(self.count) == 0:
            return 0
        else:
            return max(self.count.values())

    @property
    def max_index(self):
        return max(self.count)
    @property
    def min_index(self):
        return min(self.count)

    def pop_bit(self, index, pos=0):
        if not index in self.heap:
            return None
        bit = self.heap[index].pop(pos)
        self.count[index] -= 1
        if self.count[index] == 0:
            self.heap.pop(index)
            self.count.pop(index)
        #print "popping bit {} from weight {}".format(bit, index)
        return bit

    def bit_count(self, index):
        if index in self.count:
            return self.count[index]
        else:
            return 0

    def pop_bits(self, index, max_num=1):
        if not index in self.count:
            return []
        else:
            pop_len = min(max_num, self.count[index])
            result = [self.pop_bit(index) for i in range(pop_len)]
            return result

    def pop_lower_bits(self, max_num=1):
        """ try to pop @p max_num bits from the lowest
            index in bit_heap 
            @return list of bits, weigth """
        lower_index = min(self.count)
        assert self.count[lower_index] > 0
        return self.pop_bits(lower_index, max_num), lower_index




class MultArray(ML_Entity("mult_array")):
    def __init__(self,
                 arg_template = DefaultEntityArgTemplate,
                 precision = ML_Binary32,
                 accuracy    = ML_Faithful,
                 debug_flag = False,
                 target = VHDLBackend(),
                 output_file = "mult_array.vhd",
                 entity_name = "mult_array",
                 language = VHDL_Code,
                 acc_prec = None,
                 pipelined = False):
        # initializing I/O precision
        precision = ArgDefault.select_value([arg_template.precision, precision])
        io_precisions = [precision] * 2

        # initializing base class
        ML_EntityBasis.__init__(self,
            base_name = "mult_array",
            entity_name = entity_name,
            output_file = output_file,

            io_precisions = io_precisions,
            abs_accuracy = None,

            backend = target,

            debug_flag = debug_flag,
            language = language,
            arg_template = arg_template
        )

        self.accuracy    = accuracy
        # main precision (used for product operand and default for accumulator)
        self.precision = precision
        # enable operator pipelining
        self.pipelined = pipelined
        # multiplication input descriptor
        self.op_expr = arg_template.op_expr
        self.dummy_mode = arg_template.dummy_mode
        # reduction method
        self.reduction_method = arg_template.method

    ## default argument template generation
    @staticmethod
    def get_default_args(**kw):
        default_dict = {
            "precision": fixed_point(32,0),
            "target": VHDLBackend(),
            "output_file": "mult_array.vhd",
            "entity_name": "mult_array",
            "language": VHDL_Code,
            "Method": ReductionMethod.Wallace_4to2,
            "pipelined": False,
            "dummy_mode": False,
            "passes": [
                ("beforepipelining:size_datapath"),
                ("beforepipelining:rtl_legalize"),
                ("beforepipelining:unify_pipeline_stages"),
                #("beforecodegen:dump"),
                ],
        }
        default_dict.update(kw)
        return DefaultEntityArgTemplate(
            **default_dict
        )

    def generate_scheme(self):
        if self.dummy_mode:
            return self.generate_dummy_scheme()
        else:
            return self.generate_advanced_scheme()


    def generate_dummy_scheme(self):
        Log.report(
            Log.Info,
            "generating MultArray with output precision {precision}".format(
                precision = self.precision))

        acc = None
        a_inputs = {}
        b_inputs = {}

        for index, mult_input in enumerate(self.op_expr):
            print "{} x {}".format(mult_input.lhs_precision, mult_input.rhs_precision)
            #lhs_precision = fixed_point(mult_input.lhs_precision.get_integer_size(), mult_input.lhs_precision.get_frac_size(), signed=mult_input.lhs_precision.signed)
            #rhs_precision = fixed_point(mult_input.rhs_precision.get_integer_size(), mult_input.rhs_precision.get_frac_size(), signed=mult_input.rhs_precision.signed)
            #mult_input.lhs_precision = lhs_precision
            #mult_input.rhs_precision = rhs_precision

        for index, mult_input in enumerate(self.op_expr):
            a_i = self.implementation.add_input_signal("a_%d_i" % index, mult_input.lhs_precision)
            b_i = self.implementation.add_input_signal("b_%d_i" % index, mult_input.rhs_precision)
            a_inputs[index] = a_i
            b_inputs[index] = b_i

            if acc is None:
                acc = a_i * b_i
            else:
                acc = acc + a_i * b_i

        result = Conversion(acc, precision=self.precision)
        self.implementation.add_output_signal("result_o", result)

        return [self.implementation]

    def generate_advanced_scheme(self):
        ## Generate Fused multiply and add comput <x> . <y> + <z>
        Log.report(
            Log.Info,
            "generating MultArray with output precision {precision}".format(
                precision = self.precision))

        acc = None


        # fixing precision
        for index, operation in enumerate(self.op_expr):
            if isinstance(operation, MultInput):
                print "{} x {}".format(operation.lhs_precision, operation.rhs_precision)
            elif isinstance(operation, OpInput):
                print " + {}".format(operation.precision)
            # lhs_precision = fixed_point(mult_input.lhs_precision.get_integer_size(), mult_input.lhs_precision.get_frac_size(), signed=mult_input.lhs_precision.signed)
            # rhs_precision = fixed_point(mult_input.rhs_precision.get_integer_size(), mult_input.rhs_precision.get_frac_size(), signed=mult_input.rhs_precision.signed)
            # mult_input.lhs_precision = lhs_precision
            # mult_input.rhs_precision = rhs_precision

        product_inputs = []
        addition_inputs = []

        # generating input signals
        for index, operand_input in enumerate(self.op_expr):
            if isinstance(operand_input, MultInput):
                a_i = self.implementation.add_input_signal("a_%d_i" % index, operand_input.lhs_precision)
                b_i = self.implementation.add_input_signal("b_%d_i" % index, operand_input.rhs_precision)
                product_inputs.append((a_i, b_i))
            elif isinstance(operand_input, OpInput):
                c_i = self.implementation.add_input_signal("c_%d_i" % index, operand_input.precision)
                addition_inputs.append(c_i)


        # heap of positive bits
        pos_bit_heap = BitHeap()
        # heap of negative bits
        neg_bit_heap = BitHeap()

        # Partial Product generation
        for product in product_inputs:
            a_i, b_i = product
            a_i_precision = a_i.get_precision()
            b_i_precision = b_i.get_precision()
            a_i_signed = a_i_precision.get_signed()
            b_i_signed = b_i.get_precision().get_signed()
            unsigned_prod = not(a_i_signed) and not(b_i_signed)
            a_i_size = a_i_precision.get_bit_size()
            b_i_size = b_i_precision.get_bit_size()
            for pp_index in range(a_i_size):
                a_j_signed = a_i_signed and (pp_index == a_i_size - 1) 
                bit_a_j = BitSelection(a_i, pp_index)
                pp = Select(equal_to(bit_a_j, 1), b_i, 0)
                offset = pp_index - a_i_precision.get_frac_size()
                for b_index in range(b_i_size):
                    b_k_signed = b_i_signed and (b_index == b_i_size - 1)
                    pp_signed = a_j_signed ^ b_k_signed
                    pp_weight = offset + b_index
                    local_bit = BitSelection(pp, b_index)
                    if pp_signed:
                        neg_bit_heap.insert_bit(pp_weight, local_bit)
                    else:
                        pos_bit_heap.insert_bit(pp_weight, local_bit)



        STAGE_LEVEL_LIMIT = 8
        # Partial Product reduction
        while pos_bit_heap.max_count() > STAGE_LEVEL_LIMIT:
            pos_bit_heap = REDUCTION_METHOD_MAP[self.reduction_method](pos_bit_heap)
        while neg_bit_heap.max_count() > STAGE_LEVEL_LIMIT:
            neg_bit_heap = REDUCTION_METHOD_MAP[self.reduction_method](neg_bit_heap)

        if self.pipelined:
            self.implementation.start_new_stage()

        for add_op in addition_inputs:
            precision = add_op.get_precision()
            size = precision.get_bit_size()
            offset = -precision.get_frac_size()
            # most significant bit
            if precision.get_signed():
                neg_bit_heap.insert_bit(size -1 + offset, BitSelection(add_op, size - 1))
            else:
                pos_bit_heap.insert_bit(size -1 + offset, BitSelection(add_op, size - 1))
            # any other bit
            for index in range(size - 1):
                pos_bit_heap.insert_bit(index + offset, BitSelection(add_op, index))


        # Partial Product reduction
        while pos_bit_heap.max_count() > 2:
            pos_bit_heap = REDUCTION_METHOD_MAP[self.reduction_method](pos_bit_heap)
        while neg_bit_heap.max_count() > 2:
            neg_bit_heap = REDUCTION_METHOD_MAP[self.reduction_method](neg_bit_heap)



        def convert_bit_heap_to_fixed_point(current_bit_heap, signed=False):
            # final propagating sum
            op_index = 0
            op_list = []
            op_statement = Statement()
            while current_bit_heap.max_count() > 0:
                op_size = current_bit_heap.max_index - current_bit_heap.min_index + 1
                op_format = ML_StdLogicVectorFormat(op_size)
                op_reduce = Signal("op_%d" % op_index, precision=op_format, var_type=Variable.Local)

                offset_index = current_bit_heap.min_index

                for index in range(current_bit_heap.min_index, current_bit_heap.max_index + 1):
                    out_index = index - offset_index
                    bit_list = current_bit_heap.pop_bits(index, 1)
                    if len(bit_list) == 0:
                        op_statement.push(ReferenceAssign(BitSelection(op_reduce, out_index), Constant(0, precision=ML_StdLogic)))
                    else:
                        assert len(bit_list) == 1
                        op_statement.push(ReferenceAssign(BitSelection(op_reduce, out_index), bit_list[0]))

                op_precision = fixed_point(op_size + offset_index, -offset_index, signed=signed)
                op_list.append(
                    PlaceHolder(
                        TypeCast(
                            op_reduce,
                            precision=op_precision),
                        op_statement
                    )
                )
                op_index += 1

            return op_list, op_statement

        pos_op_list, pos_assign_statement = convert_bit_heap_to_fixed_point(pos_bit_heap, signed=False)
        neg_op_list, neg_assign_statement = convert_bit_heap_to_fixed_point(neg_bit_heap, signed=False)

        # a PlaceHolder is inserted to force forwarding of op_statement
        # which will be removed otherwise as it does not appear anywhere in
        # the final operation graph
        acc = None
        if len(pos_op_list) > 0:
            reduced_pos_sum = reduce(operator.__add__, pos_op_list)
            reduced_pos_sum.set_attributes(tag="reduced_pos_sum", debug=debug_fixed)
            pos_acc = PlaceHolder(reduced_pos_sum, pos_assign_statement)
            acc = pos_acc
        if len(neg_op_list) > 0:
            reduced_neg_sum = reduce(operator.__add__, neg_op_list)
            reduced_neg_sum.set_attributes(tag="reduced_neg_sum", debug=debug_fixed)
            neg_acc = PlaceHolder(reduced_neg_sum, neg_assign_statement)
            acc = neg_acc if acc is None else acc - neg_acc

        acc.set_attributes(tag="raw_acc", debug=debug_fixed)

        self.precision = fixed_point(
            self.precision.get_integer_size(),
            self.precision.get_frac_size(),
            signed=self.precision.get_signed()
        )
        result = Conversion(acc, tag="result", precision=self.precision, debug=debug_fixed)
        self.implementation.add_output_signal("result_o", result)

        return [self.implementation]

    @property
    def standard_test_cases(self):
        test_case_max = {}
        test_case_min = {}
        for index, operation in enumerate(self.op_expr):
            if isinstance(operation, MultInput):
                test_case_max["a_%d_i" % index] = operation.lhs_precision.get_max_value()
                test_case_max["b_%d_i" % index] = operation.rhs_precision.get_max_value()

                test_case_min["a_%d_i" % index] = mult_input.lhs_precision.get_min_value()
                test_case_min["b_%d_i" % index] = mult_input.rhs_precision.get_min_value()
            elif isinstance(operation, OpInput):
                test_case_max["c_%d_i" % index] = operation.precision.get_max_value()

                test_case_min["c_%d_i" % index] = operation.precision.get_min_value()
            else:
                raise NotImplementedError

        return [(test_case_max, None), (test_case_min, None)]


    def numeric_emulate(self, io_map):
        acc = 0
        for index, operation in enumerate(self.op_expr):
            if isinstance(operation, MultInput):
                a_i = io_map["a_%d_i" % index]
                b_i = io_map["b_%d_i" % index]
                acc += a_i * b_i
            elif isinstance(operation, OpInput):
                c_i = io_map["c_%d_i" % index]
                acc += c_i

        # assert acc >= 0
        return {"result_o": acc}




if __name__ == "__main__":
        # auto-test
        arg_template = ML_EntityArgTemplate(
            default_entity_name="mult_array",
            default_output_file="ml_mult_array.vhd",
            default_arg=MultArray.get_default_args()
        )
        # accumulator precision (also the output format)
        arg_template.parser.add_argument(
            "--mult-desc",
            dest="op_expr",
            type=multiplication_descriptor_parser,
            default=None,
            help="Multiplication Input descriptor")
        arg_template.parser.add_argument(
            "--dummy-mode",
            dest="dummy_mode",
            default=False,
            const=True,
            action="store_const",
            help="select advance/dummy mode")
        arg_template.parser.add_argument(
            "--method",
            type=ReductionMethod,
            default=ReductionMethod.Wallace,
            choices=list(ReductionMethod),
            help="define compression reduction methode"
        )
        # argument extraction
        args = parse_arg_index_list = arg_template.arg_extraction()

        ml_hw_mpfma            = MultArray(args, pipelined=args.pipelined)

        ml_hw_mpfma.gen_implementation()
