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
import re

from enum import Enum

import sollya
import operator

from sollya import Interval, floor, round, log2
from sollya import parse as sollya_parse
S2 = sollya.SollyaObject(2)

from metalibm_core.core.ml_operations import (
    Addition, Multiplication, BitLogicAnd, BitLogicXor, Select,
    BitLogicOr, Statement, Variable, ReferenceAssign, TypeCast,
    Constant, Conversion,
    LogicalOr, Equal, LogicalAnd, LogicalNot,
    BitLogicNegate,
)
from metalibm_core.code_generation.vhdl_backend import VHDLBackend
from metalibm_core.core.ml_entity import ML_Entity, ML_EntityBasis, DefaultEntityArgTemplate

from metalibm_core.utility.ml_template import (
    ML_EntityArgTemplate
)
from metalibm_core.utility.log_report    import Log

from metalibm_core.core.precisions import ML_Faithful
from metalibm_core.core.ml_formats import ML_Bool


from metalibm_core.core.ml_hdl_format import (
    ML_StdLogicVectorFormat, ML_StdLogic, fixed_point,
)
from metalibm_core.code_generation.code_constant import VHDL_Code
from metalibm_core.core.ml_hdl_operations import (
    BitSelection, Signal, PlaceHolder,
    equal_to, Concatenation, SubSignalSelection
)

from metalibm_core.utility.rtl_debug_utils import (
        debug_fixed, debug_std,
)
from metalibm_core.utility.ml_template import hdl_precision_parser


# re pattern to match format and stage index strings
OP_PATTERN = "(?P<format>F[US]-?\d+\.-?\d+)(\[(?P<stage>\d+)\])?"


class OpInput(object):
    def __init__(self, precision, stage=None):
        self.precision = precision
        self.stage = stage

    def __str__(self):
        return "+ %s[%s]" % (str(self.precision), self.stage)


def extract_format_stage(s):
    group_match = re.match(OP_PATTERN, s)
    precision = group_match.group("format")
    stage = group_match.group("stage")
    return precision, stage if stage is None else int(stage)

class MultInput:
    def __init__(self, lhs_precision, rhs_precision, lhs_stage=None, rhs_stage=None):
        self.lhs_precision = lhs_precision
        self.rhs_precision = rhs_precision
        self.lhs_stage = lhs_stage
        self.rhs_stage = rhs_stage


    def __str__(self):
        return "%s[%s] x %s[%s]" % (str(self.lhs_precision), self.lhs_stage, str(self.rhs_precision), self.rhs_stage)

    @staticmethod
    def parse(s):
        if "x" in s:
            lhs, rhs = s.split("x")
            lhs_format, lhs_stage = extract_format_stage(lhs)
            rhs_format, rhs_stage = extract_format_stage(rhs)
            return MultInput(
                hdl_precision_parser(lhs_format),
                hdl_precision_parser(rhs_format),
                lhs_stage=lhs_stage,
                rhs_stage=rhs_stage
            )
        else:
            precision, stage = extract_format_stage(s)
            return OpInput(hdl_precision_parser(precision), stage)


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
            print "dropping bits without compression"
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


def booth_radix4_multiply(lhs, rhs, pos_bit_heap, neg_bit_heap):
    """ Compute the multiplication @p lhs x @p rhs using radix 4 Booth
        recoding and drop the generated partial product in @p
        pos_bit_heap and @p neg_bit_heap based on their sign """
    # booth recoded partial product for n-th digit
    # is based on digit from n-1 to n+1
    #    (n+1) | (n) | (n-1) |  PP  |
    #    ------|-----|-------|------|
    #      0   |  0  |   0   |  +0  |
    #      0   |  0  |   1   |  +X  |
    #      0   |  1  |   0   |  +X  |
    #      0   |  1  |   1   |  +2x |
    #      1   |  0  |   0   |  -2X |
    #      1   |  0  |   1   |  -X  |
    #      1   |  1  |   0   |  -X  |
    #      1   |  1  |   1   |  +0  |
    #    ------|-----|-------|------|
    assert lhs.get_precision().get_bit_size() >= 2

    # lhs is the recoded operand
    # RECODING DIGITS
    # first recoded digit is padded right by 0
    first_digit = Concatenation(
        SubSignalSelection(lhs, 0, 1, precision=ML_StdLogicVectorFormat(2)),
        Constant(0, precision=ML_StdLogic),
        precision=ML_StdLogicVectorFormat(3),
        debug=debug_std,
        tag="booth_digit_0"
    )
    digit_list = [(first_digit, 0)]

    for digit_index in range(2, lhs.get_precision().get_bit_size(), 2):
        if digit_index + 1 < lhs.get_precision().get_bit_size():
            # digits exist completely in lhs
            digit = SubSignalSelection(lhs, digit_index - 1, digit_index + 1, tag="booth_digit_%d" % digit_index, debug=debug_std)
        else:
            # MSB padding required
            sign_ext = Constant(0, precision=ML_StdLogic) if not(lhs.get_precision().get_signed()) else BitSelection(lhs, lhs.get_precision().get_bit_size() - 1)
            digit = Concatenation(
                sign_ext,
                SubSignalSelection(lhs, digit_index - 1, digit_index),
                precision=ML_StdLogicVectorFormat(3),
                debug=debug_std,
                tag="booth_digit_%d" % digit_index
            )
        digit_list.append((digit, digit_index))
    # if lhs size is a mutiple of two and it is unsigned
    # than an extra digit must be generated to ensure a positive result
    if lhs.get_precision().get_bit_size() % 2 == 0 and not(lhs.get_precision().get_signed()):
        digit_index = lhs.get_precision().get_bit_size() - 1
        digit = Concatenation(
            Constant(0, precision=ML_StdLogicVectorFormat(2)),
            BitSelection(lhs, digit_index),
            precision=ML_StdLogicVectorFormat(3),
            debug=debug_std,
            tag="booth_digit_%d" % (digit_index + 1)
        )
        digit_list.append((digit, digit_index + 1))

    def DCV(value):
        """ Digit Constante Value """
        return Constant(value, precision=ML_StdLogicVectorFormat(3))

    # PARTIAL PRODUCT GENERATION
    # Radix-4 booth recoding requires the following Partial Products
    # -2.rhs, -rhs, 0, rhs and 2.rhs
    # Negative PP are obtained by 1's complement of the value correctly shifted
    # adding a positive one to the LSB (inserted separately) and assuming
    # MSB digit has a negative weight
    for digit, index in digit_list:
        pp_zero = LogicalOr(
            Equal(digit, DCV(0), precision=ML_Bool),
            Equal(digit, DCV(7), precision=ML_Bool),
            precision=ML_Bool
        )
        pp_shifted = LogicalOr(
            Equal(digit, DCV(3), precision=ML_Bool),
            Equal(digit, DCV(4), precision=ML_Bool),
            precision=ML_Bool
        )
        # excluding zero case
        pp_neg_bit = BitSelection(digit, 2)
        pp_neg = equal_to(pp_neg_bit, 1)
        pp_neg_lsb_carryin = Select(
            LogicalAnd(pp_neg, LogicalNot(pp_zero)),
            Constant(1, precision=ML_StdLogic),
            Constant(0, precision=ML_StdLogic),
            tag="pp_%d_neg_lsb_carryin" % index,
            debug=debug_std
        )

        # LSB digit
        lsb_pp_digit = Select(
            pp_shifted,
            Constant(0, precision=ML_StdLogic),
            BitSelection(rhs, 0),
            precision=ML_StdLogic
        )
        lsb_local_pp = Select(
            pp_zero,
            Constant(0, precision=ML_StdLogic),
            Select(
                pp_neg,
                BitLogicNegate(lsb_pp_digit),
                lsb_pp_digit,
                precision=ML_StdLogic
            ),
            debug=debug_std,
            tag="lsb_local_pp_%d" % index,
            precision=ML_StdLogic
        )
        pos_bit_heap.insert_bit(index, lsb_local_pp)
        pos_bit_heap.insert_bit(index, pp_neg_lsb_carryin)

        # other digits
        rhs_size = rhs.get_precision().get_bit_size()
        for k in range(1, rhs_size):
            pp_digit = Select(
                pp_shifted,
                BitSelection(rhs, k-1),
                BitSelection(rhs, k),
                precision=ML_StdLogic
            )
            local_pp = Select(
                pp_zero,
                Constant(0, precision=ML_StdLogic),
                Select(
                    pp_neg,
                    BitLogicNegate(pp_digit),
                    pp_digit,
                    precision=ML_StdLogic
                ),
                debug=debug_std,
                tag="local_pp_%d_%d" % (index, k),
                precision=ML_StdLogic
            )
            pos_bit_heap.insert_bit(index + k, local_pp)
        # MSB digit
        msb_pp_digit = pp_digit = Select(
            pp_shifted,
            BitSelection(rhs, rhs_size-1),
            # TODO: fix for signed rhs
            Constant(0, precision=ML_StdLogic) if not(rhs.get_precision().get_signed()) else BitSelection(rhs, rhs_size - 1),
            precision=ML_StdLogic
        )
        msb_pp = Select(
            pp_zero,
            Constant(0, precision=ML_StdLogic),
            Select(
                pp_neg,
                BitLogicNegate(msb_pp_digit),
                msb_pp_digit,
                precision=ML_StdLogic
            ),
            debug=debug_std,
            tag="msb_pp_%d" % (index),
            precision=ML_StdLogic
        )
        if rhs.get_precision().get_signed():
            neg_bit_heap.insert_bit(index + rhs_size, msb_pp)
        else:
            pos_bit_heap.insert_bit(index + rhs_size, msb_pp)
            # MSB negative digit,
            # 'rhs_size + index) is the position of the MSB digit of rhs shifted by 1
            # we add +1 to get to the sign position
            neg_bit_heap.insert_bit(index + rhs_size  + 1, pp_neg_lsb_carryin)


class MultArray(ML_Entity("mult_array")):
    def __init__(self,
                 arg_template = DefaultEntityArgTemplate,
                 precision = fixed_point(32, 0, signed=False),
                 accuracy    = ML_Faithful,
                 debug_flag = False,
                 target = VHDLBackend(),
                 output_file = "mult_array.vhd",
                 entity_name = "mult_array",
                 language = VHDL_Code,
                 acc_prec = None,
                 pipelined = False):
        # initializing I/O precision
        precision = arg_template.precision
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
        self.booth_mode = arg_template.booth_mode
        # reduction method
        self.reduction_method = arg_template.method
        # limit of height for each compression stage
        self.stage_height_limit = arg_template.stage_height_limit

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
                #("beforepipelining:dump_with_stages"),
                #("beforepipelining:quit"),
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


    def clean_stage(self, stage_id):
        """ translate stage_id to current stage value if stage_id
            is undefined (None) """
        if stage_id is None:
            return self.implementation.get_current_stage()
        else:
            return stage_id


    def generate_dummy_scheme(self):
        Log.report(
            Log.Info,
            "generating MultArray with output precision {precision}".format(
                precision = self.precision))

        acc = None
        a_inputs = {}
        b_inputs = {}

        for index, operation_input in enumerate(self.op_expr):
            print "%s" % str(operation_input)

        stage_map = {}

        for index, operation_input in enumerate(self.op_expr):
            if isinstance(operation_input, MultInput):
                mult_input = operation_input
                a_i = self.implementation.add_input_signal("a_%d_i" % index, mult_input.lhs_precision)
                b_i = self.implementation.add_input_signal("b_%d_i" % index, mult_input.rhs_precision)
                lhs_stage = clean_stage(mult_input.lhs_stage)
                rhs_stage = clean_stage(mult_input.rhs_stage)
                a_i.set_attributes(init_stage=lhs_stage)
                b_i.set_attributes(init_stage=rhs_stage)
                op_stage = max(lhs_stage, rhs_stage)
                if not op_stage in stage_map: stage_map[op_stage] = []
                stage_map[op_stage].append((Multiplication, [a_i, b_i]))
            elif isinstance(operation_input, OpInput):
                c_i = self.implementation.add_input_signal("c_%d_i" % index, operation_input.precision)
                op_stage = operation_input.stage
                c_i.set_attributes(init_stage=self.clean_stage(op_stage))
                if not op_stage in stage_map: stage_map[op_stage] = []
                stage_map[op_stage].append((lambda v: v, [c_i]))

        stage_index_list = sorted(stage_map.keys())
        for stage_id in stage_index_list:
            # synchronizing pipeline stage
            if stage_id is None:
                pass
            else:
                while stage_id > self.implementation.get_current_stage():
                    self.implementation.start_new_stage()
            operation_list = stage_map[stage_id]
            for ctor, operand_list in operation_list:
                new_term = ctor(*tuple(operand_list))
                if acc is None:
                    acc = new_term
                else:
                    acc = Addition(acc, new_term)

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


        def merge_product_in_heap(operand_list, pos_bit_heap, neg_bit_heap):
            a_i, b_i = operand_list
            if self.booth_mode:
                booth_radix4_multiply(a_i, b_i, pos_bit_heap, neg_bit_heap)
            else:
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

        def merge_addition_in_heap(operand_list, pos_bit_heap, neg_bit_heap):
            add_op = operand_list[0]
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


        # fixing precision
        for index, operation in enumerate(self.op_expr):
            print str(operation)

        stage_operation_map = {}

        # generating input signals
        for index, operation_input in enumerate(self.op_expr):
            if isinstance(operation_input, MultInput):
                a_i = self.implementation.add_input_signal("a_%d_i" % index, operation_input.lhs_precision)
                b_i = self.implementation.add_input_signal("b_%d_i" % index, operation_input.rhs_precision)
                lhs_stage = self.clean_stage(operation_input.lhs_stage)
                rhs_stage = self.clean_stage(operation_input.rhs_stage)
                a_i.set_attributes(init_stage=lhs_stage)
                b_i.set_attributes(init_stage=rhs_stage)
                op_stage = max(lhs_stage, rhs_stage)
                if not op_stage in stage_operation_map: stage_operation_map[op_stage] = []
                stage_operation_map[op_stage].append((merge_product_in_heap, [a_i, b_i]))
            elif isinstance(operation_input, OpInput):
                c_i = self.implementation.add_input_signal("c_%d_i" % index, operation_input.precision)
                op_stage = self.clean_stage(operation_input.stage)
                c_i.set_attributes(init_stage=self.clean_stage(op_stage))
                if not op_stage in stage_operation_map: stage_operation_map[op_stage] = []
                stage_operation_map[op_stage].append((merge_addition_in_heap, [c_i]))


        # heap of positive bits
        pos_bit_heap = BitHeap()
        # heap of negative bits
        neg_bit_heap = BitHeap()

        def reduce_heap(pos_bit_heap, neg_bit_heap, limit=2):
            """ reduce both pos_bit_heap and neg_bit_heap until their height
                is lower or equal to @p limit """
            # Partial Product reduction
            while pos_bit_heap.max_count() > limit:
                pos_bit_heap = REDUCTION_METHOD_MAP[self.reduction_method](pos_bit_heap)
            while neg_bit_heap.max_count() > limit:
                neg_bit_heap = REDUCTION_METHOD_MAP[self.reduction_method](neg_bit_heap)
            return pos_bit_heap, neg_bit_heap

        stage_index_list = sorted(stage_operation_map.keys())
        for stage_id in stage_index_list:
            # synchronizing pipeline stage
            if stage_id is None:
                pass
            else:
                while stage_id > self.implementation.get_current_stage():
                    print "reducing bit heaps and inserting new stage"
                    pos_bit_heap, neg_bit_heap = reduce_heap(pos_bit_heap, neg_bit_heap)
                    self.implementation.start_new_stage()

            operation_list = stage_operation_map[stage_id]
            for merge_ctor, operand_list in operation_list:
                merge_ctor(operand_list, pos_bit_heap, neg_bit_heap)


        # final stage reduction
        pos_bit_heap, neg_bit_heap = reduce_heap(pos_bit_heap, neg_bit_heap)

        # final conversion to scalar operands
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

                test_case_min["a_%d_i" % index] = operation.lhs_precision.get_min_value()
                test_case_min["b_%d_i" % index] = operation.rhs_precision.get_min_value()
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
            "--booth",
            dest="booth_mode",
            default=False,
            const=True,
            action="store_const",
            help="activate booth recoding")
        arg_template.parser.add_argument(
            "--method",
            type=ReductionMethod,
            default=ReductionMethod.Wallace,
            choices=list(ReductionMethod),
            help="define compression reduction methode"
        )
        arg_template.parser.add_argument(
            "--stage-height-limit",
            type=lambda s: [int(v) for v in s.split(",")],
            default=[None],
            help="deefine the list of height limit (lower) for each compression stage"
        )
        # argument extraction
        args = parse_arg_index_list = arg_template.arg_extraction()

        ml_hw_mpfma            = MultArray(args, pipelined=args.pipelined)

        ml_hw_mpfma.gen_implementation()
