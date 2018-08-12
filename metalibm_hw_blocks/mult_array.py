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

import sollya

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
from metalibm_core.utility.ml_template import precision_parser

from metalibm_core.targets.kalray.k1c_fp_utils import (
        rnd_mode_format, rnd_rne, rnd_ru, rnd_rd, rnd_rz
)

from metalibm_hw_blocks.rtl_blocks import zext, rzext
from metalibm_hw_blocks.lzc import ML_LeadingZeroCounter
from metalibm_hw_blocks.lza import ML_LeadingZeroAnticipator


class MultInput:
    def __init__(self, lhs_precision, rhs_precision):
        self.lhs_precision = lhs_precision
        self.rhs_precision = rhs_precision

    @staticmethod
    def parse(s):
        lhs, rhs = s.split("x")
        print lhs, rhs
        return MultInput(
            precision_parser(lhs),
            precision_parser(rhs)
        )

def multiplication_descriptor_parser(arg_str):
    return [MultInput.parse(s) for s in arg_str.split("+")]



class BitHeap:
    def __init__(self):
        self.heap = {}
        self.count = {}

    def insert_bit(self, index, value):
        print "inserting bit {} with weight {}".format(value, index)
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
        print "popping bit {} from weight {}".format(bit, index)
        return bit

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



sys.setrecursionlimit(1500)

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
        self.mult_desc = arg_template.mult_desc

    ## default argument template generation
    @staticmethod
    def get_default_args(**kw):
        default_dict = {
            "precision": fixed_point(32,0),
            "target": VHDLBackend(),
            "output_file": "mult_array.vhd",
            "entity_name": "mult_array",
            "language": VHDL_Code,
            "pipelined": False,
            "passes": [
                ("beforepipelining:size_datapath"),
                ("beforepipelining:rtl_legalize"),
                ("beforepipelining:unify_pipeline_stages"),
                # ("beforecodegen:dump"),
                ],
        }
        default_dict.update(kw)
        return DefaultEntityArgTemplate(
            **default_dict
        )

    def generate_scheme(self):
        ## Generate Fused multiply and add comput <x> . <y> + <z>
        Log.report(
            Log.Info,
            "generating MultArray with output precision {precision}".format(
                precision = self.precision))

        acc = None

        a_inputs = {}
        b_inputs = {}
        # fixing precision
        for index, mult_input in enumerate(self.mult_desc):
            print "{} x {}".format(mult_input.lhs_precision, mult_input.rhs_precision)
            lhs_precision = fixed_point(mult_input.lhs_precision.get_integer_size(), mult_input.lhs_precision.get_frac_size(), signed=mult_input.lhs_precision.signed)
            rhs_precision = fixed_point(mult_input.rhs_precision.get_integer_size(), mult_input.rhs_precision.get_frac_size(), signed=mult_input.rhs_precision.signed)
            mult_input.lhs_precision = lhs_precision
            mult_input.rhs_precision = rhs_precision

        # generating input signals
        for index, mult_input in enumerate(self.mult_desc):
            a_i = self.implementation.add_input_signal("a_%d_i" % index, mult_input.lhs_precision)
            b_i = self.implementation.add_input_signal("b_%d_i" % index, mult_input.rhs_precision)
            a_inputs[index] = a_i
            b_inputs[index] = b_i

        NUM_PRODUCTS = len(self.mult_desc)

        # heap of positive bits
        pos_bit_heap = BitHeap()

        # Partial Product generation
        for index in range(NUM_PRODUCTS): 
            a_i = a_inputs[index]
            b_i = b_inputs[index]
            a_i_precision = a_i.get_precision()
            for pp_index in range(a_i_precision.get_bit_size()):
                bit_a_j = BitSelection(a_i, pp_index) 
                pp = Select(equal_to(bit_a_j, 1), b_i, 0)
                offset = pp_index - a_i_precision.get_frac_size()
                for b_index in range(a_i_precision.get_bit_size()):
                    pp_weight = offset + b_index
                    local_bit = BitSelection(pp, b_index)
                    pos_bit_heap.insert_bit(pp_weight, local_bit)

        def comp_3to2(a, b, c):
            a = TypeCast(a, precision=fixed_point(1, 0, signed=False)) 
            b = TypeCast(b, precision=fixed_point(1, 0, signed=False)) 
            c = TypeCast(c, precision=fixed_point(1, 0, signed=False))

            full = TypeCast(Conversion(a +b + c, precision=fixed_point(2, 0, signed=False)), precision=ML_StdLogicVectorFormat(2))
            carry = BitSelection(full, 1)
            digit = BitSelection(full, 0)
            return carry, digit

        # Partial Product reduction
        current_bit_heap = pos_bit_heap
        while current_bit_heap.max_count() > 2:
            new_bit_heap = BitHeap()
            while current_bit_heap.max_count() > 0:
                bit_list, w = current_bit_heap.pop_lower_bits(3)
                if len(bit_list) <= 2:
                    for b in bit_list:
                        new_bit_heap.insert_bit(w, b)
                else:
                    b_wp1, b_w = comp_3to2(bit_list[0], bit_list[1], bit_list[2])
                    new_bit_heap.insert_bit(w + 1, b_wp1)
                    new_bit_heap.insert_bit(w, b_w)
            current_bit_heap = new_bit_heap
        # final propagating sum
        op_size = current_bit_heap.max_index - current_bit_heap.min_index + 1
        op_format = ML_StdLogicVectorFormat(op_size)
        op_carry = Signal("op_carry", precision=op_format, var_type=Variable.Local) 
        op_sum = Signal("op_sum", precision=op_format, var_type=Variable.Local) 

        op_statement = Statement()
        offset_index = current_bit_heap.min_index

        for index in range(current_bit_heap.min_index, current_bit_heap.max_index + 1):
            out_index = index - offset_index 
            bit_list = current_bit_heap.pop_bits(index, 2)
            if len(bit_list) == 0:
                op_statement.push(ReferenceAssign(BitSelection(op_carry, out_index), Constant(0, precision=ML_StdLogic)))
                op_statement.push(ReferenceAssign(BitSelection(op_sum, out_index), Constant(0, precision=ML_StdLogic)))
            elif len(bit_list) == 1:
                op_statement.push(ReferenceAssign(BitSelection(op_carry, out_index), Constant(0, precision=ML_StdLogic)))
                op_statement.push(ReferenceAssign(BitSelection(op_sum, out_index), bit_list[0]))
            else:
                op_statement.push(ReferenceAssign(BitSelection(op_carry, out_index), bit_list[1]))
                op_statement.push(ReferenceAssign(BitSelection(op_sum, out_index), bit_list[0]))

        # a PlaceHolder is inserted to force forwarding of op_statement
        # which will be removed otherwise as it does not appear anywhere in
        # the final operation graph
        acc = PlaceHolder(
            Addition(
                TypeCast(
                    op_carry,
                    precision=fixed_point(op_size - offset_index,offset_index, signed=False) 
                ),
                TypeCast(
                    op_sum,
                    precision=fixed_point(op_size - offset_index,offset_index, signed=False) 
                )
            ),
            op_statement
        )

        self.precision = fixed_point(
            self.precision.get_integer_size(),
            self.precision.get_frac_size(),
            signed=self.precision.get_signed()
        )
        result = Conversion(acc, precision=self.precision)
        self.implementation.add_output_signal("result_o", result)

        return [self.implementation]

    def numeric_emulate(self, io_map):
        acc = 0
        for index, mult_input in enumerate(self.mult_desc):
            a_i = io_map["a_%d_i" % index]
            b_i = io_map["b_%d_i" % index]
            acc += a_i * b_i

        assert acc >= 0
        return {"result_o": acc}




if __name__ == "__main__":
        # auto-test
        arg_template = ML_EntityArgTemplate(
            default_entity_name="new_mult_array",
            default_output_file="ml_mult_array.vhd",
            default_arg=MultArray.get_default_args()
        )
        # accumulator precision (also the output format)
        arg_template.parser.add_argument(
            "--mult-desc",
            dest="mult_desc",
            type=multiplication_descriptor_parser,
            default=None,
            help="Multiplication Input descriptor")
        # argument extraction
        args = parse_arg_index_list = arg_template.arg_extraction()

        ml_hw_mpfma            = MultArray(args, pipelined=args.pipelined)

        ml_hw_mpfma.gen_implementation()
