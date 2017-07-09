# -*- coding: utf-8 -*-

""" Adaptative fixed-point size unit test """

import sollya

from sollya import parse as sollya_parse

from metalibm_core.core.ml_operations import (
    Comparison, Addition, Select, Constant, ML_LeafNode, Conversion
)
from metalibm_core.code_generation.code_constant import VHDL_Code
from metalibm_core.core.ml_formats import (
    ML_Int32, ML_Bool
)
from metalibm_core.code_generation.vhdl_backend import VHDLBackend
from metalibm_core.core.ml_entity import (
    ML_Entity, ML_EntityBasis, DefaultEntityArgTemplate
)
from metalibm_core.utility.ml_template import \
    ML_EntityArgTemplate
from metalibm_core.utility.log_report import Log

from metalibm_core.core.ml_hdl_format import (
    is_fixed_point, fixed_point
)

from metalibm_core.opt.rtl_fixed_point_utils import (
    test_format_equality,
    solve_equal_formats
)


## determine generic operation precision
def solve_format_BooleanOp(optree):
    """ legalize BooleanOperation node """
    if optree.get_precision() is None:
        optree.set_precision(ML_Bool)
    return optree.get_precision()

## Determine comparison node precision
def solve_format_Comparison(optree):
    """ Legalize Comparison node """
    assert isinstance(optree, Comparison)
    if optree.get_precision() is None:
        optree.set_precision(ML_Bool)
    lhs = optree.get_input(0)
    rhs = optree.get_input(1)
    merge_format = solve_equal_formats([lhs, rhs])
    propagate_format_to_input(merge_format, optree, [0, 1])
    return solve_format_BooleanOp(optree)

## determine Addition node precision
def solve_format_Addition(optree):
    """ Legalize Addition node """
    assert isinstance(optree, Addition)
    lhs = optree.get_input(0)
    rhs = optree.get_input(1)
    lhs_precision = lhs.get_precision()
    rhs_precision = rhs.get_precision()

    if is_fixed_point(lhs_precision) and is_fixed_point(rhs_precision):
        # +1 for carry overflow
        int_size = max(
            lhs_precision.get_integer_size(),
            rhs_precision.get_integer_size()
        ) + 1
        frac_size = max(
            lhs_precision.get_frac_size(),
            rhs_precision.get_frac_size()
        )
        is_signed = lhs_precision.get_signed() or rhs_precision.get_signed()
        return fixed_point(
            int_size,
            frac_size,
            signed=is_signed
        )
    else:
        return optree.get_precision()

## determine Constant node precision
def solve_format_Constant(optree):
    """ Legalize Constant node """
    assert isinstance(optree, Constant)
    value = optree.get_value()
    assert int(value) == value
    abs_value = abs(value)
    signed = value < 0

    int_size = int(sollya.ceil(sollya.log2(abs_value))) + (1 if signed else 0)
    frac_size = 0
    return fixed_point(int_size, frac_size, signed=signed)


def format_set_if_undef(optree, new_format):
    """ Define a new format to @p optree if no format was previously
        set. """
    if optree.get_precision() is None:
        optree.set_precision(new_format)
    return optree.get_precision()


def solve_format_Select(optree):
    """ Legalize Select node """
    assert isinstance(optree, Select)
    cond = optree.get_input(0)
    solve_format_BooleanOp(cond)
    true_value = optree.get_input(1)
    false_value = optree.get_input(2)
    unified_format = solve_equal_formats([optree, true_value, false_value])
    format_set_if_undef(optree, unified_format)
    format_set_if_undef(true_value, unified_format)
    format_set_if_undef(false_value, unified_format)

    optree.set_precision(unified_format)
    true_value.set_precision(unified_format)
    false_value.set_precision(unified_format)
    return unified_format

## Test if @p optree is a Operation node propagating format
#  if it does return the list of @p optree's input index
#   where a format should be propagated
def does_node_propagate_format(optree):
    """ Test whether @p optree propagate a format definition
        to its operand. """
    if isinstance(optree, Select):
        return [1, 2]
    return []


def is_constant(optree):
    """ Test if optree is a Constant node  """
    return isinstance(optree, Constant)


def format_does_fit(cst_optree, new_format):
    """ Test if @p cst_optree fits into the precision @p new_format """
    assert is_constant(cst_optree)
    assert is_fixed_point(new_format)
    min_format = solve_format_Constant(cst_optree)
    sign_bias = 1 if (new_format.get_signed() and not min_format.get_signed()) \
        else 0
    return (new_format.get_integer_size() - sign_bias) >= \
        min_format.get_integer_size() and \
        new_format.get_frac_size() >= min_format.get_frac_size() and \
           (new_format.get_signed() or not min_format.get_signed())


# propagate the precision @p new_format to every node in
#  @p optree_list with undefined precision or instanciate
#  a Conversion node if precisions differ
def propagate_format_to_input(new_format, optree, input_index_list):
    """ Propgate new_format to @p optree's input whose index is listed in
        @p input_index_list """
    for op_index in input_index_list:
        op_input = optree.get_input(op_index)
        if op_input.get_precision() is None:
            op_input.set_precision(new_format)
            index_list = does_node_propagate_format(op_input)
            propagate_format_to_input(new_format, op_input, index_list)
        elif not test_format_equality(new_format, op_input.get_precision()):
            if is_constant(op_input):
                if format_does_fit(op_input, new_format):
                    Log.report(
                        Log.Info,
                        "Simplify Constant Conversion {} to larger Constant: {}".format(
                            op_input.get_str(display_precision=True),
                            str(new_format)
                        )
                    )
                    op_input.set_precision(new_format)
                else:
                    Log.report(
                        Log.Error,
                        "Constant is about to be reduced to a too constrained format: {}".format(
                            op_input.get_str(display_precision=True)
                        )
                    )
            else:
                new_input = Conversion(
                    op_input,
                    precision=new_format
                )
                optree.set_input(op_index, new_input)


def solve_format_rec(optree, memoization_map=None):
    """ Recursively legalize formats from @p optree, using memoization_map
        to store resolved results """
    memoization_map = {} if memoization_map is None else memoization_map
    if optree in memoization_map:
        return memoization_map[optree]
    elif isinstance(optree, ML_LeafNode):
        new_format = optree.get_precision()
        if isinstance(optree, Constant):
            new_format = solve_format_Constant(optree)

        Log.report(Log.Info,
                   "new format {} determined for {}".format(
                       str(new_format), optree.get_str(display_precision=True)
                   )
                   )

        # updating optree format
        optree.set_precision(new_format)
        memoization_map[optree] = new_format

    else:
        for op_input in optree.get_inputs():
            solve_format_rec(op_input)
        new_format = optree.get_precision()
        if not new_format is None:
            Log.report(
                Log.Info,
                "format {} has already been determined for {}".format(
                    str(new_format), optree.get_str(display_precision=True)
                )
            )
        elif isinstance(optree, Comparison):
            new_format = solve_format_Comparison(optree)
        elif isinstance(optree, Addition):
            new_format = solve_format_Addition(optree)
        elif isinstance(optree, Select):
            new_format = solve_format_Select(optree)
        elif isinstance(optree, Conversion):
            Log.report(
                Log.Error,
                "Conversion {} must have a defined format".format(
                    optree.get_str()
                )
            )
        else:
            Log.report(
                Log.Error,
                "unsupported operation in solve_format_rec: {}".format(
                    optree.get_str())
            )

        # updating optree format
        Log.report(Log.Info,
                   "new format {} determined for {}".format(
                       str(new_format), optree.get_str(display_precision=True)
                   )
                   )
        optree.set_precision(new_format)
        memoization_map[optree] = new_format

        # format propagation
        prop_index_list = does_node_propagate_format(optree)
        propagate_format_to_input(new_format, optree, prop_index_list)


class AdaptativeEntity(ML_Entity("ml_adaptative_entity")):
    """ Adaptative Entity unit-test """
    @staticmethod
    def get_default_args(width=32):
        """ generate default argument template """
        return DefaultEntityArgTemplate(
            precision=ML_Int32,
            debug_flag=False,
            target=VHDLBackend(),
            output_file="my_adapative_entity.vhd",
            entity_name="my_adaptative_entity",
            language=VHDL_Code,
            width=width,
        )

    def __init__(self, arg_template=None):
        """ Initialize """
        # building default arg_template if necessary
        arg_template = AdaptativeEntity.get_default_args() if \
            arg_template is None else arg_template
        # initializing I/O precision
        self.width = arg_template.width
        precision = arg_template.precision
        io_precisions = [precision] * 2
        Log.report(
            Log.Info,
            "generating Adaptative Entity with width={}".format(self.width)
        )

        # initializing base class
        ML_EntityBasis.__init__(self,
                                base_name="adaptative_design",
                                arg_template=arg_template
                                )

        self.accuracy = arg_template.accuracy
        self.precision = arg_template.precision

    def numeric_emulate(self, io_map):
        """ Meta-Function numeric emulation """
        raise NotImplementedError

    def generate_scheme(self):
        """ main scheme generation """
        Log.report(Log.Info, "width parameter is {}".format(self.width))

        input_precision = fixed_point(self.width, 0)
        output_precision = fixed_point(self.width, 0)

        # declaring main input variable
        var_x = self.implementation.add_input_signal("x", input_precision)
        var_y = self.implementation.add_input_signal("y", input_precision)

        test = (var_x > 20)

        large_add = (var_x + var_y)

        pre_result = Select(
            test,
            20,
            large_add
        )

        result = Conversion(pre_result, precision=output_precision)
        print result.get_str(depth=None, display_precision=True)
        solve_format_rec(result)
        print result.get_str(depth=None, display_precision=True)

        self.implementation.add_output_signal("vr_out", result)

        return [self.implementation]

    standard_test_cases = [sollya_parse(x) for x in ["1.1", "1.5"]]


if __name__ == "__main__":
        # auto-test
    main_arg_template = ML_EntityArgTemplate(
        default_entity_name="new_adapt_entity",
        default_output_file="mt_adapt_entity.vhd",
        default_arg=AdaptativeEntity.get_default_args()
    )
    main_arg_template.parser.add_argument(
        "--width", dest="width", type=int, default=32,
        help="set input width value (in bits)"
    )
    # argument extraction
    args = parse_arg_index_list = main_arg_template.arg_extraction()

    ml_adaptative = AdaptativeEntity(args)

    ml_adaptative.gen_implementation()
