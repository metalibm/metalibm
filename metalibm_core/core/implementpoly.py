# -*- coding: utf-8 -*-
import sys

import sollya
import os
from sollya import Interval, sup

# constant for sollya's version of number 2
S2 = sollya.SollyaObject(2)
sollya.settings.display = sollya.decimal


from metalibm_core.core.ml_formats import (
    ML_DoubleDouble, ML_TripleDouble, ML_Binary64, ML_Void, ML_FP_MultiElementFormat
)
from metalibm_core.core.ml_complex_formats import ML_Pointer_Format
from metalibm_core.core.precisions import ML_CorrectlyRounded
from metalibm_core.code_generation.code_constant import C_Code
from metalibm_core.core.polynomials import Polynomial
from metalibm_core.core.ml_operations import (
    Return, Conversion, Addition, Multiplication,
    BuildFromComponent, ReferenceAssign, Dereference, Statement,
    Variable, Constant,
)
from metalibm_core.utility.log_report import Log

from metalibm_core.utility.ml_template import (
    DefaultArgTemplate, ML_NewArgTemplate
)

from metalibm_core.opt.ml_blocks import (
    MP_Node, get_Addition_MB_compatible_list,
    get_Multiplication_MB_compatible_list,
    get_MB_cost, MB_Normalize_33_td, MB_Identity,
    MB_Wrapper_2Op, MB_PostWrapper_2Op, is_tri_limb_precision,
)

from metalibm_core.code_generation.generic_processor import GenericProcessor


sollya.settings.verbosity = 2


def get_accuracy_from_epsilon(epsilon):
    """ convert a numerical relative error into
        a number of accuracy bits """
    return sollya.floor(-sollya.log2(abs(epsilon)))

class MLL_Format:
    """ Container for all information required to
        describe a node format as infered by MLL_Context """
    def __init__(self, mp_node, meta_block, eps_target):
        # support format
        self.mp_node = mp_node
        # upper bound on the relative error, eps <= 2^-self.accuracy
        self.accuracy = get_accuracy_from_epsilon(mp_node.epsilon)
        # meta_block used to generate this node
        self.meta_block = meta_block
        # initial requirement for epsilon target
        self.eps_target = eps_target

    @property
    def epsilon(self):
        """ upper bound to the relative error of the associated
            multi-precision node """
        return self.mp_node.epsilon

    def __str__(self):
        """ convert to string (used for comprehensive content display) """
        return "(mp_node={}, accuracy={}, meta_block={}, eps_target={})".format(
            str(self.mp_node), str(self.accuracy), str(self.meta_block), str(self.eps_target)
        )


def get_MB_compatible_list(OpClass, lhs, rhs):
    """ return a list of metablock instance implementing an operation of
        type OpClass and compatible with format descriptor @p lhs and @p rhs
        """
    fct_map = {
        Addition: get_Addition_MB_compatible_list,
        Multiplication: get_Multiplication_MB_compatible_list
    }
    return fct_map[OpClass](lhs, rhs)

class MLL_Context:
    """ Class to wrapper the opaque type which bridges computeErrorBounds
        and metalibm Lugdunum """
    def __init__(self, var_format, var_interval):
        # dummy 1000-bit accuracy
        if isinstance(var_format, ML_FP_MultiElementFormat):
                # no overlap
            overlaps = [S2**-var_format.get_limb_precision(0).get_mantissa_size()] * (var_format.limb_num-1)
        else:
            overlaps = []
        self.variableFormat = MLL_Format(
            MP_Node(
                var_format,
                0,
                overlaps,
                interval=var_interval
            ),
            None,
            0
        )
        # dictionnary of power -> MLL_Format
        self.power_map = {
            #1: [(0, self.variableFormat)]
        }
        # dict k -> (i, j) which indicates that X^k must be compute as
        #                  X^i * X^j
        self.power_schemes = {}
        # maximum number of word in the largest multi-word
        self.LIMB_NUM_THRESHOLD = 3
        # minimal difference factor betwen hi and medium limb which triggers
        # the insertion of a renormalization operation
        self.LIMB_DIFF_RENORM_THRESHOLD = S2**-15

    def __str__(self):
        return "MLL_Context"

    def get_ml_format_from_accuracy(self, accuracy):
        """ return a tuple (ml_format, limb_diff_factors) which
            best fit accuracy requirement """
        if accuracy <= 53:
            return ML_Binary64, []
        elif accuracy <= 106:
            return ML_DoubleDouble, [S2**-53]
        elif accuracy <= 159:
            return ML_TripleDouble, [S2**-53, S2**-53]
        else:
            return None, []

    def get_smaller_format_min_error(self, ml_format):
        """ return the maximal accuracy / minimal error
            of the format just before @p ml_format in term
            of size """
        MIN_ERROR_MAP = {
            ML_Binary64: -sollya.log2(0), # no format smaller than ML_Binary64
            ML_DoubleDouble: S2**-53,
            ML_TripleDouble: S2**-106
        }
        return MIN_ERROR_MAP[ml_format]


    def get_format_from_accuracy(self, accuracy, eps_target=None, interval=None, exact=False):
        # TODO: manage ML_Binary32
        epsilon = 0 if exact else S2**-accuracy
        ml_format, limb_diff_factors = self.get_ml_format_from_accuracy(accuracy)
        if ml_format is None:
            Log.report(Log.Error, "unable to find a format for accuracy={}, eps_target={}, interval={}", accuracy, eps_target, interval)
        else:
            eps_target = S2**-accuracy if eps_target is None else eps_target
            return MLL_Format(MP_Node(ml_format, epsilon, limb_diff_factors, interval), None, eps_target)


    def computeBoundAddition(self, out_format, input_format_lhs, input_format_rhs):
        eps = sollya.SollyaObject(out_format.meta_block.local_relative_error_eval(
            input_format_lhs.mp_node, input_format_rhs.mp_node
        ))
        return eps


    def computeBoundMultiplication(self, out_format, input_format_lhs, input_format_rhs):
        eps = sollya.SollyaObject(out_format.meta_block.local_relative_error_eval(
            input_format_lhs.mp_node, input_format_rhs.mp_node
        ))
        return eps


    def computeBoundPower(self, k, out_format, var_format):
        # TODO: fix
        epsilon = out_format.mp_node.epsilon
        eps_target = out_format.eps_target
        # to avoid derivating to larger and larger format when post-processing
        # powerings, we over-estimate the error while matching eps_target
        if eps_target > epsilon and epsilon > 0:
            # limiting error to limit precision explosion
            l_eps_target = sollya.log2(eps_target)
            l_epsilon = sollya.log2(epsilon)

            virtual_error_log = (l_eps_target + l_epsilon) / 2.0
            virtual_error = sollya.evaluate(sollya.SollyaObject(S2**(virtual_error_log)), 1)
            print("lying on power_error target=2^{}, epsilon=2^{}, virtual_error=2^{} / {}".format(
                l_eps_target, l_epsilon, virtual_error_log, virtual_error))
            return virtual_error
        else:
            return sollya.SollyaObject(epsilon)


    def compareFormats(self, format_a, format_b):
        if format_a.accuracy < format_b.accuracy:
            return -1
        elif format_a.accuracy == format_b.accuracy:
            return 0
        else:
            return 1


    def computeBoundVariableRounding(self, format_a, format_b):
        """ Returns a bound on the relative error implied by "rounding" _x_,
           which is naturally stored as an @p format_a variable,
           to a format_b variable.

           If bf is "larger" (i.e. more precise) than af,
           the error is most probably zero as this just means
           "upconverting" a variable.

           The inputs af and bf are of the "opaque format type".

           The result is a positive or zero real number.
           """
        if format_a.accuracy <= format_b.accuracy:
            return 0
        else:
            return S2**(-format_b.accuracy)# + 1)

    def compute_output_format(self, OpClass, epsTarget, lhs, rhs):
        """ compute output format for a generic operation class @p OpClass """
        def get_renormalize_MB(op):
            """ determine if there exists a renormalization meta-block
                compatible with @p op """
            for mb in [MB_Normalize_33_td]:
                if mb.check_input_descriptors(op) and op.limb_diff_factor[0] >= self.LIMB_DIFF_RENORM_THRESHOLD:
                    return mb
            return None
        ft_compatible_list = get_MB_compatible_list(OpClass, lhs, rhs)
        def check_mb_error_target(mb, epsTarget, lhs, rhs):
            return mb.local_relative_error_eval(lhs, rhs) <= epsTarget
        valid_eps_list = [mb for mb in ft_compatible_list if check_mb_error_target(mb, epsTarget, lhs, rhs)]
        renormalize_rhs = False
        renormalize_lhs = False
        if not len(valid_eps_list):
            # trying to insert a normalize to see if a compatible meta-block appears
            # left-hand side
            renorm_MB_lhs = get_renormalize_MB(lhs)
            if renorm_MB_lhs:
                Log.report(Log.Info, "Renormalizing from {} ", lhs)
                lhs = renorm_MB_lhs.get_output_descriptor(lhs)
                Log.report(Log.Info, "  to {}", lhs)
                renormalize_lhs = True
            # right hand side
            renorm_MB_rhs = get_renormalize_MB(rhs)
            if renorm_MB_rhs:
                Log.report(Log.Info, "Renormalizing from {} ", rhs)
                rhs = renorm_MB_rhs.get_output_descriptor(rhs)
                Log.report(Log.Info, "  to {}", rhs)
                renormalize_rhs = True

            ft_compatible_list = get_MB_compatible_list(OpClass, lhs, rhs)
            valid_eps_list = [mb for mb in ft_compatible_list if check_mb_error_target(mb, epsTarget, lhs, rhs)]

            if not len(valid_eps_list):
                Log.report(Log.Error, "unable to find a MB for OpClass={}, epsTarget={}, lhs={}, rhs={}", OpClass, epsTarget, lhs, rhs)
                return None

        meta_block = min(valid_eps_list, key=get_MB_cost)
        print("select meta_block is {}".format(meta_block))
        out_format = meta_block.get_output_descriptor(lhs, rhs, global_error=True)
        if renormalize_lhs or renormalize_rhs:
            lhs_block = renorm_MB_lhs if renormalize_lhs else MB_Identity
            rhs_block = renorm_MB_rhs if renormalize_rhs else MB_Identity
            return MLL_Format(out_format, MB_Wrapper_2Op(meta_block, lhs_block, rhs_block), epsTarget)
        else:
            if is_tri_limb_precision(out_format.precision) and out_format.limb_diff_factor[0] >= self.LIMB_DIFF_RENORM_THRESHOLD:
                # reduction in precision is too big => Normalize insertion
                renorm_MB = get_renormalize_MB(out_format)
                Log.report(Log.Info, "Renormalizing {} to", out_format)
                out_format = renorm_MB.get_output_descriptor(lhs)
                Log.report(Log.Info, "  {}", out_format)
                meta_block = MB_PostWrapper_2Op(meta_block, renorm_MB)
            return MLL_Format(out_format, meta_block, epsTarget)


    def computeOutputFormatAddition(self, epsTarget, inputFormatA, inputFormatB):
        """
           Returns the output format of an addition  that will be fed two operands
           of formats inputFormatA and inputFormatB and that is supposed to have a
           relative error less than or equal to epsTarget.

           The input and output formats are of type "opaque format type".
           The epsTarget input is a positive or zero real number.
        """

        lhs = inputFormatA.mp_node
        rhs = inputFormatB.mp_node
        return self.compute_output_format(Addition, epsTarget, lhs, rhs)


    def computeOutputFormatMultiplication(self, epsTarget, inputFormatA, inputFormatB):
        """ Returns the output format of a multiplication that will be fed two operands
           of formats inputFormatA and inputFormatB and that is supposed to have a
           relative error less than or equal to epsTarget.

           The input and output formats are of type "opaque format type".
           The epsTarget input is a positive or zero real number.
        """
        lhs = inputFormatA.mp_node
        rhs = inputFormatB.mp_node
        return self.compute_output_format(Multiplication, epsTarget, lhs, rhs)


    def computeConstantFormat(self, c):
        if c == 0:
            # default to double precision
            return self.get_format_from_accuracy(53, eps_target=0, interval=Interval(c), exact=True)
        else:
            accuracy = 0
            cN = c
            limb_num = 0
            while cN != 0 and limb_num < self.LIMB_NUM_THRESHOLD:
                cR = sollya.round(cN, sollya.binary64, sollya.RN)
                cN = cN - cR
                accuracy += 53
                limb_num += 1
            if accuracy > 159 or limb_num > self.LIMB_NUM_THRESHOLD:
                eps_target = S2**--accuracy
                accuracy = 159
            else:
                eps_target = 0 if cN == 0 else S2**-accuracy
            return self.get_format_from_accuracy(accuracy, eps_target=eps_target, interval=Interval(c), exact=True)


    
    def computeNeededVariableFormat(self, I, epsTarget, variableFormat):
        if epsTarget > 0:
            # TODO: fix to support ML_Binary32
            if epsTarget >= S2**-53 or variableFormat.mp_node.precision is ML_Binary64:
                # FIXME: default to minimal precision ML_Binary64
                return variableFormat
            else:
                target_accuracy = sollya.ceil(-sollya.log2(epsTarget))
                target_format   = self.get_format_from_accuracy(target_accuracy, eps_target=epsTarget, interval=variableFormat.mp_node.interval)
                if target_format.mp_node.precision.get_bit_size() < variableFormat.mp_node.precision.get_bit_size():
                    return target_format
                else:
                    # if variableFormat is smaller (less bits) and more accurate
                    # then we use it
                    return variableFormat
        else:
            return variableFormat

    
    def computeOutputFormatPower(self, k, epsTarget, variableFormat):
        if not k is None:
            k = int(k)
        if epsTarget > 0:
            if k == 1:
                # if k is 1, then the format is the one of Variable verifying
                # epsTarget constraint
                return self.computeNeededVariableFormat(None, epsTarget, variableFormat)
            else:
                final_format = None
                if k in self.power_map:
                    for error_bound, power_format_list in self.power_map[k]:
                        # if record error_bound is less than epsTarget
                        # TODO: implement search for tighter bound
                        if epsTarget > error_bound:
                            # final_format is last element of power_format_list
                            final_format = power_format_list[-1]
                            if self.get_smaller_format_min_error(final_format.mp_node.precision) < epsTarget:
                                # there is possibly a smaller format than final_format which
                                # could match epsTarget constraint
                                final_format = None # GR: to be checked ??
                                continue
                            # updating final_format eps_target to make sure it
                            # always keep track of the most constraining
                            # eps_target
                            final_format.eps_target = min(final_format.eps_target, epsTarget)
                            return final_format
                        # as error_bound are ordered from larger to smaller
                if final_format is None:
                    if k == 2:
                        lhs_k, rhs_k = 1, 1
                    else:
                        lhs_k, rhs_k = self.power_schemes[k]
                    # TODO: implement more complex error budget
                    lhs_format = self.computeOutputFormatPower(lhs_k, epsTarget / 4, variableFormat)
                    rhs_format = self.computeOutputFormatPower(rhs_k, epsTarget / 4, variableFormat)
                    mult_format = self.computeOutputFormatMultiplication(epsTarget / 2, lhs_format, rhs_format)

                    lhs_error = self.computeBoundPower(lhs_k, lhs_format, variableFormat)
                    rhs_error = self.computeBoundPower(rhs_k, rhs_format, variableFormat)
                    mult_error = self.computeBoundMultiplication(mult_format, lhs_format, rhs_format)


                    # TODO: should take into account second order error
                    final_error = lhs_error + rhs_error + mult_error
                    if final_error > epsTarget:
                        print("possible ERROR in computeOutputFormatPower: failed to meet epsTarget")

                    final_format = mult_format
                    final_format.eps_target = epsTarget

                    record = (final_error, [lhs_format, rhs_format, final_format])

                    if not k in self.power_map:
                        self.power_map[k] = []
                    self.power_map[k].append(record)
                    # sorting list from large error to small errors
                    self.power_map[k].sort(key=(lambda v: v[0]), reverse=True)

                return final_format
        else:
            Log.report(Log.Error, "unable to computeOutputFormatPower for k={}, epsTarget={}, variableFormat={}", k, epsTarget, variableFormat)
            return None


    def roundConstant(self, c, epsTarget):
        """ Rounds a given coefficient c into a format that guarantees
           that the rounding error is less than epsTarget. The function
           does not return the retained format but the rounded number.

           epsTarget is a positive OR ZERO number.

           If epsTarget is zero, the function is supposed to check
           whether there exists a format such that the constant can be
           represented exactly.

           The function returns a structure with at least two fields

           *   .okay indicating that the rounding was able to be performed
           *   .c    the rounded constant

        """
        if epsTarget >= 0:
                if c == 0:
                    # 0 value can always fit exactly in a format
                    return c
                else:
                    cR = sollya.round(c, sollya.binary64, sollya.RN)
                    limb_num = 1
                    while abs(cR / c - 1) > epsTarget and limb_num < self.LIMB_NUM_THRESHOLD:
                        cN = sollya.round(c - cR, sollya.binary64, sollya.RN)
                        cR += cN
                        limb_num += 1
                    if limb_num > self.LIMB_NUM_THRESHOLD or abs(cR / c - 1) > epsTarget:
                        return None
                    else:
                        return cR
        else:
            return None


def legalize_node_format(node, expected_format):
    if node.precision is expected_format:
        return node
    else:
        return Conversion(node, precision=expected_format)

def get_add_error_budget(lhs, rhs, eps_target):
    """ How accurate should be the addition of lhs + rhs to
        ensure that the overall global relative error is limited to eps_target
    """
    # lhs_eps = lhs.epsilon
    # rhs_eps = rhs.epsilon
    # real result = (lhs (1 + lhs_eps) + rhs (1 + rhs_eps)) (1 + add_eps)
    # real result = (lhs + rhs + lhs . lhs_eps + rhs . rhs_eps) (1 + add_eps)
    #             = exact result (1 + (lhs . lhs_eps + rhs . rhs_eps) / exact_result) (1 + add_eps)
    #  eps_in = (lhs . lhs_eps + rhs . rhs_eps) / exact_result
    # real result = exact resul * (1 + eps_in + add_eps + eps_in * add_eps)
    #
    # objective (eps_in + add_eps + eps_in * add_eps) <= eps_target
    #           |eps_in| + |add_eps| + |eps_in| * |add_eps| <= eps_target
    #           |add_eps| (1 + |eps_in|) <= |eps_target| - |eps_in|
    #           |add_eps|  <= (|eps_target| - |eps_in|) / (1 + |eps_in|)
    # assuming |eps_target| > |eps_in|
    eps_in = (sup(abs(lhs.internal)) * lhs.epsilon + sup(abs(rhs.interval)) * rhs.epsilon) / inf(abs(lhs.interval + rhs.interval))
    assert eps_in > 0
    assert eps_in >= eps_target
    add_eps_bound = (eps_target - eps_in) / (1 + eps_in)
    return add_eps_bound

def get_add_error_split(lhs_interval, rhs_interval, eps_target):
    """ provide a repartition of the error budget accross lhs's error, rhs'error
        and addition lhs + rhs error such that the overall relative error is
        bounded by @p eps_target """
    min_exact_result = inf(abs(lhs_interval + rhs_interval))
    lhs_split = (lhs_interval / min_exact_result) * 0.5
    rhs_split = (rhs_interval / min_exact_result) * 0.5
    return (lhs_split * eps_target), (rhs_split * eps_target), (0.5 * eps_target)


def get_mul_error_budget(lhs, rhs, eps_target):
    """ How accurate should be the multiplication of lhs * rhs to
        ensure that the overall global relative error is limited to eps_target
    """
    # lhs_eps = lhs.epsilon
    # rhs_eps = rhs.epsilon
    # real result = (lhs (1 + lhs_eps) * rhs (1 + rhs_eps)) (1 + mul_eps)
    # real result = (lhs * rhs) (1 + lhs_eps + rhs_eps + lhs_eps * rhs_eps) (1 + mul_eps)
    #
    #  eps_in = lhs_eps + rhs_eps + lhs_eps * rhs_eps
    # real result = exact resul * (1 + eps_in + mul_eps + eps_in * mul_eps)
    #
    # objective (eps_in + mul_eps + eps_in * mul_eps) <= eps_target
    #           |eps_in| + |mul_eps| + |eps_in| * |mul_eps| <= eps_target
    #           |mul_eps| (1 + |eps_in|) <= |eps_target| - |eps_in|
    #           |mul_eps|  <= (|eps_target| - |eps_in|) / (1 + |eps_in|)
    # assuming |eps_target| > |eps_in|
    eps_in = lhs.epsilon + rhs.epsilon + lhs.epsilon * rhs.epsilon
    assert eps_in > 0
    assert eps_in >= eps_target
    mul_eps_bound = (eps_target - eps_in) / (1 + eps_in)
    return mul_eps_bound

def mll_implementpoly_horner(ctx, poly_object, eps, variable):
    """ generate an implementation of polynomail @p poly_object of @p variable
        whose evalution error is bounded by @p eps. @p variable must have a
        interval and a precision set

        @param ctx multi-word precision context to use
        @param poly_object polynomial object to implement
        @param eps target relative error bound
        @param variable polynomial input variable

        @return <implementation node>, <real relative error>"""
    if poly_object.degree == 0:
        # constant only
        cst = poly_object.coeff_map[0]
        rounded_cst = ctx.roundConstant(cst, eps)
        cst_format = ctx.computeConstantFormat(rounded_cst)
        return Constant(cst, precision=cst_format), cst_format.epsilon

    elif poly_object.degree == 1:
        # cst0 + cst1 * var
        # final relative error is
        # (cst0 (1 + e0) + cst1 * var (1 + e1) (1 + ev) (1 + em))(1 + ea)
        # (cst0  + e0 * cst0  + cst1 * var (1 + e1 + ev + e1 * ev) (1 + em))(1 + ea)
        # (cst0  + e0 * cst0  + cst1 * var (1 + e1 + ev + e1 * ev + em + e1 * em + ev * em + e1 * ev * em) )(1 + ea)
        # (cst0 + cst1 * var) (1 + ea) (1 + e0 * cst0 + + e1 + ev + e1 * ev + em + e1 * em + ev * em + e1 * ev * em)
        # em is epsilon for the multiplication
        # ea is epsilon for the addition
        # overall error is
        cst0 = poly_object.coeff_map[0]
        cst1 = poly_object.coeff_map[1]
        eps_mul = eps / 4
        eps_add = eps / 2

        cst1_rounded = ctx.roundConstant(cst1, eps / 4)
        cst1_error = abs((cst1 - cst1_rounded) / cst1_rounded)
        cst1_format = ctx.computeConstantFormat(cst1_rounded)
        cst0_rounded = ctx.roundConstant(cst0, eps / 4)
        cst0_format = ctx.computeConstantFormat(cst0_rounded)

        eps_var = eps / 4
        var_format = ctx.computeNeededVariableFormat(variable.interval, eps_var, variable.precision) 
        var_node = legalize_node_format(variable, var_format)
        mul_format = ctx.computeOutputFormatMultiplication(eps_mul, cst1_format, var_format)
        add_format = ctx.computeOutputFormatAddition(eps_add, cst0_format, mul_format)

        return Addition(
            Constant(cst0_rounded, precision=cst0_format),
            Multiplication(
                Constant(cst1_rounded, precision=cst1_format),
                var_node,
                precision=mul_format
            ),
            precision=add_format
        ), add_format.epsilon # TODO: local error only

    elif poly_object.degree > 1:
        # cst0 + var * poly
        cst0 = poly_object.coeff_map[0]
        cst0_rounded = ctx.roundConstant(cst0, eps / 4)
        cst0_format = ctx.computeConstantFormat(cst0_rounded)

        eps_var = eps / 4
        var_format = ctx.computeNeededVariableFormat(variable.interval, eps_var, variable.precision) 
        var_node = legalize_node_format(variable, var_format)

        sub_poly = poly_object.sub_poly(start_index=1, offset=1)
        eps_poly = eps / 4
        poly_node, poly_accuracy = mll_implementpoly_horner(ctx, sub_poly, eps_poly, variable)

        eps_mul = eps / 4
        mul_format = ctx.computeOutputFormatMultiplication(eps_mul, var_format, poly_node.precision)

        eps_add = eps / 4
        add_format = ctx.computeOutputFormatAddition(eps_add, cst0_format, mul_format)

        return Addition(
            Constant(cst0_rounded, precision=cst0_format),
            Multiplication(
                var_node,
                poly_node,
                precision=mul_format
            ),
            precision=add_format
        ), add_format.epsilon # TODO: local error only
    else:
        Log.report(Log.Error, "poly degree must be positive or null. {}, {}", poly_object.degree, poly_object)


if __name__ == "__main__":
    approx_interval = Interval(-S2**-5, S2**-5)
    ctx = MLL_Context(ML_Binary64, approx_interval)
    vx = Variable("x", precision=ctx.variableFormat, interval=approx_interval)
    poly_object = Polynomial.build_from_approximation(
        sollya.exp(sollya.x), 6,
        [sollya.doubledouble] * 11,
        vx.interval
    )
    print("poly object is {}".format(poly_object))
    eps_target = S2**-51
    poly_graph, poly_epsilon = mll_implementpoly_horner(ctx, poly_object, eps_target, vx)
    print("poly_graph is {}".format(poly_graph.get_str(depth=None, display_precision=True)))
    print("poly epsilon is {}".format(float(poly_epsilon)))
    print("poly accuracy is {}".format(get_accuracy_from_epsilon(poly_epsilon)))

