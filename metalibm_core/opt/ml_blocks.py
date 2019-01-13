# -*- coding: utf-8 -*-
# vim: sw=2 sts=2

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
# Meta-blocks that return an optree to be used in an optimization pass
# Created:          Apr 28th, 2017
# last-modified:    Mar  7th, 2018
# Author(s):  Hugues de Lassus <hugues.de-lassus@univ-perp.fr>
###############################################################################

import sollya
S2 = sollya.SollyaObject(2)

from metalibm_core.core.ml_operations import (
    BitLogicRightShift, BitLogicAnd, BitArithmeticRightShift,
    Subtraction, BitLogicLeftShift, BitLogicNegate, Addition, Multiplication,
    ExponentExtraction, ExponentInsertion,
    Max, Min,
    FMS, FMA, Constant
)
from metalibm_core.core.ml_formats import (
    ML_Binary32, ML_Binary64,
    # TODO: move those import while externalizing MB-class instanciations
    ML_SingleSingle, ML_DoubleDouble, ML_TripleSingle, ML_TripleDouble,

    ML_FP_MultiElementFormat,
)

# Dynamic implementation of a vectorizable leading zero counter.
# The algorithm is taken from the Hacker's Delight and works only for 32-bit
# registers.
# TODO Adapt to other register sizes.
def generate_count_leading_zeros(vx):
    """Generate a vectorizable LZCNT optree."""

    y = - BitLogicRightShift(vx, 16)    # If left half of x is 0,
    m = BitLogicAnd(BitArithmeticRightShift(y, 16), 16)
                                        # set n = 16.  If left half
    n = Subtraction(16, m)              # is nonzero, set n = 0 and
    vx_2 = BitLogicRightShift(vx, m)    # shift x right 16.
    # Now x is of the form 0000xxxx.
    y = vx_2 - 0x100                    # If positions 8-15 are 0,
    m = BitLogicAnd(BitLogicRightShift(y, 16), 8)
                                        # add 8 to n and shift x left 8.
    n = n + m
    vx_3 = BitLogicLeftShift(vx_2, m)

    y = vx_3 - 0x1000                   # If positions 12-15 are 0,
    m = BitLogicAnd(BitLogicRightShift(y, 16), 4)
                                        # add 4 to n and shift x left 4.
    n = n + m
    vx_4 = BitLogicLeftShift(vx_3, m)

    y = vx_4 - 0x4000                   # If positions 14-15 are 0,
    m = BitLogicAnd(BitLogicRightShift(y, 16), 2)
                                        # add 2 to n and shift x left 2.
    n = n + m
    vx_5 = BitLogicLeftShift(vx_4, m)

    y = BitLogicRightShift(vx_5, 14)    # Set y = 0, 1, 2, or 3.
    m = BitLogicAnd(
            y,
            BitLogicNegate(
                BitLogicRightShift(y, 1)
                )
            )                           # Set m = 0, 1, 2, or 2 resp.
    return n + 2 - m


# NOTES: All multi-element / multi-precision operations must take
# argument field and return result field from most significant to least
# significant
# Result are always a tuple

def generate_twosum(vx, vy, precision=None):
    """Return two optrees for a TwoSum operation.
       The return value is a tuple (hi,lo)=(sum, error).
    """
    s  = Addition(vx, vy, precision=precision)
    _x = Subtraction(s, vy, precision=precision)
    _y = Subtraction(s, _x, precision=precision)
    dx = Subtraction(vx, _x, precision=precision)
    dy = Subtraction(vy, _y, precision=precision)
    e  = Addition(dx, dy, precision=precision)
    return s, e


def generate_fasttwosum(vx, vy, precision=None):
    """Return two optrees for a FastTwoSum operation.

    Precondition: |vx| >= |vy|.
    The return value is a tuple (sum, error).
    """
    s = Addition(vx, vy, precision=precision, prevent_optimization=True)
    b = Subtraction(s, vx, precision=precision, prevent_optimization=True)
    e = Subtraction(vy, b, precision=precision, prevent_optimization=True)
    return s, e


class MP_Node:
    """ Multi-precision node for error / precision evaluation """
    def __init__(self, precision, epsilon, limb_diff_factor, interval=None):
        # multi-limb format
        self.precision = precision
        # upper bound of the relative error of this
        # node compared to its exact version
        self.epsilon = epsilon
        # upper bound on the factor between to consecutive limbs from
        # hi to lo order
        self.limb_diff_factor = limb_diff_factor
        # range of value
        self.interval = interval

    def __str__(self):
        return "({}, eps={}, ldf={}, interval={})".format(
            self.precision, self.epsilon, self.limb_diff_factor, self.interval
        )

class MetaBlock:
    """ abstract class to document meta-block methods """
    def __init__(self, main_precision, out_precision):
        self.main_precision = main_precision
        self.out_precision = out_precision

    def __str__(self):
        return "{}(main={}, out={})".format(self.__class__.__name__, self.main_precision, self.out_precision)
    def expand(self, *args):
        """ expand an operation into single-word operations """
        raise NotImplementedError

    def global_relative_error_eval(self, *args):
        """ give an upper bound to the relative error of the
            meta-block applied to @p args while taking into
            account the error of the inputs """
        raise NotImplementedError
    def local_relative_error_eval(self, *args):
        """ give an upper bound to the relative error of the result
            ONLY introduced by the meta-block, that is assuming inputs
            were exact """
        raise NotImplementedError

    def relative_error_eval(self, lhs, rhs, global_error=True):
        # TODO: only works for 2-operand meta blocks
        if global_error:
            return self.global_relative_error_eval(lhs, rhs)
        else:
            return self.local_relative_error_eval(lhs, rhs)

    def get_output_descriptor(self, lhs, rhs, global_error=True):
        """ return a MultiLimb object describing the output
            when inputs are @p input_desc """
        # TODO: only valid for 2-operand operation
        raise NotImplementedError

    def check_input_descriptors(self, *input_decs):
        """ Verify that input descriptors @p match the condition
            of the operator (for example overlap)
            @return [bool] check result """
        raise NotImplementedError

def MB_CommutedVersion(BaseClass):
    """ Build reflexive version of BaseClass """
    def decorator(OpClass):
        class NewClass(OpClass, BaseClass):
            def __str__(self):
                return "COM-{}(main={}, out={})".format(BaseClass.__name__, self.main_precision, self.out_precision)
            def expand(self, lhs, rhs):
                print("commuting")
                return BaseClass.expand(self, rhs, lhs)
            def check_input_descriptors(self, lhs, rhs):
                print("commuting")
                return BaseClass.check_input_descriptors(self, rhs, lhs)
            def global_relative_error_eval(self, lhs, rhs):
                print("commuting")
                return BaseClass.global_relative_error_eval(self, rhs, lhs)
            def local_relative_error_eval(self, lhs, rhs):
                print("commuting")
                return BaseClass.local_relative_error_eval(self, rhs, lhs)
        return NewClass
    return decorator

class Op_1LimbOut_MetaBlock(MetaBlock):
    """ Virtual operation which returns a 1-limb output """
    def __init__(self, main_precision):
        self.main_precision = main_precision
        self.out_precision = main_precision

class Op_2LimbOut_MetaBlock(MetaBlock):
    """ Virtual operation which returns a 2-limb output """
    def __init__(self, main_precision):
        self.main_precision = main_precision
        self.out_precision = {
            ML_Binary32: ML_SingleSingle,
            ML_Binary64: ML_DoubleDouble
        }[self.main_precision]

class Op_3LimbOut_MetaBlock(MetaBlock):
    """ Virtual operation which returns a 3-limb output """
    def __init__(self, main_precision):
        self.main_precision = main_precision
        self.out_precision = {
            ML_Binary32: ML_TripleSingle,
            ML_Binary64: ML_TripleDouble
        }[self.main_precision]

def is_single_limb_precision(prec):
    return prec in [ML_Binary32, ML_Binary64]
def is_dual_limb_precision(prec):
    return isinstance(prec, ML_FP_MultiElementFormat) and prec.limb_num == 2
def is_tri_limb_precision(prec):
    return isinstance(prec, ML_FP_MultiElementFormat) and prec.limb_num == 3
def limb_prec_match(mp_prec, prec):
    return all(mp_prec.get_limb_precision(i) == prec for i in range(mp_prec.limb_num))

class Op111_MetaBlock(Op_1LimbOut_MetaBlock):
    """ Virtual operation which returns a 2-limb output from two 1-limb inputs """
    def local_relative_error_eval(self, x, y):
        return S2**-self.main_precision.get_mantissa_size()

    def check_input_descriptors(self, lhs, rhs):
        return is_single_limb_precision(lhs.precision) and \
             is_single_limb_precision(rhs.precision) and \
            (rhs.precision == self.main_precision) and \
            (lhs.precision == self.main_precision)

    def get_output_descriptor(self, lhs_desc, rhs_desc, global_error=True):
        epsilon = self.relative_error_eval(lhs_desc, rhs_desc, global_error=global_error)
        return MP_Node(self.out_precision, epsilon, [])

class MB_Mul111(Op111_MetaBlock):
    def expand(self, x, y):
        return (Multiplication(*(x + y), precision=self.main_precision),)
    def global_relative_error_eval(self, x, y):
        eps_op = self.local_relative_error_eval(x, y)
        # error bound (first order approximation)
        return x.epsilon + y.epsilon + eps_op

MB_Mul111_s = MB_Mul111(ML_Binary32)
MB_Mul111_d = MB_Mul111(ML_Binary64)

class MB_Add111(Op111_MetaBlock):
    def expand(self, lhs, rhs):
        return (Addition(*(lhs + rhs), precision=self.main_precision),)
    def global_relative_error_eval(self, lhs, rhs):
        # error bound (first order approximation)
        eps_op = self.local_relative_error_eval(lhs, rhs)
        eps_in = max(lhs.epsilon, rhs.epsilon)
        return eps_op + eps_in + eps_op * eps_in

MB_Add111_s = MB_Add111(ML_Binary32)
MB_Add111_d = MB_Add111(ML_Binary64)

class Op211_ExactMetaBlock(Op_2LimbOut_MetaBlock):
    """ Virtual operation which returns a 2-limb output from two 1-limb inputs """
    def local_relative_error_eval(self, x, y):
        # ExactMetaBlock is exact
        return 0.0

    def check_input_descriptors(self, lhs, rhs):
        lhs_isl = is_single_limb_precision(lhs.precision)
        rhs_isl = is_single_limb_precision(rhs.precision)
        prec_equal = (rhs.precision == self.main_precision) and (lhs.precision == self.main_precision)
        return lhs_isl and rhs_isl and prec_equal

    def get_output_descriptor(self, lhs_desc, rhs_desc, global_error=True):
        epsilon = self.relative_error_eval(lhs_desc, rhs_desc, global_error=global_error)
        return MP_Node(self.out_precision, epsilon, [S2**-self.main_precision.get_mantissa_size()])


class MB_Mul211(Op211_ExactMetaBlock):
    def expand(self, x, y):
        return Mul211(*(x + y), precision=self.main_precision)
    def global_relative_error_eval(self, x, y):
        # ExactMetaBlock is exact
        return x.epsilon + y.epsilon + x.epsilon * y.epsilon


class MB_Mul211_FMA(MB_Mul211):
    def expand(self, x, y):
        return Mul211(*(x + y), precision=self.main_precision, fma=True)


class MB_Add211(Op211_ExactMetaBlock):
    def expand(self, x, y):
        zh, zl = generate_twosum(*(x + y), precision=self.main_precision)
        return zh, zl
    def global_relative_error_eval(self, x, y):
        # TODO: check error approximation bound
        return max(x.epsilon, y.epsilon)


class MB_Mul221(Op_2LimbOut_MetaBlock):
    DELTA_ERR = 4
    def expand(self, lhs, rhs):
        return Mul221(*(lhs + rhs), precision=self.main_precision)

    def check_input_descriptors(self, lhs, rhs):
        return is_dual_limb_precision(lhs.precision) and \
             is_single_limb_precision(rhs.precision) and \
             (rhs.precision == self.main_precision) and \
            limb_prec_match(lhs.precision, self.main_precision)

    def local_relative_error_eval(self, lhs_desc, rhs_desc):
        lhs_prec = lhs_desc.precision
        eps_op = S2**-(lhs_prec.get_limb_precision(0).get_mantissa_size() + lhs_prec.get_limb_precision(1).get_mantissa_size() - self.DELTA_ERR)
        # error bound (first order approximation)
        return eps_op

    def global_relative_error_eval(self, lhs_desc, rhs_desc):
        eps_op = self.local_relative_error_eval(lhs_desc, rhs_desc)
        # error bound (first order approximation)
        eps = lhs_desc.epsilon + rhs_desc.epsilon + eps_op
        return eps

    def get_output_descriptor(self, lhs, rhs, global_error=True):
        epsilon = self.relative_error_eval(lhs, rhs, global_error=global_error)
        return MP_Node(self.out_precision, epsilon, [S2**-self.main_precision.get_mantissa_size()])

@MB_CommutedVersion(MB_Mul221)
class MB_Mul212(Op_2LimbOut_MetaBlock):
    """ Commutated version of MP_Mul221 """
    def global_relative_error_eval(self, rhs_desc, lhs_desc):
        eps_op = self.local_relative_error_eval(rhs_desc, lhs_desc)
        # error bound (first order approximation)
        eps = lhs_desc.epsilon + rhs_desc.epsilon + eps_op
        return eps
    def get_output_descriptor(self, rhs, lhs, global_error=True):
        epsilon = self.relative_error_eval(rhs, lhs, global_error=global_error)
        return MP_Node(self.out_precision, epsilon, [S2**-self.main_precision.get_mantissa_size()])

# Meta-block instanciations with precisions
MB_Add211_ss = MB_Add211(ML_Binary32)
MB_Add211_dd = MB_Add211(ML_Binary64)
MB_Mul211_ss = MB_Mul211(ML_Binary32)
MB_Mul211_dd = MB_Mul211(ML_Binary64)

MB_Mul221_ss = MB_Mul221(ML_Binary32)
MB_Mul221_dd = MB_Mul221(ML_Binary64)
MB_Mul212_ss = MB_Mul212(ML_Binary32)
MB_Mul212_dd = MB_Mul212(ML_Binary64)


def Split(a, precision=None):
    """... splitting algorithm for Dekker TwoMul"""
    cst_value = {
        ML_Binary32: 4097,
        ML_Binary64: 134217729
    }[a.precision]
    s = Constant(cst_value, precision = a.get_precision(), tag = 'fp_split')
    c = Multiplication(s, a, precision=precision)
    tmp = Subtraction(a, c, precision=precision);
    ah = Addition(tmp, c, precision=precision)
    al = Subtraction(a, ah, precision=precision)
    return ah, al

def Mul211(x, y, precision=None, fma=True):
    """ Multi-precision Multiplication HI, LO = x * y """
    zh = Multiplication(x, y, precision=precision)
    if fma == True:
        zl = FMS(x, y, zh, precision=precision)
    else:
        xh, xl = Split(x, precision=precision)
        yh, yl = Split(y, precision=precision)
        r1 = Multiplication(xh, yh, precision=precision)
        r2 = Subtraction(r1, zh, precision=precision)
        r3 = Multiplication(xh, yl, precision=precision)
        r4 = Multiplication(xl, yh, precision=precision)
        r5 = Multiplication(xl, yl, precision=precision)
        r6 = Addition(r2, r3, precision=precision)
        r7 = Addition(r6, r4, precision=precision)
        zl = Addition(r7, r5, precision=precision)
    return zh, zl


def Add_round_to_odd(x, y, precision=None):
    pass

def Add1111(x, y, z, precision=None):
    uh, ul = Add211(y, z, precision=precision)
    th, tl = Add211(x, uh, precision=precision)
    v = Add_round_to_odd(tl, ul, precision=precision)
    return Addition(v, th, precision=precision)

def Add211(x, y, precision=None):
    """ Multi-precision Addition (2sum) HI, LO = x + y
        TODO: missing assumption on input order """
    zh, zl = generate_twosum(x, y, precision)
    return zh, zl

def Add212(xh, yh, yl, precision=None):
    """ Multi-precision Addition:
        HI, LO = xh + [yh:yl] """
    # r = xh + yh
    # s1 = xh - r
    # s2 = s1 + yh
    # s = s2 + yl
    # zh = r + s
    # zl = (r - zh) + s
    r = Addition(xh, yh, precision=precision)
    s1 = Subtraction(xh, r, precision=precision)
    s2 = Addition(s1, yh, precision=precision)
    s = Addition(s2, yl, precision=precision)
    zh = Addition(r, s, precision=precision)
    zl = Addition(Subtraction(r, zh, precision=precision), s, precision=precision)
    return zh, zl

def Add221(xh, xl, yh, precision=None):
    """ Multi-precision Addition:
        HI, LO = [xh:xl] + yh """
    return Add212(yh, xh, xl, precision)

def Add222(xh, xl, yh, yl, precision=None):
    """ Multi-precision Addition:
        HI, LO = [xh:xl] + [yh:yl] """
    rh, rl = generate_twosum(xl, yl, precision=precision)
    th, tl = generate_twosum(rh, xh, precision=precision)
    zh, sl = generate_twosum(th, yh, precision=precision)
    uh, ul = generate_twosum(rl, tl, precision=precision)
    zl, _ = generate_twosum(sl, uh, precision=precision)
    return generate_fasttwosum(zh, zl, precision=precision)
    #r = Addition(xh, yh, precision=precision)
    #s1 = Subtraction(xh, r, precision=precision)
    #s2 = Addition(s1, yh, precision=precision)
    #s3 = Addition(s2, yl, precision=precision)
    #s = Addition(s3, xl, precision=precision)
    #zh = Addition(r, s, precision=precision)
    #zl = Addition(Subtraction(r, zh, precision=precision), s, precision=precision)
    #return zh, zl


class MB_Add222(Op_2LimbOut_MetaBlock):
    def expand(self, lhs, rhs):
        return Add222(*(lhs + rhs), precision=self.main_precision)

    def check_input_descriptors(self, lhs, rhs):
        # TODO: condition on abs(bh) <= 0.75 abs(ah) or
        # sign of bh and ah not checked
        # input limbs overlaps by at most 49 bits (for double precision limbs)
        return is_dual_limb_precision(lhs.precision) and \
            is_dual_limb_precision(rhs.precision) and \
            lhs.limb_diff_factor[0] <= S2**-self.main_precision.get_mantissa_size() and \
            rhs.limb_diff_factor[0] <= S2**-self.main_precision.get_mantissa_size()

    def global_relative_error_eval(self, lhs_desc, rhs_desc):
        eps_op = self.local_relative_error_eval(lhs_desc, rhs_desc)
        eps_in = max(lhs_desc.epsilon, rhs_desc.epsilon)
        return eps_op + eps_in + eps_op * eps_in

    def local_relative_error_eval(self, lhs_desc, rhs_desc):
        # TODO: very approximative
        ESTIMATED_ERROR_FACTOR = 6
        return S2**-(self.main_precision.get_mantissa_size() * 2 - ESTIMATED_ERROR_FACTOR)

    def get_output_descriptor(self, lhs, rhs, global_error=True):
        epsilon = self.relative_error_eval(lhs, rhs, global_error=global_error)
        limb_diff_factors = [
            # no overlap between medium and lo limb
            S2**-self.main_precision.get_mantissa_size()
        ]
        return MP_Node(self.out_precision, epsilon, limb_diff_factors)

MB_Add222_dd = MB_Add222(ML_Binary64)
MB_Add222_ss = MB_Add222(ML_Binary32)

class MB_Add122(Op_1LimbOut_MetaBlock):
    def expand(self, lhs, rhs):
        return Add122(*(lhs + rhs), precision=self.main_precision)

    def check_input_descriptors(self, lhs, rhs):
        # TODO: condition on abs(bh) <= 0.75 abs(ah) or
        # sign of bh and ah not checked
        # input limbs overlaps by at most 49 bits (for double precision limbs)
        return is_dual_limb_precision(lhs.precision) and \
            is_dual_limb_precision(rhs.precision) and \
            lhs.limb_diff_factor[0] <= S2**-self.main_precision.get_mantissa_size() and \
            rhs.limb_diff_factor[0] <= S2**-self.main_precision.get_mantissa_size()

    def global_relative_error_eval(self, lhs_desc, rhs_desc):
        eps_op = self.local_relative_error_eval(lhs_desc, rhs_desc)
        eps_in = max(lhs_desc.epsilon, rhs_desc.epsilon)
        return eps_op + eps_in + eps_op * eps_in

    def local_relative_error_eval(self, lhs_desc, rhs_desc):
        # TODO: very approximative
        return S2**-self.main_precision.get_mantissa_size()

    def get_output_descriptor(self, lhs, rhs, global_error=True):
        epsilon = self.relative_error_eval(lhs, rhs, global_error=global_error)
        return MP_Node(self.out_precision, epsilon, [])

MB_Add122_d = MB_Add122(ML_Binary64)
MB_Add122_s = MB_Add122(ML_Binary32)

def Add122(xh, xl, yh, yl, precision=None):
    """ Multi-precision Addition:
        HI = [xh:xl] + [yh:yl] """
    zh, _ = Add222(xh, xl, yh, yl, precision)
    return zh,

def Add121(xh, xl, yh, precision=None):
    """ Multi-precision Addition:
        HI = [xh:xl] + yh """
    zh, _ = Add221(xh, xl, yh, precision)
    return zh,

def Add112(xh, yh, yl, precision=None):
    """ Multi-precision Addition:
        HI = xh + [yh:yl] """
    zh, _ = Add212(xh, yh, yl, precision)
    return zh,

def Mul212(x, yh, yl, precision=None, fma=True):
    """ Multi-precision Multiplication:
        HI, LO = x * [yh:yl],
        error <= 2^-102
    """
    t1, t2 = Mul211(x, yh, precision, fma)
    t3 = Multiplication(x, yl, precision=precision)
    t4 = Addition(t2, t3, precision=precision)
    return Add211(t1, t4, precision)
def Mul221(xh, xl, y, precision=None, fma=True):
    """ Multi-precision Multiplication:
        HI, LO = [xh:xl] * y """
    return Mul212(y, xh, xl, precision=precision, fma=fma)

def Mul222(xh, xl, yh, yl, precision=None, fma=True):
    """ Multi-precision Multiplication:
        HI, LO = [xh:xl] * [yh:yl] """
    if fma == True:
        # ph, pl = xh * yh
        ph = Multiplication(xh, yh, precision=precision, prevent_optimization=True)
        pl = FMS(xh, yh, ph, precision=precision)
        # pl += xh * yl
        pl = FMA(xh, yl, pl, precision=precision)
        # pl += xl * yh
        pl = FMA(xl, yh, pl, precision=precision)
        zh, zl = generate_fasttwosum(ph, pl, precision=precision)
        #zh = Addition(ph, pl, precision=precision)
        #zl = Subtraction(ph, zh, precision=precision)
        #zl = Addition(zl, pl, precision=precision)
    else:
        t1, t2 = Mul211(xh, yh, precision, fma)
        t3 = Multiplication(xh, yl, precision=precision)
        t4 = Multiplication(xl, yh, precision=precision)
        t5 = Addition(t3, t4, precision=precision)
        t6 = Addition(t2, t5, precision=precision)
        zh, zl = Add211(t1, t6, precsion);
    return zh, zl

def Mul122(xh, xl, yh, yl, precision=None):
    """ Multi-precision Multiplication:
        HI = [xh:xl] * [yh:yl] """
    zh, _ = Mul222(xh, xl, yh, yl, precision)
    return zh,

def Mul121(xh, xl, yh, precision=None):
    """ Multi-precision Multiplication:
        HI = [xh:xl] * yh """
    zh, _ = Mul221(xh, xl, yh, precision)
    return zh,

def Mul112(xh, yh, yl, precision=None):
    """ Multi-precision Multiplication:
        HI = xh * [yh:yl] """
    return Mul121(yh, yl, xh, precision)

class MB_Mul222(Op_2LimbOut_MetaBlock):
    def expand(self, lhs, rhs):
        return Mul222(*(lhs + rhs), precision=self.main_precision)

    def check_input_descriptors(self, lhs, rhs):
        # TODO: condition on abs(bh) <= 0.75 abs(ah) or
        # sign of bh and ah not checked
        # input limbs overlaps by at most 49 bits (for double precision limbs)
        return is_dual_limb_precision(lhs.precision) and \
            is_dual_limb_precision(rhs.precision) and \
            lhs.limb_diff_factor[0] <= S2**-self.main_precision.get_mantissa_size() and \
            rhs.limb_diff_factor[0] <= S2**-self.main_precision.get_mantissa_size()

    def global_relative_error_eval(self, lhs_desc, rhs_desc):
        eps_op = self.local_relative_error_eval(lhs_desc, rhs_desc)
        eps_in = max(lhs_desc.epsilon, rhs_desc.epsilon)
        return eps_op + eps_in + eps_op * eps_in

    def local_relative_error_eval(self, lhs_desc, rhs_desc):
        # TODO: very approximative
        ESTIMATED_ERROR_FACTOR = 4
        return S2**-(self.main_precision.get_mantissa_size() * 2 - ESTIMATED_ERROR_FACTOR)

    def get_output_descriptor(self, lhs, rhs, global_error=True):
        epsilon = self.relative_error_eval(lhs, rhs, global_error=global_error)
        limb_diff_factors = [
            # no overlap between medium and lo limb
            S2**-self.main_precision.get_mantissa_size()
        ]
        return MP_Node(self.out_precision, epsilon, limb_diff_factors)

MB_Mul222_dd = MB_Mul222(ML_Binary64)
MB_Mul222_ss = MB_Mul222(ML_Binary32)

class MB_Mul122(Op_1LimbOut_MetaBlock):
    def expand(self, lhs, rhs):
        return Mul122(*(lhs + rhs), precision=self.main_precision)

    def check_input_descriptors(self, lhs, rhs):
        # TODO: condition on abs(bh) <= 0.75 abs(ah) or
        # sign of bh and ah not checked
        # input limbs overlaps by at most 49 bits (for double precision limbs)
        return is_dual_limb_precision(lhs.precision) and \
            is_dual_limb_precision(rhs.precision) and \
            lhs.limb_diff_factor[0] <= S2**-self.main_precision.get_mantissa_size() and \
            rhs.limb_diff_factor[0] <= S2**-self.main_precision.get_mantissa_size()

    def global_relative_error_eval(self, lhs_desc, rhs_desc):
        eps_op = self.local_relative_error_eval(lhs_desc, rhs_desc)
        eps_in = max(lhs_desc.epsilon, rhs_desc.epsilon)
        return eps_op + eps_in + eps_op * eps_in

    def local_relative_error_eval(self, lhs_desc, rhs_desc):
        # TODO: very approximative
        return S2**-(self.main_precision.get_mantissa_size())

    def get_output_descriptor(self, lhs, rhs, global_error=True):
        epsilon = self.relative_error_eval(lhs, rhs, global_error=global_error)
        limb_diff_factors = [
            # no overlap between medium and lo limb
            S2**-self.main_precision.get_mantissa_size()
        ]
        return MP_Node(self.out_precision, epsilon, limb_diff_factors)

MB_Mul122_d = MB_Mul122(ML_Binary64)
MB_Mul122_s = MB_Mul122(ML_Binary32)

def MP_FMA2111(x, y, z, precision=None, fma=True):
    mh, ml = Mul211(x, y, precision=precision, fma=fma)
    ah, al = Add221(mh, ml, z, precision=precision)
    return ah, al

def MP_FMA2211(xh, xl, y, z, precision=None, fma=True):
    mh, ml = Mul221(xh, xl, y, precision=precision, fma=fma)
    ah, al = Add221(mh, ml, z, precision=precision)
    return ah, al
def MP_FMA2121(x, yh, yl, z, precision=None, fma=True):
    return MP_FMA2211(yh, yl, x, z, precision=None, fma=True)

def MP_FMA2112(x, y, zh, zl, precision=None, fma=True):
    mh, ml = Mul211(x, y, precision=precision, fma=fma)
    ah, al = Add222(mh, ml, zh, zl, precision=precision)
    return ah, al

def MP_FMA2122(x, yh, yl, zh, zl, precision=None, fma=True):
    mh, ml = Mul212(x, yh, yl, precision=precision, fma=fma)
    ah, al = Add222(mh, ml, zh, zl, precision=precision)
    return ah, al

def MP_FMA2212(xh, xl, y, zh, zl, precision=None, fma=True):
    mh, ml = Mul212(xh, xl, y, precision=precision, fma=fma)
    ah, al = Add222(mh, ml, zh, zl, precision=precision)
    return ah, al

def MP_FMA2222(xh, xl, yh, yl, zh, zl, precision=None, fma=True):
    mh, ml = Mul222(xh, xl, yh, yl, precision=precision, fma=fma)
    ah, al = Add222(mh, ml, zh, zl, precision=precision)
    return ah, al

class MB_Add333(Op_3LimbOut_MetaBlock):
    def expand(self, lhs, rhs):
        return MP_Add333(*(lhs + rhs), precision=self.main_precision)

    def check_input_descriptors(self, lhs, rhs):
        # TODO: condition on abs(bh) <= 0.75 abs(ah) or
        # sign of bh and ah not checked
        # input limbs overlaps by at most 49 bits (for double precision limbs)
        return is_tri_limb_precision(lhs.precision) and \
            is_tri_limb_precision(rhs.precision) and \
            lhs.limb_diff_factor[0] <= S2**-4 and \
            lhs.limb_diff_factor[1] <= S2**-4 and \
            rhs.limb_diff_factor[0] <= S2**-4 and \
            rhs.limb_diff_factor[1] <= S2**-4

    def global_relative_error_eval(self, lhs_desc, rhs_desc):
        eps_op = self.local_relative_error_eval(lhs_desc, rhs_desc)
        eps_in = max(lhs_desc.epsilon, rhs_desc.epsilon)
        return eps_op + eps_in + eps_op * eps_in
    def local_relative_error_eval(self, lhs_desc, rhs_desc):
        # TODO: 47 and 98 are specialized for ML_Binary64
        a_o = sollya.floor(-sollya.log2(lhs_desc.limb_diff_factor[0]))
        a_u = sollya.floor(-sollya.log2(lhs_desc.limb_diff_factor[1]))
        b_o = sollya.floor(-sollya.log2(rhs_desc.limb_diff_factor[0]))
        b_u = sollya.floor(-sollya.log2(rhs_desc.limb_diff_factor[1]))
        return S2**(-min(a_o + a_u, b_o+b_u) - 47) + S2**(-min(a_o,a_u)-98)

    def get_output_descriptor(self, lhs_desc, rhs_desc, global_error=True):
        epsilon = self.relative_error_eval(lhs_desc, rhs_desc, global_error=global_error)
        a_o = sollya.floor(-sollya.log2(lhs_desc.limb_diff_factor[0]))
        b_o = sollya.floor(-sollya.log2(rhs_desc.limb_diff_factor[0]))
        limb_diff_factors = [
            S2**(-min(a_o, b_o) + 5),
            # no overlap between medium and lo limb
            S2**-self.main_precision.get_mantissa_size()
        ]
        return MP_Node(self.out_precision, epsilon, limb_diff_factors)



class MB_Add332(Op_3LimbOut_MetaBlock):
    def expand(self, lhs, rhs):
        return MP_Add332(*(lhs + rhs), precision=self.main_precision)

    def check_input_descriptors(self, lhs, rhs):
        # TODO: condition on abs(bh) <= 0.75 abs(ah) or
        # sign of bh and ah not checked
        # input limbs overlaps by at most 49 bits (for double precision limbs)
        # 2-limb input must not overlap
        return is_tri_limb_precision(lhs.precision) and \
            is_dual_limb_precision(rhs.precision) and \
            lhs.limb_diff_factor[0] <= S2**-2 and \
            lhs.limb_diff_factor[1] <= S2**-1 and \
            rhs.limb_diff_factor[0] <= S2**-self.main_precision.get_mantissa_size()

    def global_relative_error_eval(self, lhs_desc, rhs_desc):
        eps_op = self.local_relative_error_eval(lhs_desc, rhs_desc)
        eps_in = max(lhs_desc.epsilon, rhs_desc.epsilon)
        return eps_op + eps_in + eps_op * eps_in
    def local_relative_error_eval(self, lhs_desc, rhs_desc):
        # TODO: 52, 104, 153 are specialized for ML_Binary64
        b_o = sollya.floor(-sollya.log2(rhs_desc.limb_diff_factor[0]))
        b_u = sollya.floor(-sollya.log2(rhs_desc.limb_diff_factor[1]))
        return S2**(-b_o-b_u-52) + S2**(-b_o-104) + S2**-153

    def get_output_descriptor(self, lhs_desc, rhs_desc, global_error=True):
        epsilon = self.relative_error_eval(lhs_desc, rhs_desc, global_error=global_error)
        b_o = sollya.floor(-sollya.log2(rhs_desc.limb_diff_factor[0]))
        b_u = sollya.floor(-sollya.log2(rhs_desc.limb_diff_factor[1]))
        limb_diff_factors = [
            S2**(-min(45, b_o - 4, b_o + b_u + 2)),
            # no overlap between medium and lo limb
            S2**-self.main_precision.get_mantissa_size()
        ]
        return MP_Node(self.out_precision, epsilon, limb_diff_factors)

@MB_CommutedVersion(MB_Add332)
class MB_Add323(Op_3LimbOut_MetaBlock):
    """ Commuted version of Add332 """
    def get_output_descriptor(self, rhs_desc, lhs_desc, global_error=True):
        epsilon = self.relative_error_eval(rhs_desc, lhs_desc, global_error=global_error)
        b_o = sollya.floor(-sollya.log2(rhs_desc.limb_diff_factor[0]))
        b_u = sollya.floor(-sollya.log2(rhs_desc.limb_diff_factor[1]))
        limb_diff_factors = [
            S2**(-min(45, b_o - 4, b_o + b_u + 2)),
            # no overlap between medium and lo limb
            S2**-self.main_precision.get_mantissa_size()
        ]
        return MP_Node(self.out_precision, epsilon, limb_diff_factors)

def MP_Add333(xh, xm, xl, yh, ym, yl, precision=None):
    rh, t1 = Add211(xh, yh, precision=precision)
    t2, t3 = Add211(xm, ym, precision=precision)
    t7, t4 = Add211(t1, t2, precision=precision)
    t6 = Addition(xl, yl, precision=precision)
    t5 = Addition(t3, t4, precision=precision)
    t8 = Addition(t5, t6, precision=precision)
    rm, rl = Add211(t7, t8, precision=precision)
    return rh, rm, rl

def MP_Add332(xh, xm, xl, yh, yl, precision=None):
    rh, t1 = Add211(xh, yh, precision=precision)
    t2, t3 = Add211(xm, yl, precision=precision)
    t7, t4 = Add211(t1, t2, precision=precision)
    t6 = xl
    t5 = Addition(t3, t4, precision=precision)
    t8 = Addition(t5, t6, precision=precision)
    rm, rl = Add211(t7, t8, precision=precision)
    return rh, rm, rl

def MP_Add323(xh, xl, yh, ym, yl, precision=None):
    return MP_Add332(yh, ym, yl, xh, xl)

def MP_Add321(xh, xl, y, precision=None):
    rh, t1 = Add211(xh, y, precision=precision)
    rm, rl = Add211(t1, xl, precision=precision)
    return rh, rm, rl

def MP_Add312(x, yh, yl, precision=None):
    return MP_Add321(yh, yl, x, precision)


class MB_Add321(Op_3LimbOut_MetaBlock):
    def expand(self, lhs, rhs):
        return MP_Add321(*(lhs + rhs), precision=self.main_precision)

    def check_input_descriptors(self, lhs, rhs):
        return is_dual_limb_precision(lhs.precision) and \
            is_single_limb_precision(rhs.precision) and \
            is_interval_lt(rhs, S2**-2, lhs) and \
             lhs.limb_diff_factor[0] <= S2**-self.main_precision.get_mantissa_size()

    def global_relative_error_eval(self, lhs_desc, rhs_desc):
        eps_op = self.local_relative_error_eval(lhs_desc, rhs_desc)
        eps_in = max(lhs_desc.epsilon, rhs_desc.epsilon)
        return eps_op + eps_in + eps_op * eps_in
    def local_relative_error_eval(self, lhs_desc, rhs_desc):
        return 0.0

    def get_output_descriptor(self, lhs_desc, rhs_desc, global_error=True):
        epsilon = self.relative_error_eval(lhs_desc, rhs_desc, global_error=global_error)
        limb_diff_factors = [
            S2**(-self.main_precision.get_mantissa_size()+1),
            # no overlap between medium and lo limb
            S2**-self.main_precision.get_mantissa_size()
        ]
        return MP_Node(self.out_precision, epsilon, limb_diff_factors)

@MB_CommutedVersion(MB_Add321)
class MB_Add312(Op_3LimbOut_MetaBlock):
    def get_output_descriptor(self, rhs_desc, lhs_desc, global_error=True):
        epsilon = self.relative_error_eval(rhs_desc, lhs_desc, global_error=global_error)
        limb_diff_factors = [
            S2**(-self.main_precision.get_mantissa_size()+1),
            # no overlap between medium and lo limb
            S2**-self.main_precision.get_mantissa_size()
        ]
        return MP_Node(self.out_precision, epsilon, limb_diff_factors)

def MP_Add322(xh, xl, yh, yl, precision=None):
    # TODO use fast2sum when xh <= 2**-2 yh condition has been verified
    rh, t1 = Add211(xh, yh, precision=precision)
    t2, t3 = Add211(al, bl, precision=precision)
    t4, t5 = Add211(t1, t2, precision=precision)
    t6 = Addition(t3, r5, precision=precision)
    rm, rl = Addition(t4, t6)
    return rh, rm, rl

def is_interval_le(op0, factor, op1):
    """ is op0 <= factor * op1 verifier """
    return True
    # TODO: fix
    # return sollya.sup(abs(op0.interval)) <= fector * sollya.inf(abs(op1.interval))

def is_interval_lt(op0, factor, op1):
    """ is op0 < factor * op1 verifier """
    return True
    # TODO: fix
    # return sollya.sup(abs(op0.interval)) < factor * sollya.inf(abs(op1.interval))

class MB_Add322(Op_3LimbOut_MetaBlock):
    def expand(self, lhs, rhs):
        return MP_Add322(*(lhs + rhs), precision=self.main_precision)

    def check_input_descriptors(self, lhs, rhs):
        return is_dual_limb_precision(lhs.precision) and \
            is_dual_limb_precision(rhs.precision) and \
            is_interval_le(rhs, S2**-2, lhs) and \
            lhs.limb_diff_factor[0] <= S2**-self.main_precision.get_mantissa_size() and \
            rhs.limb_diff_factor[0] <= S2**-self.main_precision.get_mantissa_size()

    def global_relative_error_eval(self, lhs_desc, rhs_desc):
        eps_op = self.local_relative_error_eval(lhs_desc, rhs_desc)
        eps_in = max(lhs_desc.epsilon, rhs_desc.epsilon)
        return eps_op + eps_in + eps_op * eps_in
    def local_relative_error_eval(self, lhs_desc, rhs_desc):
        # TODO: dummy value
        return S2**-120

    def get_output_descriptor(self, lhs_desc, rhs_desc, global_error=True):
        epsilon = self.relative_error_eval(lhs_desc, rhs_desc, global_error=global_error)
        limb_diff_factors = [
            S2**(-self.main_precision.get_mantissa_size()+1),
            # no overlap between medium and lo limb
            S2**-self.main_precision.get_mantissa_size()
        ]
        return MP_Node(self.out_precision, epsilon, limb_diff_factors)


def MP_Add313(x, yh, ym, yl, precision=None):
    rh , t1 = Add211(x, yh, precision=precision)
    t2, t3 = Add211(t1, ym, precision=precision)
    t4 = Addition(t3, yl, precision=precision)
    rm, rl = Add211(t2, t4, precision=precision)
    return rh, rm, rl

class MB_Add331(Op_3LimbOut_MetaBlock):
    def expand(self, lhs, rhs):
        return MP_Add313(*(rhs + lhs), precision=self.main_precision)

    def check_input_descriptors(self, lhs, rhs):
        return is_tri_limb_precision(lhs.precision) and \
            is_single_limb_precision(rhs.precision) and \
            is_interval_lt(lhs, S2**-2, rhs) and \
            lhs.limb_diff_factor[0] <= S2**-2 and \
            lhs.limb_diff_factor[1] <= S2**-1

    def global_relative_error_eval(self, lhs_desc, rhs_desc):
        eps_op = self.local_relative_error_eval(lhs_desc, rhs_desc)
        eps_in = max(lhs_desc.epsilon, rhs_desc.epsilon)
        return eps_op + eps_in + eps_op * eps_in
    def local_relative_error_eval(self, lhs_desc, rhs_desc):
        b_o = sollya.floor(-sollya.log2(lhs_desc.limb_diff_factor[0]))
        b_u = sollya.floor(-sollya.log2(lhs_desc.limb_diff_factor[1]))
        return S2**(-52-b_o-b_u) + S2**-154

    def get_output_descriptor(self, lhs_desc, rhs_desc, global_error=True):
        epsilon = self.relative_error_eval(lhs_desc, rhs_desc, global_error=global_error)
        b_o = sollya.floor(-sollya.log2(lhs_desc.limb_diff_factor[0]))
        b_u = sollya.floor(-sollya.log2(lhs_desc.limb_diff_factor[1]))
        limb_diff_factors = [
            S2**(min(-47,3-b_o,1-b_o-b_u)),
            # no overlap between medium and lo limb
            S2**-self.main_precision.get_mantissa_size()
        ]
        return MP_Node(self.out_precision, epsilon, limb_diff_factors)

@MB_CommutedVersion(MB_Add331)
class MB_Add313(Op_3LimbOut_MetaBlock):
    """ """
    def get_output_descriptor(self, rhs_desc, lhs_desc, global_error=True):
        epsilon = self.relative_error_eval(rhs_desc, lhs_desc, global_error=global_error)
        b_o = sollya.floor(-sollya.log2(lhs_desc.limb_diff_factor[0]))
        b_u = sollya.floor(-sollya.log2(lhs_desc.limb_diff_factor[1]))
        limb_diff_factors = [
            S2**(min(-47,3-b_o,1-b_o-b_u)),
            # no overlap between medium and lo limb
            S2**-self.main_precision.get_mantissa_size()
        ]
        return MP_Node(self.out_precision, epsilon, limb_diff_factors)

MB_Add333_ts = MB_Add333(ML_Binary32)
MB_Add332_ts = MB_Add332(ML_Binary32)
MB_Add323_ts = MB_Add323(ML_Binary32)
MB_Add322_ts = MB_Add322(ML_Binary32)
MB_Add321_ts = MB_Add321(ML_Binary32)
MB_Add312_ts = MB_Add312(ML_Binary32)

MB_Add333_td = MB_Add333(ML_Binary64)
MB_Add332_td = MB_Add332(ML_Binary64)
MB_Add323_td = MB_Add323(ML_Binary64)
MB_Add322_td = MB_Add322(ML_Binary64)
MB_Add321_td = MB_Add321(ML_Binary64)
MB_Add312_td = MB_Add312(ML_Binary64)
MB_Add331_td = MB_Add331(ML_Binary64)
MB_Add313_td = MB_Add313(ML_Binary64)

def MP_Mul322(xh, xl, yh, yl, precision=None):
    rh, t1 = Mul211(xh, yh, precision=precision)
    t2, t3 = Mul211(xh, yl, precision=precision)
    t4, t5 = Mul211(xl, yh, precision=precision)
    t6 = Multiplication(xl, yl, precision=precision)
    t7, t8 = Add222(t2, t3, t4, t5, precision=precision)
    t9, t10 = Add211(t1, t6, precision=precision)
    rm, rl = Add222(t7, t8, t9, t10, precision=precision)
    return rh, rm, rl

class MB_Mul322(Op_3LimbOut_MetaBlock):
    def expand(self, lhs, rhs):
        return MP_Mul322(*(lhs + rhs), precision=self.main_precision)

    def check_input_descriptors(self, lhs, rhs):
        return is_dual_limb_precision(lhs.precision) and \
            is_dual_limb_precision(rhs.precision) and \
            lhs.limb_diff_factor[0] <= S2**-self.main_precision.get_mantissa_size() and \
            rhs.limb_diff_factor[0] <= S2**-self.main_precision.get_mantissa_size()

    def global_relative_error_eval(self, lhs_desc, rhs_desc):
        # error bound (first order approximation)
        eps = lhs_desc.epsilon + rhs_desc.epsilon + eps_op
        return eps
    def local_relative_error_eval(self, lhs_desc, rhs_desc):
        # TODO: numeric constant specific to ML_Binary64
        return S2**-149

    def get_output_descriptor(self, lhs_desc, rhs_desc, global_error=True):
        epsilon = self.relative_error_eval(lhs_desc, rhs_desc, global_error=global_error)
        limb_diff_factors = [
            S2**-49, # numeric constant specific to ML_Binary64
            # no overlap between medium and lo limb
            S2**-self.main_precision.get_mantissa_size()
        ]
        return MP_Node(self.out_precision, epsilon, limb_diff_factors)

class MB_Mul332(Op_3LimbOut_MetaBlock):
    def expand(self, lhs, rhs):
        return MP_Mul332(*(lhs + rhs), precision=self.main_precision)

    def check_input_descriptors(self, lhs, rhs):
        return is_tri_limb_precision(lhs.precision) and \
            is_dual_limb_precision(rhs.precision) and \
            lhs.limb_diff_factor[1] <= S2**-self.main_precision.get_mantissa_size() and \
            rhs.limb_diff_factor[0] <= S2**-self.main_precision.get_mantissa_size()

    def global_relative_error_eval(self, lhs_desc, rhs_desc):
        # error bound (first order approximation)
        eps = lhs_desc.epsilon + rhs_desc.epsilon + eps_op
        return eps
    def local_relative_error_eval(self, lhs_desc, rhs_desc):
        # TODO: numeric constant specific to ML_Binary64
        print("MB_Mul332.local_relative_error_eval", lhs_desc, rhs_desc)
        b_o = sollya.floor(-sollya.log2(lhs_desc.limb_diff_factor[0]))
        b_u = sollya.floor(-sollya.log2(lhs_desc.limb_diff_factor[1]))
        return (S2**(-99 - b_o) + S2**(-99-b_o-b_u) + S2**-152) / (1 - S2**-53 - S2**(-b_o+1) - S2**(-b_o-b_u+1))

    def get_output_descriptor(self, lhs_desc, rhs_desc, global_error=True):
        print("MB_Mul332.get_output_descriptor", lhs_desc, rhs_desc)
        epsilon = MB_Mul332.relative_error_eval(self, lhs_desc, rhs_desc, global_error=global_error)

        b_o = sollya.floor(-sollya.log2(lhs_desc.limb_diff_factor[0]))
        b_u = sollya.floor(-sollya.log2(lhs_desc.limb_diff_factor[1]))
        gamma = min(48, b_o-4,b_o+b_u-4)
        limb_diff_factors = [
            S2**-(gamma), #TODO: check gamma minus sign
        #    # no overlap between medium and lo limb
            S2**-self.main_precision.get_mantissa_size()
        ]
        return MP_Node(self.out_precision, epsilon, limb_diff_factors)

@MB_CommutedVersion(MB_Mul332)
class MB_Mul323(Op_3LimbOut_MetaBlock):
    """ Commutated version of MB_Mul332 """
    def get_output_descriptor(self, rhs_desc, lhs_desc, global_error=True):
        print("MB_Mul332.get_output_descriptor", rhs_desc, lhs_desc)
        epsilon = MB_Mul332.relative_error_eval(self, rhs_desc, lhs_desc, global_error=global_error)

        b_o = sollya.floor(-sollya.log2(lhs_desc.limb_diff_factor[0]))
        b_u = sollya.floor(-sollya.log2(lhs_desc.limb_diff_factor[1]))
        gamma = min(48, b_o-4,b_o+b_u-4)
        limb_diff_factors = [
            S2**-(gamma), #TODO: check gamma minus sign
        #    # no overlap between medium and lo limb
            S2**-self.main_precision.get_mantissa_size()
        ]
        return MP_Node(self.out_precision, epsilon, limb_diff_factors)
    pass

def MP_Mul323(xh, xl, yh, ym, yl, precision=None):
    rh, t1 = Mul211(xh, yh, precision=precision)
    t2, t3 = Mul211(xh, ym, precision=precision)
    t4, t5 = Mul211(xh, yl, precision=precision)
    t6, t7 = Mul211(xl, yh, precision=precision)
    t8, t9 = Mul211(xl, ym, precision=precision)
    t10 = Multiplication(xl, yl, precision=precision)
    t11, t12 = Add222(t2, t3, t4, t5, precision=precision)
    t13, t14 = Add222(t6, t7, t8, t9, precision=precision)
    t15, t16 = Add222(t11, t12, t13, t14, precision=precision)
    t17, t18 = Add211(t1, t10, precision=precision)
    rm, rl = Add222(t17, t18, t15, t16, precision=precision)
    return rh, rm, rl


MB_Mul332_td = MB_Mul332(ML_Binary64)
MB_Mul323_td = MB_Mul323(ML_Binary64)

def MP_Mul332(xh, xm, xl, yh, yl, precision=None):
    return MP_Mul323(yh, yl, xh, xm, xl, precision=precision)


def MP_Mul333(xh, xm, xl, yh, ym, yl, precision=None):
    rh, t1 = Mul211(xh, yh, precision)
    t2, t3 = Mul211(xh, ym, precision)
    t4, t5 = Mul211(xm, yh, precision)
    t6, t7 = Mul211(xm, ym, precision)

    t8 = Multiplication(xh, yl, precision=precision)
    t9 = Multiplication(xl, yh, precision=precision)
    t10 = Multiplication(xm, yl, precision=precision)
    t11 = Multiplication(xl, ym, precision=precision)

    t12 = Addition(t8, t9, precision=precision)
    t13 = Addition(t10, t11, precision=precision)
    # check between fast2sum and 2sum possibilities
    t14, t15 = Add211(t1, t6, precision=precision)
    t16 = Addition(t7, t15, precision=precision)
    t17 = Addition(t12, t13, precision=precision)
    t18 = Addition(t16, t17, precision=precision)

    t19, t20 = Add211(t14, t18, precision=precision)
    t21, t22 = Add222(t2, t3, t4, t5, precision=precision)
    rm, rl = Add222(t21, t22, t19, t20, precision=precision)

    return rh, rm, rl

class MB_Mul333(Op_3LimbOut_MetaBlock):
    def expand(self, lhs, rhs):
        return MP_Mul333(*(lhs + rhs), precision=self.main_precision)

    def check_input_descriptors(self, lhs, rhs):
        if not is_tri_limb_precision(lhs.precision) or not is_tri_limb_precision(rhs.precision):
            return False
        a_o = sollya.floor(-sollya.log2(lhs.limb_diff_factor[0]))
        a_u = sollya.floor(-sollya.log2(lhs.limb_diff_factor[1]))
        b_o = sollya.floor(-sollya.log2(rhs.limb_diff_factor[0]))
        b_u = sollya.floor(-sollya.log2(rhs.limb_diff_factor[1]))
        return a_o >= 5 and a_u >= 5 and b_o >= 5 and b_u >= 5

    def global_relative_error_eval(self, lhs_desc, rhs_desc):
        # error bound (first order approximation)
        eps = lhs_desc.epsilon + rhs_desc.epsilon + eps_op
        return eps
    def local_relative_error_eval(self, lhs, rhs):
        # TODO: numeric constant specific to ML_Binary64
        a_o = sollya.floor(-sollya.log2(lhs.limb_diff_factor[0]))
        a_u = sollya.floor(-sollya.log2(lhs.limb_diff_factor[1]))
        b_o = sollya.floor(-sollya.log2(rhs.limb_diff_factor[0]))
        b_u = sollya.floor(-sollya.log2(rhs.limb_diff_factor[1]))
        # TODO: coefficient specific for ML_Binary64
        return S2**-151 + S2**(-99-a_o) + S2**(-99-b_o) + \
            S2**(-49-a_o-a_u) + S2**(-49-b_o-b_u) + S2**(50-a_o-b_o-b_u) + \
            S2**(50-a_o-b_o-b_u) + S2**(-101-a_o-b_o) + S2**(-52-a_o-a_u-b_o-b_u)

    def get_output_descriptor(self, lhs_desc, rhs_desc, global_error=True):
        epsilon = self.relative_error_eval(lhs_desc, rhs_desc, global_error=global_error)
        a_o = sollya.floor(-sollya.log2(lhs.limb_diff_factor[0]))
        b_o = sollya.floor(-sollya.log2(rhs.limb_diff_factor[0]))
        # TODO: coefficient specific for ML_Binary64
        g_o = min(48, -4+a_o, -4+b_o, -4+a_o-b_o)
        limb_diff_factors = [
            S2**-(g_o), 
        #    # no overlap between medium and lo limb
            S2**-self.main_precision.get_mantissa_size()
        ]
        return MP_Node(self.out_precision, epsilon, limb_diff_factors)

MB_Mul333_td = MB_Mul333(ML_Binary64)


def MP_Mul313(x, yh, ym, yl, precision=None):
    rh, t2 = Mul211(x, yh, precision)
    t3, t4 = Mul211(x, ym, precision)
    t5 = Multiplication(x, yl, precision=precision)
    t9, t7 = Add211(t2, t3, precision=precision)

    t8 = Addition(t4, t5, precision=precision)
    t10 = Addition(t7, t8, precision=precision)
    rm, rl = Add211(t9, t10, precision=precision)
    return rh, rm, rl

class MB_Mul313(Op_3LimbOut_MetaBlock):
    def expand(self, lhs, rhs):
        return MP_Mul313(*(lhs + rhs), precision=self.main_precision)

    def check_input_descriptors(self, lhs, rhs):
        if not is_single_limb_precision(lhs.precision) or not is_tri_limb_precision(rhs.precision):
            return False
        b_o = sollya.floor(-sollya.log2(rhs.limb_diff_factor[0]))
        b_u = sollya.floor(-sollya.log2(rhs.limb_diff_factor[1]))
        return b_o >= 2 and b_u >= 2

    def global_relative_error_eval(self, lhs_desc, rhs_desc):
        # error bound (first order approximation)
        eps = lhs_desc.epsilon + rhs_desc.epsilon + eps_op
        return eps
    def local_relative_error_eval(self, lhs, rhs):
        # TODO: numeric constant specific to ML_Binary64
        b_o = sollya.floor(-sollya.log2(rhs.limb_diff_factor[0]))
        b_u = sollya.floor(-sollya.log2(rhs.limb_diff_factor[1]))
        # TODO: coefficient specific for ML_Binary64
        return S2**(-49-b_o-b_u) + S2**(-101-b_o) + 2**-156

    def get_output_descriptor(self, lhs, rhs, global_error=True):
        epsilon = self.relative_error_eval(lhs, rhs, global_error=global_error)
        # TODO: coefficient specific for ML_Binary64
        b_o = sollya.floor(-sollya.log2(rhs.limb_diff_factor[0]))
        b_u = sollya.floor(-sollya.log2(rhs.limb_diff_factor[1]))
        g_o = min(47, -5+b_o, -5+b_o+b_u) # FIX sign
        limb_diff_factors = [
            S2**-(g_o), 
        #    # no overlap between medium and lo limb
            S2**-self.main_precision.get_mantissa_size()
        ]
        return MP_Node(self.out_precision, epsilon, limb_diff_factors)

@MB_CommutedVersion(MB_Mul313)
class MB_Mul331(Op_3LimbOut_MetaBlock):
    def get_output_descriptor(self, rhs, lhs, global_error=True):
        epsilon = self.relative_error_eval(rhs, lhs, global_error=global_error)
        # TODO: coefficient specific for ML_Binary64
        b_o = sollya.floor(-sollya.log2(rhs.limb_diff_factor[0]))
        b_u = sollya.floor(-sollya.log2(rhs.limb_diff_factor[1]))
        g_o = min(47, -5-b_o, -5+b_o+b_u) # FIX sign
        limb_diff_factors = [
            S2**-(g_o), 
        #    # no overlap between medium and lo limb
            S2**-self.main_precision.get_mantissa_size()
        ]
        return MP_Node(self.out_precision, epsilon, limb_diff_factors)


MB_Mul331_td = MB_Mul331(ML_Binary64)
MB_Mul313_td = MB_Mul313(ML_Binary64)


def Normalize_33(xh, xm, xl, precision=None):
    t1h, t1l = Add211(xm, xl, precision=precision)
    rh, t2l = Add211(xh, t1h, precision=precision)
    rm, rl = Add211(t2l, t1l, precision=precision)
    return rh, rm, rl

def Normalize_23(xh, xm, xl, precision=None):
    t1h, t1l = Add211(xm, xl, precision=precision)
    rh, t2l = Add211(xh, t1h, precision=precision)
    rl = Addition(t2l, t1l, precision=precision)
    return rh, rl

def subnormalize(x_list, factor, precision=None, fma=True):
    """ x_list is a multi-component number with components ordered from the
        most to the least siginificant.
        x_list[0] must be the rounded evaluation of (x_list[0] + x_list[1] + ...)
        @return the field of x as a floating-point number assuming
        the exponent of the result is exponent(x) + factor
        and managing field subnormalization if required """
    x_hi = x_list[0]
    int_precision=precision.get_integer_format()
    ex = ExponentExtraction(x_hi, precision=int_precision)
    scaled_ex = ex + factor
    # difference betwen x's real exponent and the minimal exponent
    # for a floating of format precision
    CI0 = Constant(0, precision=int_precision)
    CI1 = Constant(1, precision=int_precision)
    delta = Max(
        Min(
            precision.get_emin() - scaled_ex,
            CI0
        ),
        Constant(precision.get_mantissa_size(), precision=int_precision)
    )

    casted_int_x = TypeCast(x_hi, precision=int_precision)

    # compute a constant to be added to a casted floating-point to perform
    # rounding. This constant shall be equivalent to a half-ulp
    round_cst = BitLogicLeftShift(CI1, delta - 1, precision=int_precision)
    pre_rounded_value = TypeCast(casted_int_x + round_cst, precision=precision)

    sticky_shift = precision.get_bit_size() - (delta - 1)
    sticky = BitLogicLeftShift(casted_int_x, sticky_shift, precision=int_precision)
    low_sticky_sign = CI0
    if len(x_list) > 1:
        for x_op in x_list[1:]:
            sticky = BitLogicOr(sticky, x_op)
            low_sticky_sign = BitLogicOr(BitLogicXor(CopySign(x_hi), CopySign(x_op)), low_sticky_sign)
    # does the low sticky (x_list[1:]) differs in signedness from x_hi ?
    parity_bit = BitLogicAnd(
        casted_int_x,
        BitLogicLeftShift(1, delta, precision=int_precision),
        precision=int_precision)

    inc_select = LogicalAnd(
        Equal(sticky, CI0),
        Equal(parity_bit, CI0)
    )

    rounded_value = Select(inc_select, x, pre_rounded_value, precision=precision)
    # cleaning trailing-bits
    return TypeCast(
        BitLogicRightShift(
            BitLogicLeftShift(
                TypeCast(rounded_value, precision=int_precision),
                delta,
                precision=int_precision
            ),
            delta,
            precision=int_precision
        ),
        precision=precision)

def subnormalize_multi(x_list, factor, precision=None, fma=True):
    """ x_list is a multi-component number with components ordered from the
        most to the least siginificant.
        x_list[0] must be the rounded evaluation of (x_list[0] + x_list[1] + ...)
        @return the field of x as a floating-point number assuming
        the exponent of the result is exponent(x) + factor
        and managing field subnormalization if required """
    x_hi = x_list[0]
    int_precision=precision.get_integer_format()
    ex = ExponentExtraction(x_hi, precision=int_precision)
    scaled_ex = Addition(ex, factor, precision=int_precision)
    CI0 = Constant(0, precision=int_precision)
    CI1 = Constant(1, precision=int_precision)

    # difference betwen x's real exponent and the minimal exponent
    # for a floating of format precision
    delta = Max(
        Min(
            Subtraction(
                Constant(precision.get_emin_normal(), precision=int_precision),
                scaled_ex,
                precision=int_precision
            ),
            CI0,
            precision=int_precision
        ),
        Constant(precision.get_mantissa_size(), precision=int_precision),
        precision=int_precision
    )

    round_factor_exp = Addition(delta, ex, precision=int_precision)
    round_factor = ExponentInsertion(round_factor_exp, precision=precision)

    # to force a rounding as if x_hi was of precision p - delta
    # we use round_factor as follows:
    # o(o(round_factor + x_hi) - round_factor)
    if len(x_list) == 2:
        rounded_x_hi = Subtraction(
            Add112(round_factor, x_list[0], x_list[1], precision=precision)[0],
            round_factor,
            precision=precision
        )
    elif len(x_list) == 3:
        rounded_x_hi = Subtraction(
            Add113(round_factor, x_list[0], x_list[1], x_list[2], precision=precision)[0],
            round_factor,
            precision=precision
        )
    else:
        Log.report(Log.Error, "len of x_list: {} is not supported in subnormalize_multi", len(x_list))
        raise NotImplementedError

    return [rounded_x_hi] + [Constant(0, precision=precision) for i in range(len(x_list)-1)]


def get_MB_cost(mb):
    """ return a quick of dirty evaluation of the meta-block "cost"
        which is used to sort them and chose the less "expensive" one
    """
    try:
        return {
                MB_Add333_td: 6,
                MB_Add332_td: 5,
                MB_Add323_td: 5,
                MB_Add331_td: 4.5,
                MB_Add313_td: 4.5,
                MB_Add321_td: 4,
                MB_Add312_td: 4,
                MB_Add322_td: 4.5,
                MB_Add211_dd: 3,

                MB_Add222_dd: 3.8,
                MB_Add122_d: 3.5,
                
                MB_Add111_d: 1,

                MB_Mul111_d: 1,
                MB_Mul211_dd: 5,
                MB_Mul212_dd: 6,
                MB_Mul221_dd: 6,
                MB_Mul122_d: 6.25,
                MB_Mul222_dd: 6.5,
                MB_Mul331_td: 6.5,
                MB_Mul313_td: 6.5,
                MB_Mul332_td: 7,
                MB_Mul323_td: 7,
                MB_Mul333_td: 8,
        }[mb]
    except KeyError as e:
        print("KeyError in get_MB_cost: for {}".format(mb))
        raise e


def get_Addition_MB_compatible_list(lhs, rhs):
    """ return a list of metablock instance implementing an Addition
        and compatible with format descriptor @p lhs and @p rhs
    """
    #return filter(
    #    (lambda mb: mb.check_input_descriptors(lhs, rhs)),
    #    [
    #        MB_Add333_td, MB_Add332_td, MB_Add323_td,
    #        MB_Add321_td, MB_Add312_td, MB_Add322_td,
    #        MB_Add211_dd,
    #        MB_Add111_d,
    #    ]
    #)
    return [mb for mb in
        [
            MB_Add333_td, MB_Add332_td, MB_Add323_td,
            MB_Add321_td, MB_Add312_td, MB_Add322_td,
            MB_Add331_td, MB_Add313_td,
            MB_Add211_dd,
            MB_Add111_d,
            MB_Add122_d,
            MB_Add222_dd,

        ]
     if mb.check_input_descriptors(lhs, rhs)
    ]
def get_Multiplication_MB_compatible_list(lhs, rhs):
    """ return a list of metablock instance implementing a Multiplication
        and compatible with format descriptor @p lhs and @p rhs
    """
    #return filter(
    #    (lambda mb: mb.check_input_descriptors(lhs, rhs)),
    return [mb for mb in
        [
            MB_Mul212_dd, MB_Mul221_dd, MB_Mul211_dd,
            MB_Mul332_td, MB_Mul323_td, MB_Mul333_td,
            MB_Mul111_d,
            MB_Mul222_dd,
            MB_Mul122_d,
            MB_Mul331_td,
            MB_Mul313_td,
        ] if mb.check_input_descriptors(lhs, rhs)
    ]



if __name__ == "__main__":
    vx = MP_Node(ML_Binary64, 0.0, []) 

    meta_block = MB_Add211_dd # min(get_Addition_MB_compatible_list(vx, vx), key=get_MB_cost)
    add_check = meta_block.check_input_descriptors(vx, vx)
    add_format = meta_block.get_output_descriptor(vx, vx, global_error=False)
    print(add_check, add_format)

    meta_block = MB_Mul211_dd # min(get_Multiplication_MB_compatible_list(vx, vx), key=get_MB_cost)
    mul_check = meta_block.check_input_descriptors(vx, vx)
    mul_format = meta_block.get_output_descriptor(vx, vx, global_error=False)
    print(mul_check, mul_format)

    meta_block = min(get_Addition_MB_compatible_list(add_format, mul_format), key=get_MB_cost)
    add_check = meta_block.check_input_descriptors(add_format, mul_format)
    add_format = meta_block.get_output_descriptor(add_format, mul_format, global_error=False)
    print(add_check, add_format)
