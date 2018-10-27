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


from metalibm_core.core.ml_operations import (
    BitLogicRightShift, BitLogicAnd, BitArithmeticRightShift,
    Subtraction, BitLogicLeftShift, BitLogicNegate, Addition, Multiplication, 
    ExponentExtraction, ExponentInsertion,
    Max, Min,
    FMS, FMA, Constant
)
from metalibm_core.core.ml_formats import ML_Binary32, ML_Binary64

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
 
    The return value is a tuple (sum, error).
    """
    s  = Addition(vx, vy, precision=precision)
    _x = Subtraction(s, vy, precision=precision)
    _y = Subtraction(s, _x, precision=precision)
    dx = Subtraction(vx, _x, precision=precision)
    dy = Subtraction(vy, _y, precision=precision)
    e  = Addition(dx, dy, precision=precision)
    return s, e


def generate_fasttwosum(vx, vy):
    """Return two optrees for a FastTwoSum operation.
 
    Precondition: |vx| >= |vy|.
    The return value is a tuple (sum, error).
    """
    s = Addition(vx, vy)
    b = Subtraction(z, vx)
    e = Subtraction(vy, b)
    return s, e
   
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
    r = Addition(xh, yh, precision=precision)
    s1 = Subtraction(xh, r, precision=precision)
    s2 = Addition(s1, yh, precision=precision)
    s3 = Addition(s2, yl, precision=precision)
    s = Addition(s3, xl, precision=precision)
    zh = Addition(r, s, precision=precision)
    zl = Addition(Subtraction(r, zh, precision=precision), s, precision=precision)
    return zh, zl

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
        HI, LO = x * [yh:yl] """
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
        ph = Multiplication(xh, yh, precision=precision)
        pl = FMS(xh, yh, ph, precision=precision)
        pl = FMA(xh, yl, pl, precision=precision)
        pl = FMA(xl, yh, pl, precision=precision)
        zh = Addition(ph, pl, precision=precision)
        zl = Subtraction(ph, zh, precision=precision)
        zl = Addition(zl, pl, precision=precision)
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
        Constant(precision.get_field_size(), precision=int_precision)
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
        Constant(precision.get_field_size(), precision=int_precision),
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

    




