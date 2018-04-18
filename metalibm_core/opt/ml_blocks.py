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


def generate_twosum(vx, vy):
    """Return two optrees for a TwoSum operation.
 
    The return value is a tuple (sum, error).
    """
    s  = Addition(vx, vy)
    _x = Subtraction(s, vy)
    _y = Subtraction(s, _x)
    dx = Subtraction(vx, _x)
    dy = Subtraction(vy, _y)
    e  = Addition(dx, dy)
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
   
def Split(a):
    """... splitting algorithm for Dekker TwoMul"""
    # if a.get_precision() == ML_Binary32:
    s = Constant(4097, precision = a.get_precision(), tag = 'fp_split')
    # elif a.get_precision() == ML_Binary64:
    #    s = Constant(134217729, precision = a.get_precision(), tag = 'fp_split')
    c = Multiplication(s, a)
    tmp = Subtraction(a, c);
    ah = Addition(tmp, c)
    al = Subtraction(a, ah)
    return ah, al

def Mul211(x, y, fma=True):
    """ Multi-precision Multiplication HI, LO = x * y """
    zh = Multiplication(x, y)
    if fma == True:
        zl = FMS(x, y, zh)
    else:
        xh, xl = Split(x)
        yh, yl = Split(y)
        r1 = Multiplication(xh, yh)
        r2 = Subtraction(r1, zh)
        r3 = Multiplication(xh, yl)
        r4 = Multiplication(xl, yh)
        r5 = Multiplication(xl, yl)
        r6 = Addition(r2, r3)
        r7 = Addition(r6, r4)
        zl = Addition(r7, r5)
    return zh, zl

def Add211(x, y):
    """ Multi-precision Addition (2sum) HI, LO = x + y 
        TODO: missing assumption on input order """
    zh = Addition(x, y)
    t1 = Subtraction(zh, x)
    zl = Subtraction(y, t1)
    return zh, zl

def Mul212(x, yh, yl, fma=True):
    """ Multi-precision Multiplication:
        HI, LO = x * [yh:yl] """
    t1, t2 = Mul211(x, yh, fma)
    t3 = Multiplication(x, yl)
    t4 = Addition(t2, t3)
    return Add211(t1, t4)

def Mul222(xh, xl, yh, yl, fma=True):
    """ Multi-precision Multiplication:
        HI, LO = [xh:xl] * [yh:yl] """
    if fma == True:
        ph = Multiplication(xh, yh)
        pl = FMS(xh, yh, ph)
        pl = FMA(xh, yl, pl)
        pl = FMA(xl, yh, pl)
        zh = Addition(ph, pl)
        zl = Subtraction(ph, zh)
        zl = Addition(zl, pl)
    else:
        t1, t2 = Mul211(xh, yh, fma)
        t3 = Multiplication(xh, yl)
        t4 = Multiplication(xl, yh)
        t5 = Addition(t3, t4)
        t6 = Addition(t2, t5)
        zh, zl = Add211(t1, t6); 
    return zh, zl

def Add212(xh, yh, yl):
    """ Multi-precision Addition:
        HI, LO = xh + [yh:yl] """
    r = Addition(xh, yh)
    s1 = Subtraction(xh, r)
    s2 = Addition(s1, yh)
    s = Addition(s2, yl)
    zh = Addition(r, s)
    zl = Addition(Subtraction(r, zh), s)
    return zh, zl

def Add222(xh, xl, yh, yl):
    """ Multi-precision Addition:
        HI, LO = [xh:xl] + [yh:yl] """
    r = Addition(xh, yh)
    s1 = Subtraction(xh, r)
    s2 = Addition(s1, yh)
    s3 = Addition(s2, yl)
    s = Addition(s3, xl)
    zh = Addition(r, s)
    zl = Addition(Subtraction(r, zh), s)
    return zh, zl

def Add122(xh, xl, yh, yl):
    """ Multi-precision Addition:
        HI = [xh:xl] + [yh:yl] """
    zh, _ = Add222(xh, xl, yh, yl)
    return zh
