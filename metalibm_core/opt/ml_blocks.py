# -*- coding: utf-8 -*-
# vim: sw=2 sts=2

# Meta-blocks that return an optree to be used in an optimization pass
# Created:        Fri Apr 28 15:04:25 CEST 2017
# Last modified:  Fri May  5 11:09:17 CEST 2017
# Contributors:
#   Hugues de Lassus <hugues.de-lassus@univ-perp.fr>


from metalibm_core.core.ml_operations import (
    BitLogicRightShift, BitLogicAnd, BitArithmeticRightShift,
    Subtraction, BitLogicLeftShift, BitLogicNegate, Addition, Multiplication, 
    FusedMultiplyAdd, FMS, FMA
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
   
   
def Mul211(x, y):
    """ Multi-precision Multiplication HI, LO = x * y """
    zh = Multiplication(x, y)
    zl = FusedMultiplyAdd(x, y, zh, specifier = FusedMultiplyAdd.Subtract)
    return zh, zl

def Add211(x, y):
    """ Multi-precision Addition (2sum) HI, LO = x + y 
        TODO: missing assumption on input order """
    zh = Addition(x, y)
    t1 = Subtraction(zh, x)
    zl = Subtraction(y, t1)
    return zh, zl

def Mul212(x, yh, yl):
    """ Multi-precision Multiplication:
        HI, LO = x * [yh:yl] """
    t1, t2 = Mul211(x, yh)
    t3 = Multiplication(x, yl)
    t4 = Addition(t2, t3)
    return Add211(t1, t4)

def Mul222(xh, xl, yh, yl):
    """ Multi-precision Multiplication:
        HI, LO = [xh:xl] * [yh:yl] """
    ph = Multiplication(xh, yh)
    pl = FMS(xh, yh, ph)
    pl = FMA(xh, yl, pl)
    pl = FMA(xl, yh, pl)
    zh = Addition(ph, pl)
    zl = Subtraction(ph, zh)
    zl = Addition(zl, pl)
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
    """ Multi-precision Multiplication:
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
    """ Multi-precision Multiplication:
        HI = [xh:xl] + [yh:yl] """
    zh, _ = Add222(xh, xl, yh, yl)
    return zh
