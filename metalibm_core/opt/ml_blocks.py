# -*- coding: utf-8 -*-
# vim: sw=2 sts=2

# Meta-blocks that return an optree to be used in an optimization pass
# Created:        Fri Apr 28 15:04:25 CEST 2017
# Last modified:  Fri May  5 11:09:17 CEST 2017
# Contributors:
#   Hugues de Lassus <hugues.de-lassus@univ-perp.fr>


from metalibm_core.core.ml_operations import *
from metalibm_core.core.ml_formats import *


# Dynamic implementation of a vectorizable leading zero counter.
# The algorithm is taken from the Hacker's Delight and works only for 32-bit
# registers.
# TODO Adapt to other register sizes.
def generate_count_leading_zeros(vx, precision=ML_UInt32):
  """Generate a vectorizable LZCNT optree."""

  y = - BitLogicRightShift(vx, 16)    # If left half of x is 0,
  m = BitLogicAnd(BitLogicRightShift(y, 16), 16)
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


def generate_twosum(vx, vy, precision=ML_Binary64):
  """Return two optrees for a TwoSum operation.

  The return value is a tuple (sum, error).
  """
  s  = Addition(vx, vy, precision = precision)
  _x = Subtraction(s, vy, precision = precision)
  _y = Subtraction(s, _x, precision = precision)
  dx = Subtraction(vx, _x, precision = precision)
  dy = Subtraction(vy, _y, precision = precision)
  e  = Addition(dx, dy, precision = precision)
  return s, e


def generate_fasttwosum(vx, vy, precision=ML_Binary64):
  """Return two optrees for a FastTwoSum operation.

  Precondition: |vx| >= |vy|.
  The return value is a tuple (sum, error).
  """
  s = Addition(vx, vy, precision = precision)
  b = Subtraction(z, vx, precision = precision)
  e = Subtraction(vy, b, precision = precision)
  return s, e
