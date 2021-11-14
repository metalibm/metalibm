# -*- coding: utf-8 -*-
###############################################################################
# This file is part of metalibm (https://github.com/kalray/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2021 Nicolas Brunie
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
# created:          Nov  14th, 2021
# last-modified:    Nov  14th, 2021
#
# Author(s): Nicolas Brunie <nicolas.brunie@sifive.com>,
# Description: optimization pass to legalize a vector graph to RVV types
###############################################################################


from metalibm_core.core.ml_operations import is_leaf_node
from metalibm_core.core.vla_common import VLAType
from metalibm_core.targets.riscv.riscv_vector import RVV_vectorTypeMap

from metalibm_core.utility.log_report import Log

from metalibm_core.core.passes import METALIBM_PASS_REGISTER, FunctionPass, Pass, LOG_PASS_INFO

LOG_RVV_LEGALIZATION_INFO = Log.LogLevel("RVVLegalization")

@METALIBM_PASS_REGISTER
class Pass_RVV_Legalization(FunctionPass):
  """ Vector type legalization pass from generic vector formats
      to RISC-V vector extension vector types """
  pass_tag = "rvv_legalization"
  trans_table = {
  }

  def __init__(self, target):
    FunctionPass.__init__(self, target)
    self.memoization_map = {}
    self.set_descriptor("RVV vector types legalization pass")

  # standard Opt pass API
  def execute_on_optree(self, optree, fct=None, fct_group=None, memoization_map=None):
    return self.legalizeNode(optree)

  def legalizeType(self, nodeType):
    """ legalize node type """
    if not isinstance(nodeType, VLAType):
      # unchanged
      return nodeType
    lmul = nodeType.groupSize
    eltType = nodeType.baseFormat
    return RVV_vectorTypeMap[(lmul, eltType)]

  def isNodeLegal(self, node):
    return not isinstance(node.get_precision(), VLAType)

  def legalizeNode(self, node):
      """ Legalize a graph node by converting its vector type (if any)
          to a legal vector type for a RISC-V target with support for the Vector
          extension """
      if node in self.memoization_map:
        self.memoization_map[node]
      # processing node's inputs if the node is not a leaf
      if not is_leaf_node(node):
        for op in node.get_inputs():
          _ = self.legalizeNode(op)
      # translating node type
      if not self.isNodeLegal(node):
        nodeType = node.get_precision()
        nodeNewType = self.legalizeType(nodeType)
        Log.report(LOG_RVV_LEGALIZATION_INFO, "legalization node {} type from {} to {}", node, nodeType, nodeNewType)
        node.set_precision(nodeNewType)
      # memoization
      self.memoize(node)

  def memoize(self, node):
    self.memoization_map[node] = node.get_precision()