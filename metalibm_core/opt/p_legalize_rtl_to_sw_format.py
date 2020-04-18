# -*- coding: utf-8 -*-
# optimization pass to promote a scalar/vector DAG into vector registers

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

from metalibm_core.core.ml_formats import (
    ML_Custom_FixedPoint_Format
)
from metalibm_core.core.ml_hdl_format import (
    RTL_FixedPointFormat, ML_StdLogicVectorFormat
)
import metalibm_core.core.ml_hdl_format as ml_hdl_format

from metalibm_core.core.ml_table import ML_TableFormat

from metalibm_core.core.ml_operations import (
    is_leaf_node
)
from metalibm_core.core.passes import FunctionPass, METALIBM_PASS_REGISTER

from metalibm_core.utility.log_report import Log


# high verbosity log-level for expand_multi_precision pass module
LOG_LEVEL_LEGALIZE_RTL2SW = Log.LogLevel("LegalizeRTL2SWFormat")



class Pass_ExhaustiveSearch(FunctionPass):
    """ Check precision of every node appearing in the graph """
    pass_tag = "exhaustive_search"
    def __init__(self, tag, target):
        FunctionPass.__init__(self, tag, target)

    def execute_on_node(self, node):
        """ execute pass action on node without input traversal """
        raise NotImplementedError

    def execute_on_optree(self, optree, fct, fct_group, memoization_map):
        if optree in memoization_map:
            return memoization_map[optree]
        if not is_leaf_node(optree):
            for op in optree.inputs:
                _ = self.execute_on_optree(op, fct, fct_group, memoization_map)
        result = self.execute_on_node(optree)
        memoization_map[optree] = result
        return result

SW_StdLogic = ML_Custom_FixedPoint_Format(1, 0, signed=False)

def legalize_rtl_to_sw_format(node_format):
    new_format = node_format
    if node_format is ml_hdl_format.ML_StdLogic:
        new_format = SW_StdLogic
    elif isinstance(node_format, RTL_FixedPointFormat):
        new_format = ML_Custom_FixedPoint_Format(
            node_format.get_integer_size(),
            node_format.get_frac_size(),
            signed=node_format.get_signed()
        )
    elif isinstance(node_format, ML_StdLogicVectorFormat):
        new_format = ML_Custom_FixedPoint_Format(
            node_format.get_bit_size(),
            0,
            signed=False)
    return new_format

@METALIBM_PASS_REGISTER
class Pass_LegalizeRTLtoSWFortmat(Pass_ExhaustiveSearch):
    pass_tag = "legalizertl2swformat"
    def __init__(self, target):
        Pass_ExhaustiveSearch.__init__(self, "legalize RTL to SW compatible format", target)

    def execute_on_node(self, node):
        """ execute pass action on node without input traversal """
        node_format = node.get_precision()
        new_format = None
        if node_format is ml_hdl_format.ML_StdLogic:
            new_format = legalize_rtl_to_sw_format(node_format)
        elif isinstance(node_format, ML_TableFormat):
            storage_format = node_format.get_storage_precision()
            new_storage_format = legalize_rtl_to_sw_format(storage_format)
            node_format.set_storage_precision(new_storage_format)
            Log.report(Log.Info, "translating RTL storage format from {} to {}".format(storage_format, new_storage_format))
            Log.report(Log.Info, "translating RTL table format to {}, {}".format(node.get_precision(), node.get_precision().get_storage_precision()))
        elif isinstance(node_format, RTL_FixedPointFormat):
            new_format = legalize_rtl_to_sw_format(node_format)
        elif isinstance(node_format, ML_StdLogicVectorFormat):
            new_format = legalize_rtl_to_sw_format(node_format)
        if not new_format is None:
            Log.report(LOG_LEVEL_LEGALIZE_RTL2SW, "translating RTL format {} to {}".format(node_format, new_format))
            node.set_precision(new_format)
        return new_format

