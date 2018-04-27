# -*- coding: utf-8 -*-

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
""" Optimization pass which unify pipeline stage id to nodes with undefined
    id value
    """

from metalibm_core.utility.log_report import Log

from metalibm_core.core.passes import OptreeOptimization, Pass
from metalibm_core.core.ml_operations import ML_LeafNode


###############################################################################
# PASS DESCRIPTION:
# The pass implemented in this file processes an optree and  legalize every
# supported node
# the output format
###############################################################################



def optree_set_undefined_stage(optree, stage_id):
    """ Define the init stage id for node optree if no stage has been previously
        define (value was None) """
    if optree.attributes.init_stage is None:
        optree.attributes.init_stage = stage_id
    return optree.attributes.init_stage

def unify_stages_rec(optree, stage_id=None, memoization_map=None):
    """ Recursively propagate a defined stage id to node starting from optree
    """
    memoization_map = {} if memoization_map is None else memoization_map

    # looking into memoization map
    if optree in memoization_map:
        return optree

    # setting stage id if undefined or updating stage_id value
    stage_id = optree_set_undefined_stage(optree, stage_id)

    if isinstance(optree, ML_LeafNode):
        pass
    else:
        for op_input in optree.get_inputs():
            unify_stages_rec(op_input, stage_id, memoization_map)

    memoization_map[optree] = stage_id


class Pass_UnifyPipelineStages(OptreeOptimization):
    """ implementation of pipeline stage uniformisation """
    pass_tag = "unify_pipeline_stages"

    def __init__(self, target):
        """ pass initialization """
        OptreeOptimization.__init__(self, "unify_pipeline_stages", target)

    def execute(self, optree):
        """ pass execution """
        return unify_stages_rec(optree, {})

# register pass
Log.report(
    Log.Info,
    "Registering {}  pass".format(Pass_UnifyPipelineStages.pass_tag)
)
Pass.register(Pass_UnifyPipelineStages)
