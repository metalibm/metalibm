# -*- coding: utf-8 -*-
#
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


## Legalize the precision of a datapath by finely tuning the size
#  of each operations (limiting width while preventing overflow)
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
