# -*- coding: utf-8 -*-

# Copyrights: Nicolas Brunie (2017)
# email: nibrunie@gmail.com
#
# Created:       August, 8th 2017
# Last-modified: August, 8th 2017

from metalibm_core.core.ml_operations import (
    ML_LeafNode, Comparison
)


def evaluate_comparison_range(optree):
    return None

def is_comparison(optree):
    return isinstance(optree, Comparison)

## Assuming @p optree has no pre-defined range, recursively compute a range
#  from the node inputs
def evaluate_range(optree):
    """ evaluate the range of an Operation node

        Args:
            optree (ML_Operation): input Node

        Return:
            sollya Interval: evaluated range of optree or None if no range
                             could be determined
    """
    init_interval =  optree.get_interval()
    if not init_interval is None:
        return init_interval
    else:
        if isinstance(optree, ML_LeafNode):
            return optree.get_interval()
        elif is_comparison(optree):
            return evaluate_comparison_range(optree)
        else:
            args_interval = tuple(
                evaluate_range(op) for op in
                optree.get_inputs()
            )
            return optree.apply_bare_range_function(args_interval)


def forward_attributes(src, dst):
    """ forward compatible attributes from src node to dst node """
    dst.set_tag(src.get_tag())
    dst.set_debug(src.get_debug())
    dst.set_handle(src.get_handle())
    if hasattr(src.attributes, "init_stage"):
        forward_stage_attributes(src, dst)


def forward_stage_attributes(src, dst):
    dst.attributes.init_stage = src.attributes.init_stage
