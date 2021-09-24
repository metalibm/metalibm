# -*- coding: utf-8 -*-
###############################################################################
# This file is part of metalibm (https://github.com/metalibm/metalibm)
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
# Description: optimization pass to expand exception operation
###############################################################################

from metalibm_core.core.ml_operations import ClearException, RaiseException, Statement
from metalibm_core.core.ml_formats import ML_Void, ML_FPE_Type
from metalibm_core.core.passes import METALIBM_PASS_REGISTER

from metalibm_core.code_generation.generator_utility import type_strict_match

from metalibm_core.opt.generic_lowering import GenericLoweringBackend, LoweringAction, Pass_GenericLowering

class RemoveOperation(LoweringAction):
    def __init__(self):
        pass
    def lower_node(self, node):
        return Statement()

    def __call__(self, node):
        return self.lower_node(node)

    def get_source_info(self):
        """ required as implementation origin indicator by AbstractBackend """
        return None

EXCEPTION_LOWERING_TABLE = {
    ClearException: {
        None: {
            lambda _: True: {
                type_strict_match(ML_Void):
                    RemoveOperation(),
            },
        },
    },
    RaiseException: {
        None: {
            lambda _: True: {
                type_strict_match(ML_Void,ML_FPE_Type):
                    RemoveOperation(),
                type_strict_match(ML_Void,ML_FPE_Type, ML_FPE_Type):
                    RemoveOperation(),
            },
        },
    },
}


class ExceptionLoweringBackend(GenericLoweringBackend):
    # adding first default level of indirection
    target_name = "exception_lowering"
    lowering_table = {
        None: EXCEPTION_LOWERING_TABLE
    }

    @staticmethod
    def get_operation_keys(node):
        """ unwrap RegisterAssign to generate operation key from
            node's operation """
        return GenericLoweringBackend.get_operation_keys(node)


@METALIBM_PASS_REGISTER
class ExceptionLowering(Pass_GenericLowering):
    """ pass to lower exception manipulation operations """
    # adding first default level of indirection
    pass_tag = "lowering_exception"
    def __init__(self, _target):
        Pass_GenericLowering.__init__(self,
                                      ExceptionLoweringBackend(),
                                      description="exception lowering")