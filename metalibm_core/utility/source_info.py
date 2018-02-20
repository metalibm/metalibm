# -*- coding: utf-8 -*-

""" Source-info utilities """

###############################################################################
# This file is part of the new Metalibm tool
# Copyright Nicolas Brunie (2017-)
# All rights reserved
# created:          Jul 10th, 2017
# last-modified:    Jul 10th, 2017
#
# author(s): Nicolas Brunie (nibrunie@gmail.com)
###############################################################################

import os
import sys
import inspect

class SourceInfo:
    def __init__(self, filename, lineno):
        self.filename = filename
        self.lineno = lineno

    def __str__(self):
        filename = os.path.basename(self.filename)
        return "{}:{}".format(filename, self.lineno)

    @staticmethod
    def retrieve_source_info(extra_depth=0):
        current_frame = inspect.currentframe()
        frame = inspect.getouterframes(current_frame)[2 + extra_depth]
        frameinfo = frame # inspect.getframeinfo(frame)
        if sys.version_info >= (3, 5):
            return SourceInfo(frameinfo.filename, frameinfo.lineno)
        else:
            return SourceInfo(frameinfo[1], frameinfo[2])
