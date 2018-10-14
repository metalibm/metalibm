# -*- coding: utf-8 -*-

""" Source-info utilities """

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
# created:          Jul 10th, 2017
# last-modified:    Mar  7th, 2018
#
# Author(s): Nicolas Brunie <nbrunie@kalray.eu>
###############################################################################

import os
import sys
import inspect

class SourceInfo:
    # disabled by default to avoid performance issue
    enabled = False
    def __init__(self, filename, lineno):
        self.filename = filename
        self.lineno = lineno

    def __str__(self):
        filename = os.path.basename(self.filename)
        return "{}:{}".format(filename, self.lineno)

    @staticmethod
    def retrieve_source_info(extra_depth=0):
        if SourceInfo.enabled:
            current_frame = inspect.currentframe()
            frame = inspect.getouterframes(current_frame)[2 + extra_depth]
            frameinfo = frame # inspect.getframeinfo(frame)
            if sys.version_info >= (3, 5):
                return SourceInfo(frameinfo.filename, frameinfo.lineno)
            else:
                return SourceInfo(frameinfo[1], frameinfo[2])
        return SourceInfo("disabled", -1)


if "ENABLE_SOURCE_INFO" in os.environ:
    print("[INFO] enabling SourceInfo tracking (could slow down execution with python3)")
    SourceInfo.enabled = True
