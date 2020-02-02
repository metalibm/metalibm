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
# created:          Mar  3rd, 2019
# last-modified:    Mar  3rd, 2019
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################


from .code_utility import insert_line_break
from ..utility import version_info as ml_version_info

from ..utility.log_report import Log


def get_common_git_comment():
    git_comment = "generated using metalibm %s\nsha1 git: %s(%s)\n" % (ml_version_info.VERSION_NUM, ml_version_info.GIT_SHA, "clean" if ml_version_info.GIT_STATUS else "dirty")
    if not ml_version_info.GIT_STATUS:
        git_comment += "\nWARNING: git status was not clean when file was generated !\n\n"
        Log.report(Log.Warning, "git status was not clean when file was generated !")
    else:
        git_comment += "\nINFO: git status was clean when file was generated.\n\n"
    git_comment += "command used for generation:\n  %s\n" % insert_line_break(ml_version_info.extract_cmdline(), break_len=70)

    return git_comment


GLOBAL_GET_GIT_COMMENT = get_common_git_comment

class CodeConfiguration(object):
    """ constants to configure coding style """
    tab = "    "

    @staticmethod
    def get_git_comment():
        return GLOBAL_GET_GIT_COMMENT()


