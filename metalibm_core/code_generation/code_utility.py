# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/kalray/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2019 Kalray
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
# created:          Aug 10th, 2019
# last-modified:    Aug 10th, 2019
#
# author(s):    Nicolas Brunie (nicolas.brunie@kalray.eu)
# description:  This file contains implementation of basic utility functions
#               for source code generation
###############################################################################


def insert_line_break(full_line, break_char=" \\\n    ", sep=" ", break_len=80, allow_init_break=True):
    """ Break a string into multiple lines
        @param full_line line to be broken
        @param break_char the sub-string inserted when the input line is broken
        @param sep line-breaking is only possible on a sep sub-string
        @param break_len maximal size allowed for an unbroken line
        @param allow_init_break insert break_char even at the start
                                even if 1st sub-string is too long
        @return new string containing broken down version of @p full_line """
    lexem_list = full_line.split(sep)
    result_line = ""
    sub_line = ""
    if len(lexem_list[0]) > break_len and allow_init_break:
        sub_line = sep + lexem_list[0]
    else:
        sub_line = lexem_list[0]
    for lexem in lexem_list[1:]:
        if len(sub_line) + len(lexem) + len(sep) > break_len:
            result_line = result_line + sub_line + break_char
            sub_line = lexem
        else:
            sub_line = sub_line + sep + lexem
    result_line = result_line + sub_line

    return result_line


