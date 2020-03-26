# -*- coding: utf-8 -*-
""" meta-implementation of two-argument arc-tangent (atan2) function """

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
# created:          Mar 25th, 2020
# last-modified:    Mar 25th, 2020
# Author(s): Nicolas Brunie <nbrunie@kalray.eu>
###############################################################################

from metalibm_core.utility.ml_template import ML_NewArgTemplate
from metalibm_functions.ml_atan import MetaAtan2


if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate(default_arg=MetaAtan2.get_default_args())
    # extra options
    arg_template.get_parser().add_argument(
        "--method", dest="method", default="piecewise", choices=["piecewise", "single"],
        action="store", help="select approximation method")

    args = arg_template.arg_extraction()
    ml_atan2 = MetaAtan2(args)
    ml_atan2.gen_implementation()
