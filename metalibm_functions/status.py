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
# last-modified:    Mar  7th, 2018
###############################################################################
from metalibm_core.core.ml_function import ML_FunctionBasis, DefaultArgTemplate


from metalibm_core.utility.ml_template import ML_NewArgTemplate
from metalibm_core.utility.log_report  import Log

import metalibm_core.utility.gappa_utils as gappa
import metalibm_core.core.polynomials as polynomials
import metalibm_core.utility.ml_template as template



class EmptyFunction(ML_FunctionBasis):
    """ Dummy meta-function to generation Metalibm status """
    function_name = "ml_empty_function"
    def __init__(self,arg_template = DefaultArgTemplate):
        # initializing I/O precision
        precision = ArgDefault.select_value([arg_template.precision, precision])
        io_precisions = [precision] * 2

        # initializing base class
        ML_FunctionBasis.__init__(self,
          arg_template=arg_template
        )

        self.precision = precision


if __name__ == "__main__":
    # auto-test
    arg_template = ML_NewArgTemplate()

    arg_template.get_parser().add_argument(
        "--test-gappa", dest="test_gappa", action="store_const", const=True,
        default=False, help="test gappa install")
    arg_template.get_parser().add_argument(
        "--test-cgpe", dest="test_cgpe", action="store_const", const=True,
        default=False, help="test cgpe install")
    arg_template.get_parser().add_argument(
        "--full-status", dest="full_status", action="store_const", const=True,
        default=False, help="test full Metalibm status")
    # argument extraction
    args = arg_template.arg_extraction()

    if args.test_cgpe or args.full_status:
        print("CPGE available:  {}".format(polynomials.is_cgpe_available()))
    if args.test_gappa or args.full_status:
        print("Gappa available: {}".format(gappa.is_gappa_installed()))

    if args.full_status:
        print("List of registered targets:")
        template.list_targets()



