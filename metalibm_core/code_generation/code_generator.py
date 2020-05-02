# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/kalray/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2020 Kalray
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
# created:          May  1st, 2020
# last-modified:    May  1st, 2020
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

from metalibm_core.utility.log_report import Log

class CodeGenerator:
    # dict (language -> CodeGenerator class)
    code_gen_classes = {}
    def __init__(self, processor, declare_cst=True, disable_debug=False,
                 libm_compliant=False, default_rounding_mode=None,
                 default_silent=None, language=None, decorate_code=False):
        raise NotImplementedError

    def generate_expr(self, code_object, node, folded=True,
                      result_var=None, initial=False, language=None,
                      force_variable_storing=False):
        raise NotImplementedError


def RegisterCodeGenerator(language_list):
    """ associate a specific code generator class to a language """
    def __register(CodeGenClass):
        for language in language_list:
            if language in CodeGenerator.code_gen_classes:
                if CodeGenerator.code_gen_classes[language] == CodeGenClass:
                    Log.report(Log.Warning, "multiple registration of codegenerator {} for {}",
                               CodeGenClass, language)
                else:
                    Log.report(Log.Warning, "language {} has an already registered code generator class {} (when trying to register {})", 
                               language, CodeGenClass.code_gen_classes[language], CodeGenClass)
            print("associating generator {} with language {}".format(CodeGenClass, language))
            CodeGenerator.code_gen_classes[language] = CodeGenClass
        return CodeGenClass
    return __register
