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

###############################################################################
# created:          Nov  6th, 2018
# last-modified:    Nov  6th, 2018
#
# author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
###############################################################################

import ctypes
import hashlib
import subprocess
import os


from metalibm_core.core.ml_formats import (
    ML_Binary32, ML_Binary64,
    ML_Int32, ML_Int64, ML_UInt32, ML_UInt64,
)
from metalibm_core.utility.log_report import Log


def get_cmd_stdout(cmd):
    """ execute cmd on a subprocess and return return-code and stdout
        message """
    cmd_process = subprocess.Popen(
        filter(None, cmd.split(" ")), stdout=subprocess.PIPE, env=os.environ.copy())
    returncode = cmd_process.wait()
    return returncode, cmd_process.stdout.read()


def get_ctype_translate(precision):
    """ translate a Metalibm format object to its ctypes equivalent """
    return {
        ML_Binary64: ctypes.c_double,
        ML_Binary32: ctypes.c_float,
        ML_Int32: ctypes.c_int, # TODO: check size compatibility
        ML_UInt32: ctypes.c_uint, # TODO: check size compatibility
        ML_Int64: ctypes.c_longlong, # TODO: check size compatibility
        ML_UInt64: ctypes.c_ulonglong, # TODO: check size compatibility
    }[precision]


def adapt_ctypes_wrapper_to_code_function(wrapper, code_function):
    """ Adapt a ctypes' function wrapper to match code_function prototype """
    wrapper.restype = get_ctype_translate(code_function.get_output_format())
    wrapper.argtypes = tuple(get_ctype_translate(arg.get_precision()) for arg in code_function.get_arg_list())


class BinaryFile:
    def __init__(self, path, source_object, shared_object=False, main=False):
        self.path = path
        self.source_object = source_object
        self.shared_object = shared_object
        self.main = main

    def load(self):
        return LoadedBinary(self)


class LoadedBinary:
    def __init__(self, binary_file):
        self.binary_file = binary_file
        self.loaded_module = ctypes.CDLL(self.binary_file.path)

    def get_function_handle(self, function_name):
        code_function = self.binary_file.source_object.function_list.get_code_function_by_name(function_name)
        fct_handle = self.loaded_module[function_name]
        adapt_ctypes_wrapper_to_code_function(fct_handle, code_function)
        return fct_handle 

def sha256_file(filename):
    """ return the sha256 checksum of @p filename """
    BLOCKSIZE = 65536
    hasher = hashlib.sha256()
    with open(filename, 'rb') as afile:
        buf = afile.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(BLOCKSIZE)
    return hasher.hexdigest()

class SourceFile:
    def __init__(self, path, function_list):
        self.function_list = function_list
        self.path = path

    def build(self, target, bin_name=None, shared_object=False, link=False):
        """ Build @p self source file for @p target processor 
            Args:
                target: target processor
                bin_name(str): name of the binary file (build result)
                shared_object: build as shared object
                link: enable/disable link
            Return:
                BinaryFile, str (error, stdout) """
        bin_name = bin_name or sha256_file(self.path) 
        compiler = target.get_compiler()
        DEFAULT_OPTIONS = ["-O2", "-DML_DEBUG"]
        compiler_options = " ".join(DEFAULT_OPTIONS + target.get_compilation_options())
        if not(link):
            # build only, disable link
            if shared_object:
                compiler_options += " -fPIC -shared "
            else:
                compiler_options += " -c  "
        else:
            src_list += [
                "%s/metalibm_core/support_lib/ml_libm_compatibility.c" % (os.environ["ML_SRC_DIR"]),
                "%s/metalibm_core/support_lib/ml_multi_prec_lib.c" % (os.environ["ML_SRC_DIR"]),
            ]
        Log.report(Log.Info, "Compiler options: \"{}\"".format(compiler_options))

        build_command = "{compiler} {options} -I{ML_SRC_DIR}/metalibm_core \
        {src_file} -o {bin_name} -lm ".format(
            compiler=compiler,
            src_file=self.path,
            bin_name=bin_name,
            options=compiler_options,
            ML_SRC_DIR=os.environ["ML_SRC_DIR"])

        Log.report(Log.Info, "Building source with command: {}".format(build_command))
        build_result, build_stdout = get_cmd_stdout(build_command)
        Log.report(Log.Verbose, "building stdout {}\n", build_stdout)
        if build_result:
            return None
        else:
            return BinaryFile(bin_name, self, shared_object=shared_object)

        


class BuildProject:
    def __init__(self, source_file_list):
        self.source_file_list = source_file_list
        self.function_list = FunctionGroup()
        for function_group in source_file.function_list:
            self.merge_with_group(function_group)

    def build(self, target, bin_name, shared_object=False, link=False):
        bin_list = [source_file.build(shared_object=shared_object, link=False) for source_file in self.source_file_list]
        if None in bin_list:
            # at least one source file could no be built properly
            return None
        else:
            link_options = "-fPIC -shared" if shared_object else ""
            link_options += "-c" if not link else ""
            link_command = "{compiler} {link_options} {bin_src} -o {bin_name}".format(
                compiler=processor.get_compiler(),
                link_options=link_options,
                bin_src=" ".join(bin_file.path for bin_file in bin_list),
                bin_name=bin_name
            )
            Log.report(Log.Verbose, "linking project with command: {}", link_command)
            link_result, link_stdout = get_cmd_stdout(link_command)
            Log.report(Log.Verbose, "link stdout: {}", link_stdout)
            if link_result:
                return None
            else:
                return BinaryFile(bin_name, self, shared_object=shared_object)

