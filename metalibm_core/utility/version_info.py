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
# Version history:
# - 1.0: cleaning headers
# - ....
# - 0.1d: adding vhdl support (backend, code generation,
#        ml_entity and code_entity)
# - ....
###############################################################################

""" Version information for Metalibm """

import inspect
import os
import subprocess


def extract_git_hash():
    """ extract current git sha1 """
    cwd = os.getcwd()
    script_dir = os.path.join(
        os.path.dirname(
          os.path.abspath(inspect.getfile(inspect.currentframe()))
        ),
        "..",
        "..",
    )
    cmd = [sub for sub in """git log -n 1 --pretty=format:"%H" """.split(" ") if sub != ""]

    try:
        git_sha_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, cwd=script_dir)
        git_sha = git_sha_process.stdout.read()
        return git_sha
    except:
        return "<undefined>"

GIT_SHA = extract_git_hash()
VERSION_NUM = "1.0"
VERSION_DESCRIPTION = "alpha"
NOTES = """ metalibm core """

if __name__ == "__main__":
    print(extract_git_hash())
