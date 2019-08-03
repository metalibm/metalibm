# -*- coding: utf-8 -*-

###############################################################################
# This file is part of metalibm (https://github.com/kalray/metalibm)
###############################################################################
# MIT License
#
# Copyright (c) 2018- Kalray
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
#
###############################################################################
#
# Version history: see ./RELEASES.txt
#
###############################################################################

""" Version information for Metalibm """

import inspect
import os
import subprocess
import sys

# Locate script directory, it will be used to execute a git command
# (as we know the metalibm git should be accessible in the script directory)
SCRIPT_DIR = os.path.join(
    os.path.dirname(
      os.path.abspath(inspect.getfile(inspect.currentframe()))
    ),
    "..",
    "..",
)

def extract_git_hash(exec_dir=SCRIPT_DIR):
    """ extract current git sha1 """
    cmd = [sub for sub in """git log -n 1 --pretty=format:"%H" """.split(" ") if sub != ""]

    try:
        git_sha_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, cwd=exec_dir)
        git_sha = git_sha_process.stdout.read()
        return git_sha
    except:
        return "<undefined>"

def check_git_status(exec_dir=SCRIPT_DIR):
    """ Return True if git status is clean (no pending modification), False
        if it is dirty and None if an error occurs
        @param exec_dir git diff/status command is executed from this directory
    """
    cmd = [sub for sub in """git diff --quiet""".split(" ") if sub != ""]

    try:
        git_status_process = subprocess.Popen(cmd, cwd=exec_dir)
        git_status_process.wait()
        git_status = git_status_process.returncode
        return git_status == 0
    except:
        return None

def extract_cmdline():
    """ Rebuild the command line which one used to generate a function """
    return " ".join(sys.argv)

# statically extracting info once (at module init)
GIT_SHA = extract_git_hash()
VERSION_NUM = "1.1"
VERSION_DESCRIPTION = "alpha"
NOTES = """ metalibm core """
GIT_STATUS = check_git_status()

if __name__ == "__main__":
    print("git hash:   {}".format(extract_git_hash()))
    print("git status: {}".format(check_git_status()))
