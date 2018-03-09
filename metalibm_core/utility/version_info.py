# -*- coding: utf-8 -*-

# Version history:
#  0.1d: adding vhdl support (backend, code generation,
#        ml_entity and code_entity)
#
#

""" Version information for Metalibm """

import commands, inspect, os


def extract_git_hash():
    """ extract current git sha1 """
    cwd = os.getcwd()
    script_dir = os.path.dirname(
      os.path.abspath(inspect.getfile(inspect.currentframe()))
    )
    git_sha = commands.getoutput("""cd '%s' > /dev/null && \
            git log -n 1 --pretty=format:"%%H" && \
            cd '%s' > /dev/null """ % (script_dir, cwd))
    return git_sha

GIT_SHA = extract_git_hash()
VERSION_NUM = "0.1d"
VERSION_DESCRIPTION = "alpha"
NOTES = """ metalibm core """
