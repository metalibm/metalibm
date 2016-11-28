# -*- coding: utf-8 -*-

# Version history:
#  0.1d: adding vhdl support (backend, code generation, ml_entity and code_entity)
#
#

import commands, inspect, os

script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
cmd = """cd %s &&  git log -n 1 --pretty=format:"%%H" """ % script_dir
print "git cmd: ", cmd
git_sha = commands.getoutput(cmd)
version_num = "0.1d"
version_description = "alpha"
notes = """ metalibm core """
