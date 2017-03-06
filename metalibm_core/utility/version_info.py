# -*- coding: utf-8 -*-

# Version history:
#  0.1d: adding vhdl support (backend, code generation, ml_entity and code_entity)
#
#

import commands, inspect, os

cwd = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
git_sha = commands.getoutput("""cd '%s' > /dev/null && \
        git log -n 1 --pretty=format:"%%H" && \
        cd '%s' > /dev/null """ % (script_dir, cwd))
version_num = "0.1d"
version_description = "alpha"
notes = """ metalibm core """
