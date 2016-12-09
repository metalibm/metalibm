# -*- coding: utf-8 -*-

import commands, inspect, os

cwd = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
git_sha = commands.getoutput("""cd '%s' > /dev/null && \
        git log -n 1 --pretty=format:"%%H" && \
        cd '%s' > /dev/null """ % (script_dir, cwd))
version_num = "0.1c"
version_description = "alpha"
notes = """ metalibm core """
