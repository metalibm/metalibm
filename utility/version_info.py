# -*- coding: utf-8 -*-

import commands, inspect, os

script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
git_sha = commands.getoutput("""pushd %s > /dev/null &&  git log -n 1 --pretty=format:"%%H" && popd > /dev/null """ % script_dir)
version_num = "0.1"
version_description = "alpha"
notes = """ metalibm core """
