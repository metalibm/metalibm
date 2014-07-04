# -*- coding: utf-8 -*-


import commands
import time
import os
import sys


METALIBM_DIR="./metalibm"
CURRENT_DIR = os.getcwd()

current_time = time.localtime()
wdir = CURRENT_DIR + "/valid_%s-%s-%s_%sh%s" % (current_time.tm_mday, current_time.tm_mon, current_time.tm_year, current_time.tm_hour, current_time.tm_min)
test_file = "%s/expf_test.c" % wdir

def execute_cmd(cmd):
    print "cmd: "
    print "    ", cmd
    return commands.getoutput(cmd)

def execute_cmd_status(cmd):
    print "cmd: "
    print "    ", cmd
    return commands.getstatusoutput(cmd)

class Target:
    TARGET_LIST = []
    def __init__(self, ml_name = "", compiler = "gcc", runner = "", extra_options = []):
        self.ml_name = ml_name
        self.compiler = compiler
        self.runner = runner
        self.extra_options = extra_options
        Target.TARGET_LIST.append(self)

target_x86_sse4 = Target("sse", "gcc", runner = "", extra_options = ["-msse4"])

SUCCESS_LIST = []
FAIL_LIST = []


for target in Target.TARGET_LIST:
    
    print "\033[33;1m target: %s \033[0;m" % target.ml_name
    print "\033[32;1m generating expf implementation \033[0;m"
    print execute_cmd("mkdir %s" % wdir)
    print execute_cmd("python %s/framework/ml_functions/ml_exp.py --target %s --output %s" % (METALIBM_DIR, target.ml_name, test_file))

    print "\033[32;1m generating expf test \033[0;m"
    print execute_cmd("make -f %s/rfpg/Makefile gen_test NUM_TEST=1000 TEST_FUNCTION=expf FUNCTION_FILE=%s TEST_END=999 LIMITED_TEST=-DLIMITED_TEST PERF_FLAG=-DPERF_TEST" % (METALIBM_DIR, test_file))

    print "\033[32;1m building expf test \033[0;m"
    print execute_cmd("make -f %s/rfpg/Makefile build_test NUM_TEST=1000 TEST_FUNCTION=expf FUNCTION_FILE=%s TEST_END=999 LIMITED_TEST=-DLIMITED_TEST PERF_FLAG=-DPERF_TEST EXTRA_CC_OPTION=%s CC=%s" % (METALIBM_DIR, test_file, ",".join(target.extra_options), target.compiler))


    print "\033[32;1m running expf test \033[0;m"
    TEST_RESULT, TEST_OUTPUT = execute_cmd_status("%s ./k1_exp_fp32_test" % target.runner)
    print TEST_OUTPUT
    if TEST_RESULT == 0:
        print "\033[32;1m TEST SUCCESS for %s \033[0m" % target.ml_name
        SUCCESS_LIST.append(target)
    else:
        print "TR #%s#" % TEST_RESULT
        print "\033[31;1m TEST_FAIL:  \033[0m" 
        FAIL_LIST.append(target)

print "\033[32;1m SUCCESS FOR: "
for target in SUCCESS_LIST:
    print "    ", target.ml_name
print "\033[31;1m FAILURE FOR: "
for target in FAIL_LIST:
    print "    ", target.ml_name
print "\033[0m"

if FAIL_LIST != []:
    sys.exit(1)
else:
    sys.exit(0)

