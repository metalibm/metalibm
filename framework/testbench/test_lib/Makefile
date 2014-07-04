SEED ?= 17
NUM_TEST ?= 100
INF_INTERVAL ?= -104.0f
SUP_INTERVAL ?= 89.0f
TEST_FUNCTION ?= exp_fp32
FUNCTION_FILE ?= ../functions/exp/binary32/exp_fp32.c 
TEST_START ?= 0
TEST_END ?= $(NUM_TEST)
DEBUG_FLAG ?= 
PERF_FLAG ?=
LIMITED_TEST ?=
KML_MODE ?=
RESULT_EXTRACT_DIR ?= ./
RESULT_EXTRACT_FILE ?= ./result_extract.c
TEST_TEMPLATE_FILE ?= test_template.c

K1B_CC=/work1/hardware/users/nbrunie/mppa_top/mppa_fe/tools_k1/bin/k1-gcc
K1B_RUNNER=/work1/hardware/users/nbrunie/mppa_top/mppa_fe/tools_k1/bin/k1-runner


TEST_LIB_DIR?=/work1/hardware/users/nbrunie/tools/metalibm/framework/testbench/test_lib
SUPPORT_LIB_DIR ?= /work1/hardware/users/nbrunie/tools/metalibm/framework/
#$(TEST_LIB_DIR)/../../


HAL_INCLUDE ?= /work1/hardware/users/nbrunie/tools_master/HAL/machine/core/common
K1A_CC=k1-gcc -I$(HAL_INCLUDE)
EXTRA_CC_OPTION ?=
CC_OPTION=-DTEST_START=$(TEST_START) -DTEST_END=$(TEST_END) -O3 -g -Wall -I $(TEST_LIB_DIR)/src/ -I $(SUPPORT_LIB_DIR)
K1B_CC_OPTION= -Wl,--defsym,__bsp_pwrctl_skip=1 -mcore=k1bdp -mboard=developer -mos=bare

BUILD_DEP+=$(TEST_LIB_DIR)/src/test_lib.c $(SUPPORT_LIB_DIR)/support_lib/ml_utils.c 
BUILD_DEP+=$(SUPPORT_LIB_DIR)/support_lib/ml_libm_compatibility.c 
BUILD_DEP+=$(SUPPORT_LIB_DIR)/support_lib/ml_multi_prec_lib.c 
#BUILD_DEP+=$(SUPPORT_LIB_DIR)/support_lib/k1/k1a_utils.c
#BUILD_DEP+=$(SUPPORT_LIB_DIR)/support_lib/k1/k1b_utils.c

MPFR_INCLUDE_DIR ?= /work1/hardware/users/nbrunie/local/include
MPFR_LIB ?= /work1/hardware/users/nbrunie/local/lib/libmpfr.a

BUILD_CC ?= $(K1A_CC)

gen_test:
	
	echo "function to test: $(TEST_FUNCTION)" 
	g++ -g $(LIMITED_TEST) -I$(MPFR_INCLUDE_DIR) -DRESULT_EXTRACT_FILE=$(RESULT_EXTRACT_FILE) -DINF_INTERVAL=$(INF_INTERVAL) -DTEST_FUNCTION=$(TEST_FUNCTION) -DSUP_INTERVAL=$(SUP_INTERVAL) -I$(RESULT_EXTRACT_DIR) -I $(TEST_LIB_DIR)/src/ -I $(SUPPORT_LIB_DIR) $(TEST_LIB_DIR)/src/random_gen.cpp $(TEST_LIB_DIR)/src/utils.cpp ./result_extract.c ./func_test_gen.cpp -lgmp -lm $(MPFR_LIB) -o exp_fp32_test_gen 
	./exp_fp32_test_gen $(SEED) $(NUM_TEST) > gen_test.c

build_test:
	$(BUILD_CC) $(CC_OPTION) $(KML_MODE) $(DEBUG_FLAG) $(PERF_FLAG) -I$(HAL_INCLUDE) -DTEST_FUNCTION=$(TEST_FUNCTION) -DNUM_TEST=$(NUM_TEST) gen_test.c $(TEST_TEMPLATE_FILE) $(BUILD_DEP) $(FUNCTION_FILE) -lm -o k1_exp_fp32_test $(EXTRA_CC_OPTION)
	#$(BUILD_CC) $(CC_OPTION) $(KML_MODE) $(DEBUG_FLAG) $(PERF_FLAG) -I$(HAL_INCLUDE) -DTEST_FUNCTION=$(TEST_FUNCTION) -DNUM_TEST=$(NUM_TEST) gen_test.c $(TEST_LIB_DIR)/src/test_precision_template.c $(TEST_LIB_DIR)/src/test_lib.c $(SUPPORT_LIB_DIR)/ml_utils.c $(SUPPORT_LIB_DIR)/ml_libm_compatibility.c $(FUNCTION_FILE) -lm -o k1_exp_fp32_test_precision $(EXTRA_CC_OPTION)

run_test:
	k1-cluster -- ./k1_exp_fp32_test
	#k1-cluster -- ./k1_exp_fp32_test_precision

run_profile_test:
	$(K1A_CC) $(CC_OPTION) $(DEBUG_FLAG) $(KML_MODE) -DPERF_TEST -I$(HAL_INCLUDE) -DTEST_FUNCTION=$(TEST_FUNCTION) -DNUM_TEST=$(NUM_TEST) gen_test.c $(TEST_LIB_DIR)/src/test_template.c $(TEST_LIB_DIR)/src/test_lib.c $(SUPPORT_LIB_DIR)/support_lib/ml_utils.c $(SUPPORT_LIB_DIR)/support_lib/ml_libm_compatibility.c $(FUNCTION_FILE) -lm -o k1_exp_fp32_test 
	k1-cluster --cycle-based --profile -- ./k1_exp_fp32_test; k1-disasm-all -o trace profile/ 

build_k1b: 
	$(K1B_CC) $(K1B_CC_OPTION) $(CC_OPTION) $(KML_MODE) gen_test.c src/test_lib.c ../framework/support_lib/ml_utils.c  ../framework/support_lib/ml_libm_compatibility.c $(FUNCTION_FILE) -lm -o k1_exp_fp32_test

run_k1b:
	$(K1B_RUNNER) --cycle-based --mboard=mppa64 --mcluster=node0 --mcore=k1bdp --profile -- k1_exp_fp32_test 

help:
	echo "option list:"
	echo "SEED=<int>           : RNG seed"
	echo "NUM_TEST=<int>       : number of test to be generated"
	echo "INF_INTERVAL=<float> : in directed fp rng, lower bound of the interval"
	echo "SUP_INTERVAL=<float> : in directed fp rng, upper bound of the interval"
	echo "TEST_FUNCTION=<string> : name of the function to be tested"
	echo "FUNCTION_FILE=<string> : source file containing <TEST_FUNCTION> implementation"
	echo "TEST_START=<int>       : start index in the test input vector"
	echo "TEST_END=<int>         : end index in the test input vector"
	echo "DEBUG_FLAG=[-DDEBUG_FLAG|] : enable/disable debug messages during test execution"
	echo "PERF_FLAG=[-DPERF_TEST|]   : enable/disable the performance test mode (no ev checking and single _IEEE_ mode)"
	echo "LIMITED_TEST=[-DLIMITED_TEST|]" : reduce the fp rng to [INF_INTERVAL, SUP_INTERVAL], specific values (NaN, 0, infty) are discared"
	echo "KML_MODE=[-DKML_MODE|]          : enable the generation of KML_MODE test (no libm compliant)"
