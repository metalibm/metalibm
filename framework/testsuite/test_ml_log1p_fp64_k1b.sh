#! /bin/sh 
RESULT_EXTRACT_FILE=result_extract.c
TEST_TEMPLATE_FILE=test_template.c
NUM_TEST=10000
INPUT_FORMAT=fp64
OUTPUT_FORMAT=fp64
TOOLS_DIR=/work1/nbrunie/tools_multiarch_bare_build/devimage/toolchain_bare/k1tools/bin/
SPECIFIC_CC=$TOOLS_DIR/k1-gcc
SPECIFIC_RUNNER="$TOOLS_DIR/k1-cluster --mcore=k1bdp"
ML_DIR=/work1/nbrunie/tools_master/metalibm

FUNCTION_NAME=new_log1p
MPFR_FUNC_NAME=mpfr_log1p
FUNCTION_FILE=new_log1p.c

if [ -z $INF_INTERVAL ]; then INF_INTERVAL=0.5; fi
if [ -z $SUP_INTERVAL ]; then SUP_INTERVAL=1.5; fi

 echo -e "\033[32;1m Generating test vector generator \033[0m" \
 &&  python $ML_DIR/framework/testbench/result_extract_function.py --informat $INPUT_FORMAT --outformat $OUTPUT_FORMAT --output $RESULT_EXTRACT_FILE \
 &&  python $ML_DIR/framework/testbench/gen_func_test_gen.py --round-mode $RND_MODE --informat $INPUT_FORMAT --outformat $OUTPUT_FORMAT --output func_test_gen.cpp --mpfr $MPFR_FUNC_NAME --result-extract ./result_extract.c  \
 &&  echo -e "\033[32;1m Building and executing test vector generator \033[0m" \
 &&  make -f $ML_DIR/framework/testbench/test_lib/Makefile gen_test MPFR_LIB=-lmpfr TEST_LIB_DIR=$ML_DIR/framework/testbench/test_lib SUPPORT_LIB_DIR=$ML_DIR/framework MPFR_INCLUDE_DIR=/work1/nbrunie/local_install/bin TEST_FUNCTION=$FUNCTION_NAME NUM_TEST=$NUM_TEST RESULT_EXTRACT_FILE=$RESULT_EXTRACT_FILE INF_INTERVAL=$INF_INTERVAL SUP_INTERVAL=$SUP_INTERVAL SEED=$SEED RESULT_EXTRACT_DIR=./ \
 &&  echo -e "\033[32;1m Generating testbench \033[0m" \
 &&  python $ML_DIR/framework/testbench/gen_test_template.py --test-mode faithful --informat $INPUT_FORMAT --outformat $OUTPUT_FORMAT --output $TEST_TEMPLATE_FILE --test $FUNCTION_NAME \
 &&  echo -e "\033[32;1m Generating function implementation \033[0m" \
 &&  python $ML_DIR/framework/ml_functions/ml_log1p.py --target k1b --output $FUNCTION_FILE $EXTRA_GEN_OPTIONS --fname $FUNCTION_NAME --precision binary64 \
 &&  echo -e "\033[32;1m Building function implementation within testbench \033[0m" \
 &&  make -f $ML_DIR/framework/testbench/test_lib/Makefile build_test HAL_INCLUDE=/work1/nbrunie/tools_multiarch_bare_build/HAL/machine/common/core/  TEST_LIB_DIR=$ML_DIR/framework/testbench/test_lib SUPPORT_LIB_DIR=$ML_DIR/framework TEST_FUNCTION=$FUNCTION_NAME NUM_TEST=$NUM_TEST FUNCTION_FILE="$FUNCTION_FILE new_log.c" TEST_START=$TEST_START TEST_END=9999 TEST_TEMPLATE_FILE=$TEST_TEMPLATE_FILE BUILD_CC=$SPECIFIC_CC DEBUG_FLAG=-DML_DEBUG KML_MODE=-DKML_MODE  EXTRA_CC_OPTION="-march=k1b -DDISABLE_EV" \
 &&  echo -e "\033[32;1m Running testbench \033[0m" \
 &&  echo "$SPECIFIC_RUNNER -- ./k1_exp_fp32_testi" \
 &&  $SPECIFIC_RUNNER -- ./k1_exp_fp32_test


