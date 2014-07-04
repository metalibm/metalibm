#! /bin/sh 
RESULT_EXTRACT_FILE=result_extract.c
TEST_TEMPLATE_FILE=test_template.c
NUM_TEST=10000
TOOLS_DIR=/work1/nbrunie/tools_multiarch_bare_build/devimage/toolchain_bare/k1tools/bin/
SPECIFIC_CC=gcc
SPECIFIC_RUNNER=

INPUT_FORMAT=fp32
OUTPUT_FORMAT=fp32
FUNCTION_NAME=new_exp
MPFR_FUNC_NAME=mpfr_exp
FUNCTION_FILE=new_exp.c
PRECISION=binary32

if [ -z $SEED         ]; then SEED=17; fi
if [ -z $RND_MODE     ]; then RND_MODE=rn; fi
if [ -z $ML_DIR       ]; then ML_DIR=/work1/nbrunie/tools_master/metalibm; fi
if [ -z $INF_INTERVAL ]; then INF_INTERVAL=0.5; fi
if [ -z $SUP_INTERVAL ]; then SUP_INTERVAL=1.5; fi
if [ -z $TEST_START   ]; then TEST_START=0; fi

 echo -e "\033[32;1m Generating test vector generator \033[0m" \
 &&  python $ML_DIR/framework/testbench/result_extract_function.py --informat $INPUT_FORMAT --outformat $OUTPUT_FORMAT --output $RESULT_EXTRACT_FILE \
 &&  python $ML_DIR/framework/testbench/gen_func_test_gen.py --round-mode $RND_MODE --informat $INPUT_FORMAT --outformat $OUTPUT_FORMAT --output func_test_gen.cpp --mpfr $MPFR_FUNC_NAME --result-extract ./result_extract.c  \
 &&  echo -e "\033[32;1m Building and executing test vector generator \033[0m" \
 &&  make -f $ML_DIR/framework/testbench/test_lib/Makefile gen_test MPFR_LIB=-lmpfr TEST_LIB_DIR=$ML_DIR/framework/testbench/test_lib SUPPORT_LIB_DIR=$ML_DIR/framework MPFR_INCLUDE_DIR=/work1/nbrunie/local_install/bin TEST_FUNCTION=$FUNCTION_NAME NUM_TEST=$NUM_TEST RESULT_EXTRACT_FILE=$RESULT_EXTRACT_FILE INF_INTERVAL=$INF_INTERVAL SUP_INTERVAL=$SUP_INTERVAL SEED=$SEED RESULT_EXTRACT_DIR=./ \
 &&  echo -e "\033[32;1m Generating testbench \033[0m" \
 &&  python $ML_DIR/framework/testbench/gen_test_template.py --test-mode faithful --informat $INPUT_FORMAT --outformat $OUTPUT_FORMAT --output $TEST_TEMPLATE_FILE --test $FUNCTION_NAME \
 &&  echo -e "\033[32;1m Generating function implementation \033[0m" \
 &&  python $ML_DIR/framework/ml_functions/ml_exp.py --target sse --disable-fma --output $FUNCTION_FILE $EXTRA_GEN_OPTIONS --fname $FUNCTION_NAME --precision $PRECISION \
 &&  echo -e "\033[32;1m Building function implementation within testbench \033[0m" \
 &&  make -f $ML_DIR/framework/testbench/test_lib/Makefile build_test   TEST_LIB_DIR=$ML_DIR/framework/testbench/test_lib SUPPORT_LIB_DIR=$ML_DIR/framework TEST_FUNCTION=$FUNCTION_NAME NUM_TEST=$NUM_TEST FUNCTION_FILE=$FUNCTION_FILE TEST_START=$TEST_START TEST_END=9999 TEST_TEMPLATE_FILE=$TEST_TEMPLATE_FILE BUILD_CC=$SPECIFIC_CC DEBUG_FLAG=-DML_DEBUG KML_MODE=-DKML_MODE EXTRA_CC_OPTION="-msse -msse4 -DDISABLE_EV" \
 &&  echo -e "\033[32;1m Running testbench \033[0m" \
 &&  echo "$SPECIFIC_RUNNER ./k1_exp_fp32_test" \
 &&  $SPECIFIC_RUNNER ./k1_exp_fp32_test


