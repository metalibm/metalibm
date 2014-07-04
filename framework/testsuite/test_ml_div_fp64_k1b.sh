#! /bin/sh 
RESULT_EXTRACT_FILE=result_extract.c
TEST_TEMPLATE_FILE=test_template.c
NUM_TEST=10000
INPUT_FORMAT=fp64,fp64
OUTPUT_FORMAT=fp64
TOOLS_DIR=/work1/nbrunie/tools_multiarch_bare_build/devimage/toolchain_bare/k1tools/bin/
SPECIFIC_CC=$TOOLS_DIR/k1-gcc
SPECIFIC_RUNNER="$TOOLS_DIR/k1-cluster --mcore=k1bdp"
ML_DIR=/work1/nbrunie/tools_master/metalibm



 echo -e "\033[32;1m Generating test vector generator \033[0m" \
 &&  python $ML_DIR/framework/testbench/result_extract_function.py --informat $INPUT_FORMAT --outformat $OUTPUT_FORMAT --output $RESULT_EXTRACT_FILE \
 &&  python $ML_DIR/framework/testbench/gen_func_test_gen.py --round-mode $RND_MODE --informat $INPUT_FORMAT --outformat $OUTPUT_FORMAT --output func_test_gen.cpp --mpfr mpfr_div --result-extract ./result_extract.c  \
 &&  echo -e "\033[32;1m Building and executing test vector generator \033[0m" \
 &&  make -f $ML_DIR/framework/testbench/test_lib/Makefile gen_test MPFR_LIB=-lmpfr TEST_LIB_DIR=$ML_DIR/framework/testbench/test_lib SUPPORT_LIB_DIR=$ML_DIR/framework MPFR_INCLUDE_DIR=/work1/nbrunie/local_install/bin TEST_FUNCTION=new_sqrt NUM_TEST=$NUM_TEST RESULT_EXTRACT_FILE=$RESULT_EXTRACT_FILE INF_INTERVAL=-0x1.0p1023 SUP_INTERVAL=0x1.0p1023 SEED=$SEED RESULT_EXTRACT_DIR=./ \
 &&  echo -e "\033[32;1m Generating testbench \033[0m" \
 &&  python $ML_DIR/framework/testbench/gen_test_template.py --test-mode cr --informat $INPUT_FORMAT --outformat $OUTPUT_FORMAT --output $TEST_TEMPLATE_FILE --test new_div \
 &&  echo -e "\033[32;1m Generating function implementation \033[0m" \
 &&  echo `python $ML_DIR/framework/ml_functions/ml_div_clean.py --target k1b --num-iter 4 --output new_div.c $EXTRA_GEN_OPTIONS --fname new_div --precision binary64` \
 &&  python $ML_DIR/framework/ml_functions/ml_div_clean.py --target k1b --num-iter 4 --output new_div.c $EXTRA_GEN_OPTIONS --fname new_div --precision binary64 \
 &&  echo -e "\033[32;1m Building function implementation within testbench \033[0m" \
 &&  make -f $ML_DIR/framework/testbench/test_lib/Makefile build_test HAL_INCLUDE=/work1/nbrunie/tools_multiarch_bare_build/HAL/machine/common/core/  TEST_LIB_DIR=$ML_DIR/framework/testbench/test_lib SUPPORT_LIB_DIR=$ML_DIR/framework TEST_FUNCTION=new_div NUM_TEST=$NUM_TEST FUNCTION_FILE=new_div.c TEST_START=$TEST_START TEST_END=$(expr $NUM_TEST - 1) TEST_TEMPLATE_FILE=$TEST_TEMPLATE_FILE BUILD_CC=$SPECIFIC_CC DEBUG_FLAG=-DML_DEBUG KML_MODE=-DKML_MODE  EXTRA_CC_OPTION=-march=k1b\
 &&  echo -e "\033[32;1m Running testbench \033[0m" \
 &&  echo "$SPECIFIC_RUNNER -- ./k1_exp_fp32_testi" \
 &&  $SPECIFIC_RUNNER -- ./k1_exp_fp32_test


