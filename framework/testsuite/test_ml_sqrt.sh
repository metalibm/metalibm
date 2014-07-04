#! /bin/sh 
RESULT_EXTRACT_FILE=result_extract.c
TEST_TEMPLATE_FILE=test_template.c
NUM_TEST=10000
INPUT_FORMAT=fp64
OUTPUT_FORMAT=fp64


 echo -e "\033[32;1m Generating test vector generator \033[0m" \
 &&  python /work1/hardware/users/nbrunie//tools/metalibm/framework/testbench/result_extract_function.py --informat $INPUT_FORMAT --outformat $OUTPUT_FORMAT --output $RESULT_EXTRACT_FILE \
 &&  python /work1/hardware/users/nbrunie//tools/metalibm/framework/testbench/gen_func_test_gen.py --round-mode $RND_MODE --informat $INPUT_FORMAT --outformat $OUTPUT_FORMAT --output func_test_gen.cpp --mpfr mpfr_sqrt --result-extract ./result_extract.c  \
 &&  echo -e "\033[32;1m Building and executing test vector generator \033[0m" \
 &&  make -f /work1/hardware/users/nbrunie//tools/metalibm/framework/testbench/test_lib/Makefile gen_test TEST_FUNCTION=new_sqrt NUM_TEST=$NUM_TEST RESULT_EXTRACT_FILE=$RESULT_EXTRACT_FILE INF_INTERVAL=-1.0f SUP_INTERVAL=0x1.0p1023 SEED=$SEED RESULT_EXTRACT_DIR=./ \
 &&  echo -e "\033[32;1m Generating testbench \033[0m" \
 &&  python /work1/hardware/users/nbrunie//tools/metalibm/framework/testbench/gen_test_template.py --test-mode cr --informat $INPUT_FORMAT --outformat $OUTPUT_FORMAT --output $TEST_TEMPLATE_FILE --test new_sqrt \
 &&  echo -e "\033[32;1m Generating function implementation \033[0m" \
 &&  python /work1/hardware/users/nbrunie//tools/metalibm/framework/ml_functions/ml_sqrt.py --target k1a --disable-fma --output new_sqrt.c --fname new_sqrt --precision binary64 \
 &&  echo -e "\033[32;1m Building function implementation within testbench \033[0m" \
 &&  make -f /work1/hardware/users/nbrunie//tools/metalibm/framework/testbench/test_lib/Makefile build_test TEST_FUNCTION=new_sqrt NUM_TEST=$NUM_TEST FUNCTION_FILE=new_sqrt.c TEST_START=0 TEST_END=9999 TEST_TEMPLATE_FILE=$TEST_TEMPLATE_FILE BUILD_CC=k1-gcc DEBUG_FLAG=-DML_DEBUG KML_MODE=-DKML_MODE  \
 &&  echo -e "\033[32;1m Running testbench \033[0m" \
 &&  k1-cluster -- ./k1_exp_fp32_test


