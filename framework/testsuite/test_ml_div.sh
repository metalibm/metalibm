#! 
RESULT_EXTRACT_FILE=result_extract.c
TEST_TEMPLATE_FILE=test_template.c
NUM_TEST=10000

 echo -e "\033[32;1m Generating test vector generator \033[0m" \
 &&  python ../tools/metalibm/framework/testbench/result_extract_function.py --informat fp64,fp64 --outformat fp64 --output $RESULT_EXTRACT_FILE \
 &&  python ../tools/metalibm/framework/testbench/gen_func_test_gen.py --round-mode $RND_MODE --informat fp64,fp64 --outformat fp64 --output func_test_gen.cpp --mpfr mpfr_div --result-extract ./result_extract.c  \
 &&  echo -e "\033[32;1m Building and executing test vector generator \033[0m" \
 &&  make -f ../tools/metalibm/framework/testbench/test_lib/Makefile gen_test TEST_FUNCTION=div NUM_TEST=$NUM_TEST SEED=$SEED RESULT_EXTRACT_FILE=$RESULT_EXTRACT_FILE RESULT_EXTRACT_DIR=./ \
 &&  echo -e "\033[32;1m Generating testbench \033[0m" \
 &&  python ../tools/metalibm/framework/testbench/gen_test_template.py --informat fp64,fp64 --test-mode cr --outformat fp64 --output $TEST_TEMPLATE_FILE --test new_div \
 &&  echo -e "\033[32;1m Generating function implementation \033[0m" \
 &&  python ../tools/metalibm/framework/ml_functions/ml_div.py --target k1a --output new_div.c --num-iter 4 $EXTRA_GEN_ARGS --disable-fma --fname new_div --precision binary64 \
 &&  echo -e "\033[32;1m Building function implementation within testbench \033[0m" \
 &&  make -f ../tools/metalibm/framework/testbench/test_lib/Makefile build_test TEST_FUNCTION=new_div NUM_TEST=$NUM_TEST FUNCTION_FILE=new_div.c TEST_START=$TEST_START TEST_END=9999 TEST_TEMPLATE_FILE=$TEST_TEMPLATE_FILE BUILD_CC=k1-gcc DEBUG_FLAG=-DML_DEBUG KML_MODE=-DKML_MODE \
 &&  echo -e "\033[32;1m Running testbench \033[0m" \
 &&  k1-cluster -- ./k1_exp_fp32_test


