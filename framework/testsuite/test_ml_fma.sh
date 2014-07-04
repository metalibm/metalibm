#! 
RESULT_EXTRACT_FILE=result_extract.c
TEST_TEMPLATE_FILE=test_template.c
METALIBM_ROOT_DIR=/work1/hardware/users/nbrunie/tools/metalibm/framework/support_lib/

 echo -e "\033[32;1m Generating test vector generator \033[0m" \
 &&  python ../tools/metalibm/framework/testbench/result_extract_function.py --informat fp64,fp64,fp64 --outformat fp64 --output $RESULT_EXTRACT_FILE \
 &&  python ../tools/metalibm/framework/testbench/gen_func_test_gen.py --informat fp64,fp64,fp64 --outformat fp64 --output func_test_gen.cpp --mpfr mpfr_fma --result-extract ./result_extract.c  \
 &&  echo -e "\033[32;1m Building and executing test vector generator \033[0m" \
 &&  make -f ../tools/metalibm/framework/testbench/test_lib/Makefile gen_test TEST_FUNCTION=ml_fma NUM_TEST=1000 RESULT_EXTRACT_FILE=$RESULT_EXTRACT_FILE RESULT_EXTRACT_DIR=./ \
 &&  echo -e "\033[32;1m Generating testbench \033[0m" \
 &&  python ../tools/metalibm/framework/testbench/gen_test_template.py --informat fp64,fp64,fp64 --outformat fp64 --output $TEST_TEMPLATE_FILE --test ml_fma \
   echo -e "\033[32;1m Building function implementation within testbench \033[0m" \
 &&  make -f ../tools/metalibm/framework/testbench/test_lib/Makefile build_test TEST_FUNCTION=ml_fma NUM_TEST=1000 FUNCTION_FILE=$METALIBM_ROOT_DIR/ml_multi_prec_lib.c TEST_START=0 TEST_END=999 TEST_TEMPLATE_FILE=$TEST_TEMPLATE_FILE BUILD_CC=k1-gcc DEBUG_FLAG=-DML_DEBUG KML_MODE=-DKML_MODE EXTRA_CC_OPTION=-DDISABLE_EV \
 &&  echo -e "\033[32;1m Running testbench \033[0m" \
 &&  k1-cluster -- ./k1_exp_fp32_test


