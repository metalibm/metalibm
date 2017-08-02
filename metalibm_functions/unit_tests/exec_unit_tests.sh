python2 $ML_SRC_DIR/metalibm_functions/unit_tests/new_arg_template.py --target x86_avx2 &&\
python2 $ML_SRC_DIR/metalibm_functions/unit_tests/fixed_point.py --target fixed_point &&\
python2 $ML_SRC_DIR/metalibm_functions/unit_tests/function_emulate.py --target mpfr_backend &&\
python2 $ML_SRC_DIR/metalibm_functions/unit_tests/function_formats.py --target mpfr_backend &&\
python2 $ML_SRC_DIR/metalibm_functions/unit_tests/gappa_code.py &&\
python2 $ML_SRC_DIR/metalibm_functions/unit_tests/loop_operation.py &&\
python2 $ML_SRC_DIR/metalibm_functions/unit_tests/opencl_code.py &&\
python2 $ML_SRC_DIR/metalibm_functions/unit_tests/pointer_manipulation.py &&\
python2 $ML_SRC_DIR/metalibm_functions/unit_tests/static_vectorization.py --target vector &&\
python2 $ML_SRC_DIR/metalibm_functions/unit_tests/vector_code.py --target vector &&\
python2 $ML_SRC_DIR/metalibm_functions/unit_tests/call_externalization.py &&\
python2 $ML_SRC_DIR/metalibm_functions/unit_tests/auto_test.py --auto-test
# python2 $ML_SRC_DIR/metalibm_functions/unit_tests/payne_hanek.py --precision binary64 &&\
