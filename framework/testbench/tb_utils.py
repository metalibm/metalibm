# -*- coding: utf-8 -*-

from core.ml_formats import ML_Binary32, ML_Binary64, ML_UInt32, ML_UInt64
from utility.common import zip_index

hex_value_map = {
    ML_Binary32: "hex_value32",
    ML_Binary64: "hex_value64",
}


bitfield_conversion_function_map = {
    ML_Binary32: "float_to_32b_encoding",
    ML_Binary64: "double_to_64b_encoding",
}

floatformat_conversion_function_map = {
    ML_Binary32: "float_from_32b_encoding",
    ML_Binary64: "double_from_64b_encoding",
}


mpfr_env_func_map = {
    ML_Binary32: "set_mpfr_binary32_env", 
    ML_Binary64: "set_mpfr_binary64_env",
}


mpfr_get_func_map = {
    ML_Binary32: "mpfr_get_flt", 
    ML_Binary64: "mpfr_get_d",
}

bitfield_conversion_format_map = {
    ML_Binary32: ML_UInt32,
    ML_Binary64: ML_UInt64,
}

nan_test_function_map = {
    ML_Binary32: "ml_is_nanf",
    ML_Binary64: "ml_is_nan",
}

snan_test_function_map = {
    ML_Binary32: "ml_is_signaling_nanf",
    ML_Binary64: "ml_is_signaling_nan",
}

underflow_exp_env_map = {
    ML_Binary32: -125,
    ML_Binary64: -1021,
}

libm_infinity_value_map = {
    ML_Binary32: "HUGE_VALF",
    ML_Binary64: "HUGE_VAL",
}


def gen_test_result_t(result_format):
    template = """typedef struct {
    %s value;
    %s faithful_value;
    int ev;
    int errno_f;
    int errtype;
    int matherr_call;\n} test_result_t;"""
    rt_str = bitfield_conversion_format_map[result_format]
    return template % (rt_str, rt_str) 


def gen_test_input_t(input_format_list):
    template = """typedef struct {
    %s
    tb_round_mode_t rnd_mode;\n} test_input_t;"""
    input_format_str = "\n    ".join("%s value_%s;" % (bitfield_conversion_format_map[input_format], index) for input_format, index in zip_index(input_format_list))
    return template % input_format_str


def gen_test_faithful_t(result_format):
    template = """typedef struct {
    test_input_t input;
    %s cr_value;
    test_result_t result_up, result_down;
} test_faithful_t;"""
    return template % bitfield_conversion_format_map[result_format]
