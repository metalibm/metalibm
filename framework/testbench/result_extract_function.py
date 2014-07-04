# -*- coding: utf-8 -*-


import sys

from core.ml_formats import *
from utility.common import zip_index, extract_option_value, test_flag_option
from testbench.tb_utils import * #gen_test_faithful_t, gen_test_input_t, gen_test_result_t


def gen_extract_result_status(input_format_list, result_format):
    code = ""
    code += """test_result_t extract_result_status(test_input_t input, mpfr_rnd_t round_mode, %s result, mpfr_t mp_result, int ternary) {
    """ % result_format.get_c_name() 

    # result bitfield conversion function
    rbf_conv_f = bitfield_conversion_function_map[result_format]
    # result floating-point conversion function


    code += """
    /* struct result declaration */
    test_result_t result_struct;
    result_struct.value = %s(result);
    result_struct.faithful_value = %s(result);
    /* default value */""" % (rbf_conv_f, rbf_conv_f)

    code += """
    result_struct.errno_f = -1;
    result_struct.errtype = -1;

    """
    test_snan = " || ".join("%s(%s(input.value_%d))" % (snan_test_function_map[input_format], floatformat_conversion_function_map[input_format], index) for input_format, index in zip_index(input_format_list))

    test_nan = " || ".join("%s(%s(input.value_%d))" % (nan_test_function_map[input_format], floatformat_conversion_function_map[input_format], index) for input_format, index in zip_index(input_format_list))


    code += """
    int invalid_input = %s;
    int nan_input = %s;""" % (test_snan, test_nan)

    code += """
    int ex_divbyzero = mpfr_divby0_p() ? 1 : 0;
    """

    code += """
    int ex_inexact = mpfr_inexflag_p() ? 1 : 0;
    int ex_invalid = (invalid_input || (!nan_input && mpfr_nan_p(mp_result))) ? 1 : 0;
    // underflow exception
    """
    code += """
    mpfr_set_emin(%s);""" % underflow_exp_env_map[result_format]

    code += """
    mpfr_check_range(mp_result, ternary, round_mode);
    int ex_underflow = ((mpfr_underflow_p() && ex_inexact) || is_hidden_underflowf(result, ternary)) ? 1 : 0;
    // overflow exception
    int ex_overflow = mpfr_overflow_p() ? 1 : 0;
    // errno
    if (ex_underflow) {
        result_struct.errno_f = ERANGE;
        result_struct.errtype = UNDERFLOW;
        //result_struct.value = 0.0f;
    } else if (ex_overflow) {
        result_struct.errno_f = ERANGE;
        result_struct.errtype = OVERFLOW;
    """
    code += """    if (result > 0) result_struct.value = %s(%s);\n""" % (rbf_conv_f, libm_infinity_value_map[result_format]) 
    code += """    if (result < 0) result_struct.value = %s(-%s);\n""" % (rbf_conv_f, libm_infinity_value_map[result_format])
    code += """
    } else if (ex_invalid && !nan_input) {
        result_struct.errno_f = EDOM;
        result_struct.errtype = DOMAIN;
    }

    /** exception vector */
    int ev = (ex_inexact << 4) | (ex_underflow << 3) | (ex_overflow << 2) | (ex_divbyzero << 1) | (ex_invalid);
    result_struct.ev = ev;
    result_struct.matherr_call = (result_struct.errno_f > 0) || (result_struct.errtype > 0); 
    return result_struct;
}"""

    return code


if __name__ == "__main__":
    str_format_map = {
        "fp32": ML_Binary32,
        "fp64": ML_Binary64,
    }
    input_format_list = [str_format_map[i_format] for i_format in extract_option_value("--informat", "fp32").split(",")]
    result_format = str_format_map[extract_option_value("--outformat", "fp32")]

    output_file = extract_option_value("--output", "extract_result.c")
    code =  "#include <stdint.h>\n"
    code += "#include <math.h>\n"
    code += "#include <errno.h>\n"
    code += "#include <mpfr.h>\n"
    code += "#include <support_lib/ml_utils.h>\n"
    code += "#include <utils.hpp>\n"
    code += gen_test_result_t(result_format) + "\n\n"
    code += gen_test_input_t(input_format_list) + "\n\n"
    code += gen_test_faithful_t(result_format) + "\n\n"
    code += gen_extract_result_status(input_format_list, result_format) +  "\n\n"
    out_stream = open(output_file, "w")
    out_stream.write(code)
    out_stream.close()



    


