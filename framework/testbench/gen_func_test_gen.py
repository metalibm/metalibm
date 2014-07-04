# -*- coding: utf-8 -*-


import sys

from utility.common import extract_option_value, test_flag_option, zip_index
from core.ml_formats import ML_Binary32, ML_Binary64, ML_RoundToNearest, ML_RoundTowardZero, ML_RoundTowardPlusInfty, ML_RoundTowardMinusInfty
from testbench.tb_utils import * #gen_test_faithful_t, gen_test_input_t, gen_test_result_t


def gen_header_list(input_format_list, result_format):
    code = """#include <random_gen.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <mpfr.h>
#include <math.h>
#include <test_lib.h>
#include <support_lib/ml_utils.h>
#include <support_lib/ml_types.h>
    """
    code += gen_test_result_t(result_format) + "\n\n"
    code += gen_test_input_t(input_format_list) + "\n\n"
    code += gen_test_faithful_t(result_format) + "\n\n"
    code += """test_result_t extract_result_status(test_input_t input, mpfr_rnd_t round_mode, %s result, mpfr_t mp_result, int ternary);\n """ % result_format.get_c_name() 

    return code



def gen_get_test_result_str(result_format):
    hv_f = hex_value_map[result_format]

    code = """std::string get_test_result_str(test_result_t result) {
    std::stringstream s;
    s << std::hex << "{.value = " << %s(result.value) << ", .ev = " <<  hex_value32(result.ev);
    s << std::hex << ", .faithful_value = " << %s(result.faithful_value);
    s << ", .errno_f = " << hex_value32(result.errno_f);
    s << ", .errtype = " << hex_value32(result.errtype);
    s << ", .matherr_call = " << hex_value32(result.matherr_call) << "}";
    return s.str();\n}\n\n"""
    return code % (hv_f, hv_f)


def gen_get_test_input_str(input_format_list):
    template = """std::string get_test_input_str(test_input_t input) {
    std::stringstream s;
    s << std::hex << "{"""
    template += ", ".join(""".value_%d = " << %s(input.value_%d) << " """ % (index, hex_value_map[iformat], index) for iformat, index in zip_index(input_format_list))
    template += """, .rnd_mode = " <<  hex_value32(input.rnd_mode) << "}";
    return s.str();\n}\n\n"""
    return template



def gen_test_main(input_format_list, result_format, mpfr_function, round_mode):
    mpfr_input_str = ", ".join("func_input_%d" % index for iformat, index in zip_index(input_format_list))

    code = """int main(int argc, char** argv) {
    // init and seeding floating-point RNG
    ML_FloatingPointRNG* fprng = new ML_FloatingPointRNG();
    fprng->seed(atoi(argv[1]));

    mpfr_t %s, func_result;""" % mpfr_input_str


    code += "    // mpfr_t variable initialization\n"
    for iformat, index in zip_index(input_format_list):
        code += "    mpfr_init2(func_input_%d, %d);\n" % (index, iformat.get_field_size()+1)
    code += "    mpfr_init2(func_result, %d);\n" % (result_format.get_field_size()+1)

    mpfr_env_set =  "\n        // mpfr environement setting\n"
    mpfr_env_set += "        %s();\n" % (mpfr_env_func_map[result_format])


    code += mpfr_env_set


    code += """

    int n = atoi(argv[2]);
    std::cout << "#include <stdlib.h>" << std::endl;
    std::cout << "#include <stdio.h>" << std::endl;
    std::cout << "#include <math.h>" << std::endl;
    std::cout << "#include <inttypes.h>" << std::endl;
    std::cout << "#include <test_lib.h>" << std::endl;
    std::cout << "#include <support_lib/ml_utils.h>" << std::endl;

    std::cout << std::endl << std::endl;
    """
    code += """
    std::cout << "%s;" << std::endl << std::endl;
    std::cout << "%s;" << std::endl << std::endl;
    std::cout << "%s;" << std::endl << std::endl;
    """ % (gen_test_result_t(result_format).replace("\n", "\\n"), gen_test_input_t(input_format_list).replace("\n", "\\n"), gen_test_faithful_t(result_format).replace("\n", "\\n"))


    code += """
    std::cout << "test_faithful_t test_array[" << n << "] = {" << std::endl;


    int test_index;"""
    for iformat, index in zip_index(input_format_list):
        code += "   static int test_%d_index = fprng->generate_index();\n" % index

    code += """
    for (int i = 0; i < n; i++) {

        /** input value */
        """
    code += mpfr_env_set


    code +=    """        #ifdef LIMITED_TEST\n"""
    for iformat, index in zip_index(input_format_list):
        code += """        %s test_value_%d_f = fprng->generate_random_%s_interval(INF_INTERVAL, SUP_INTERVAL);\n""" % (iformat.get_c_name(), index, iformat.get_c_name())
    code += """        #else\n"""
    for iformat, index in zip_index(input_format_list):
        code += """        %s test_value_%d_f = fprng->generate_random_%s_interval_focus(INF_INTERVAL, SUP_INTERVAL, test_%d_index);\n""" % (iformat.get_c_name(), index, iformat.get_c_name(), index)
    code += """        #endif\n"""

    mpfr_conv_func_map = {
        ML_Binary32: "mpfr_set_flt",
        ML_Binary64: "mpfr_set_d", 
    }

    code += "        test_input_t input_struct;\n"

    mpfr_round_map = {
        ML_RoundToNearest: "MPFR_RNDN",
        ML_RoundTowardZero: "MPFR_RNDZ",
        ML_RoundTowardPlusInfty: "MPFR_RNDU", 
        ML_RoundTowardMinusInfty: "MPFR_RNDD"
    }

    tb_round_mode_map = {
        ML_RoundToNearest: "tb_round_nearest",
        ML_RoundTowardZero: "tb_round_toward_zero",
        ML_RoundTowardPlusInfty: "tb_round_up", 
        ML_RoundTowardMinusInfty: "tb_round_down"
    }

    for iformat, index in zip_index(input_format_list):
        code += "        %s(func_input_%d, test_value_%d_f, MPFR_RNDN);\n" % (mpfr_conv_func_map[iformat], index, index)
        code += "        input_struct.value_%d = %s(test_value_%d_f);\n" % (index, bitfield_conversion_function_map[iformat], index) 

    code += """
        input_struct.rnd_mode = %s; 
    """ % tb_round_mode_map[round_mode]
    code += """
        /** result rounded downward */
        clear_mpfr_exception();
        int ternary = %s(func_result, %s, MPFR_RNDD);""" % (mpfr_function, mpfr_input_str)

    code += """
        ternary = mpfr_subnormalize(func_result, ternary, MPFR_RNDD);
        %s result_rd = %s(func_result, MPFR_RNDN);""" % (result_format.get_c_name(), mpfr_get_func_map[result_format])

    code += """
        /** downward exceptions */
        test_result_t down_result = extract_result_status(input_struct, MPFR_RNDD, result_rd, func_result, ternary);
    """

    code += """
        /** result rounded upward */
        clear_mpfr_exception();
        """
    code += mpfr_env_set
    code += """        ternary = %s(func_result, %s, MPFR_RNDU);""" % (mpfr_function, mpfr_input_str)

    code += """
        ternary = mpfr_subnormalize(func_result, ternary, MPFR_RNDU);
        %s result_ru = %s(func_result, MPFR_RNDN); 
        /** upward exceptions */
        """ % (result_format.get_c_name(), mpfr_get_func_map[result_format])

    code += """
        test_result_t up_result = extract_result_status(input_struct, MPFR_RNDU, result_ru, func_result, ternary);

        /** correctly rounded result */
        clear_mpfr_exception();
        """
    code += mpfr_env_set


    code += """        ternary = %s(func_result, %s, %s);""" % (mpfr_function, mpfr_input_str, mpfr_round_map[round_mode])

    code += """
        ternary = mpfr_subnormalize(func_result, ternary, %s);
        %s result_rn = %s(func_result, MPFR_RNDN);
        """ % (mpfr_round_map[round_mode], result_format.get_c_name(), mpfr_get_func_map[result_format])

    cr_value_str = """%s(%s(result_rn))""" % (hex_value_map[result_format], bitfield_conversion_function_map[result_format])

    code += """

        std::cout << "{ /** test line: " << std::dec << i << "*/" << std::endl 
            << "          .input = " << get_test_input_str(input_struct) << "," << std::endl 
            << "          .result_down = " << get_test_result_str(down_result) << "," << std::endl
            << "          .result_up = " << get_test_result_str(up_result) << "," << std::endl 
            << "          .cr_value = " << %s << std::endl 
            << "        }";
    \n""" % cr_value_str
    code += """         if (i == n-1) std::cout << std::endl;
        else std::cout << "," << std::endl;
    }
    """

    code += """

    std::cout << "};" << std::endl << std::endl << std::endl; 
    
    return 0;\n}
    """
    return code


if __name__ == "__main__":
    str_format_map = {
        "fp32": ML_Binary32,
        "fp64": ML_Binary64,
    }
    rnd_map = {
        "rn": ML_RoundToNearest,
        "rz": ML_RoundTowardZero,
        "ru": ML_RoundTowardPlusInfty,
        "rd": ML_RoundTowardMinusInfty
    }
    input_format_list = [str_format_map[i_format] for i_format in extract_option_value("--informat", "fp32").split(",")]
    result_format = str_format_map[extract_option_value("--outformat", "fp32")]
    mpfr_function = extract_option_value("--mpfr", "undefined")
    output_file = extract_option_value("--output", "test_generator.c")
    result_extract_info = extract_option_value("--result-extract", "./extract_result.c")
    round_mode = rnd_map[extract_option_value("--round-mode", "rn")]

    code = gen_header_list(input_format_list, result_format)
    code += "\n\n"
    code += gen_get_test_result_str(result_format)
    code += gen_get_test_input_str(input_format_list)
    code += gen_test_main(input_format_list, result_format, mpfr_function, round_mode)

    output_stream = open(output_file, "w")
    output_stream.write(code)
    output_stream.close()
