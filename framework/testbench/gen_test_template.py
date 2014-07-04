# -*- coding: utf-8 -*-


import sys

from utility.common import extract_option_value, test_flag_option, zip_index
from core.ml_formats import ML_Binary32, ML_Binary64
from testbench.tb_utils import *

class TEST_MODE_FAITHFUL: pass
class TEST_MODE_CR: pass


def gen_template_test(input_format_list, result_format, test_function, test_mode = TEST_MODE_FAITHFUL):
    code_template = """#ifndef RTL_RUN
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <errno.h>
#include <inttypes.h>
#else
#include <stdint.h>
#include <HAL/hal/hal.h>
#endif

#include <test_lib.h>
#include <support_lib/ml_utils.h>

/* Test INDEX SELECTION */
#ifndef TEST_START
#define TEST_START 0 
#endif /* TEST_START */
#ifndef TEST_END
#define TEST_END 0
#endif /* TEST_END */
    """


    code_template += """\n#ifdef __K1__\n#include <cpu.h>\n#endif\n"""

    code_template += gen_test_result_t(result_format) + "\n\n"
    code_template += gen_test_input_t(input_format_list) + "\n\n"
    code_template += gen_test_faithful_t(result_format) + "\n\n"

    code_template += """
typedef struct {
    int lib_mode_f;
    int errno_check;
    int matherr_check;
    int faithful_check;
} test_mode_t; 

#define _KML_ 17

#ifndef RTL_RUN
test_mode_t lib_mode[5] = {
    {.lib_mode_f = _IEEE_, .matherr_check = 0, .errno_check = 0, .faithful_check = 0}, 
    {.lib_mode_f = _POSIX_, .matherr_check = 0, .errno_check = 1, .faithful_check = 0}, 
    {.lib_mode_f = _XOPEN_, .matherr_check = 1, .errno_check = 1, .faithful_check = 0},
    {.lib_mode_f = _SVID_, .matherr_check = 1, .errno_check = 1, .faithful_check = 0},
    {.lib_mode_f = _KML_, .matherr_check = 0, .errno_check = 0, .faithful_check = 0}
};
#endif /* RTL_RUN */


extern test_faithful_t test_array[NUM_TEST];
%s result_array[NUM_TEST]; 
uint32_t exception_array[NUM_TEST]; 


    """ % result_format.get_c_name()
    code_template += """
// function to test
    %s %s(%s);\n
    """ % (result_format.get_c_name(), test_function, ", ".join("%s" % iformat for iformat in input_format_list))

    bitfield_conversion_function_map = {
        ML_Binary32: "float_to_32b_encoding",
        ML_Binary64: "double_to_64b_encoding",
    }

    fp_conversion_function_map = {
        ML_Binary32: "float_from_32b_encoding",
        ML_Binary64: "double_from_64b_encoding",
    }

    code_template += """ 
int matherr_call_expected = 0;
int matherr_type_expected = -1;
int matherr_err_expected = -1;
char* matherr_name_expected = "%s";

int main(void) {
    int i, test_mode_id;
    #ifndef RTL_RUN
    _LIB_VERSION = _IEEE_;
    #endif /* RTL_RUN */

    #ifdef PERF_TEST
    for (i = TEST_START; i <= TEST_END; i++) {\n""" % test_function
    # set round mode
    code_template += """        set_rounding_mode(test_array[i].input.rnd_mode);\n"""

    for iformat, index in zip_index(input_format_list):
        code_template += """        %s input_%d = %s(test_array[i].input.value_%d);\n""" % (iformat.get_c_name(), index, fp_conversion_function_map[iformat], index)

    input_str = ", ".join("input_%d" % index for iformat, index in zip_index(input_format_list)) 

    code_template += """
        result_array[i] = %s(%s);
    };
    #endif /** PERF_TEST */
    """ % (test_function, input_str)

    code_template += """
    #ifndef RTL_RUN
    printf("\t_IEEE_: %d\\n\t_POSIX: %d\\n\t_XOPEN_: %d\\n\t_SVID_: %d\\n", _IEEE_, _POSIX_, _XOPEN_, _SVID_);
    for (test_mode_id = 0; test_mode_id < 4; test_mode_id++) { 
    #else
    {
    #endif
        #ifndef RTL_RUN
        test_mode_t test_mode = lib_mode[test_mode_id];
        printf("TEST_MODE: %d\\n", test_mode.lib_mode_f);
        #endif /* RTL_RUN */

        for (i = TEST_START; i <= TEST_END; i++) {\n"""
    for iformat, index in zip_index(input_format_list):
        code_template += """            %s input_%d = %s(test_array[i].input.value_%d);\n""" % (iformat.get_c_name(), index, fp_conversion_function_map[iformat], index)

    code_template += """
            #ifdef PERF_TEST
            %s result = result_array[i];
            #else /* !PERF_TEST */
            #ifndef RTL_RUN
            errno = -1;
            matherr_call_expected = 0;
            _LIB_VERSION = test_mode.lib_mode_f; // setting LIB MODE
            #endif /* RTL_RUN */
    """ % result_format.get_c_name()

    code_template += """        set_rounding_mode(test_array[i].input.rnd_mode);\n"""

    code_template += """
            #ifdef __K1__
            __k1_fpu_clear_exceptions(_K1_FPU_ALL_EXCEPTS);
            #endif /* __K1__ */
            %s result = %s(%s);
            #ifdef __K1__
            int fpu_ev = __k1_fpu_get_exceptions() >> 1; 
            #else
            int fpu_ev = 0; 
            #endif /* __K1__ */

            #endif /*PERF_TEST */

            """ % (result_format.get_c_name(), test_function, input_str)

    # result bitfield conversion function
    rcf = bitfield_conversion_function_map[result_format] 
    result_fp_conv = fp_conversion_function_map[result_format]

    faithful_test_map = {
        ML_Binary32: "fp32_is_faithful",
        ML_Binary64: "fp64_is_faithful",
    }

    cr_test_map = {
        ML_Binary32: "fp32_is_cr",
        ML_Binary64: "fp64_is_cr",
    }

    code_template += """

            #ifdef KML_MODE
            %s expected_rd = %s(test_array[i].result_down.faithful_value);
            %s expected_ru = %s(test_array[i].result_up.faithful_value);
            %s expected_cr = %s(test_array[i].cr_value);
    """ % (result_format.get_c_name(), result_fp_conv, result_format.get_c_name(), result_fp_conv, result_format.get_c_name(), result_fp_conv)

    code_template += """
            #else /* ! KML_MODE */
            %s expected_rd = %s(test_array[i].result_down.value);
            %s expected_ru = %s(test_array[i].result_up.value);
            %s expected_cr = %s(test_array[i].cr_value);
            #endif
    """ % (result_format.get_c_name(), result_fp_conv, result_format.get_c_name(), result_fp_conv, result_format.get_c_name(), result_fp_conv)
    if test_mode is TEST_MODE_FAITHFUL:
        code_template += """

                if (!%s(expected_rd, expected_ru, result)) {
        """ % faithful_test_map[result_format]
    elif test_mode is TEST_MODE_CR:
        code_template += """

                if (!%s(expected_cr, result)) {
        """ % cr_test_map[result_format]
    else:
        print "ERROR: undefined test mode ", test_mode
        raise Exception()

    # bitfield printf format dictionnary
    printf_bf_format_map = {
        ML_Binary32: "%\"PRIx32\"",
        ML_Binary64: "%\"PRIx64\"",
    }
    printf_format_map = {
        ML_Binary32: "%f",
        ML_Binary64: "%lf",
    }
    # result printf format
    rpf = printf_format_map[result_format]
    # result bitfield prinft format
    rpf_bf = printf_bf_format_map[result_format]

    input_error_template = ",".join("%s/%s"  for iformat in input_format_list)
    input_error_list = []
    for iformat, index in zip_index(input_format_list):
        input_error_list.append((printf_format_map[iformat], "input_%d" % index))
        input_error_list.append((printf_bf_format_map[iformat], "%s(input_%d)" % (bitfield_conversion_function_map[iformat], index)))


    result_bf = "%s(result)" % rcf
    exp_rd_bf = "%s(expected_rd)" % rcf
    exp_ru_bf = "%s(expected_ru)" % rcf
    exp_cr_bf = "%s(expected_cr)" % rcf

    def generate_printf(template, value_list):
        inst_template = template % tuple([v[0] for v in value_list])
        printf_input_list = ", ".join(v[1] for v in value_list)
        code = """printf(\"%s\", %s)""" % (inst_template, printf_input_list)
        return code

    if test_mode is TEST_MODE_FAITHFUL:
        code_template += "#ifndef RTL_RUN\n            %s;\n#endif /*RTL_RUN*/\n" % generate_printf("""result error for test %%s\\n for %s(%s), expected %%s or %%s got %%s\\n""" % (test_function, input_error_template), [("%d", "i")] + input_error_list + [(rpf_bf, exp_rd_bf), (rpf_bf, exp_ru_bf), (rpf_bf, result_bf)])
    elif test_mode is TEST_MODE_CR:
        code_template += "#ifndef RTL_RUN\n            %s;\n#endif /* RTL_RUN */\n" % generate_printf("""result error for test %%s\\n for %s(%s), expected %%s got %%s\\n""" % (test_function, input_error_template), [("%d", "i")] + input_error_list + [(rpf_bf, exp_cr_bf), (rpf_bf, result_bf)])
    else:
        pass

    nan_test_function_map = {
        ML_Binary32: "ml_is_nanf",
        ML_Binary64: "ml_is_nan",
    }

    code_template += """
                return 1;
            } else {
                #ifndef PERF_TEST
                test_result_t comp_result;

                /** selecting comparison result */
                if (expected_rd == result) comp_result = test_array[i].result_down;
                else if (expected_ru == result) comp_result = test_array[i].result_up;
                else {
                    /** NaN cases */
                    if (%s(expected_rd)) comp_result = test_array[i].result_down;
    """ % (nan_test_function_map[result_format])


    code_template += """
                    else {
                    """
    code_template += "#ifndef RTL_RUN\n            %s;\n#endif /* RTL_RUN */ \n" % generate_printf("""result error for test %%s\\n result do not match ru or rd: for %s(%s), expected %%s or %%s for %%s\\n""" % (test_function, input_error_template), [("%d", "i")] + input_error_list + [(rpf_bf, exp_rd_bf), (rpf_bf, exp_ru_bf), (rpf_bf, result_bf)])
    code_template += """
                        return 1;
                    }
                }

                #ifndef DISABLE_EV
                if (fpu_ev != comp_result.ev) { 
                    /** exception comparison */
    """
    code_template += "#ifndef RTL_RUN\n         %s;\n#endif /* RTL_RUN */\n" % generate_printf(""" exception error for test %%s\\n for %s(%s), expected ev = 0x%%s, got 0x%%s\\n""" % (test_function, input_error_template), [("%d", "i")] + input_error_list + [("%x", "comp_result.ev"), ("%x", "fpu_ev")])

    code_template += """ 
                    return 1;
                }; 
                #endif /* DISABLE_EV */
                #ifndef RTL_RUN
                #ifndef KML_MODE
                /** testing errno */
                if (test_mode.errno_check) { 
                    if (comp_result.errno_f != errno) {
                        /** exception comparison */
    """
    code_template += "         %s;\n" % generate_printf(""" errno error for test %%s\\n for %s(%s), expected errno=0x%%s, got 0x%%s\\n""" % (test_function, input_error_template), [("%d", "i")] + input_error_list + [("%d", "comp_result.errno_f"), ("%d", "errno")])

    code_template += """
                        return 1;
                    }
                } else if (errno != -1) {
                    printf("errno error for test %d \\n", i);
                    int errno_cp = errno;
    """
    code_template += "         %s;\n" % generate_printf(""" for %s(%s), unexpected modification of errno(-1) -> 0x%%s\\n""" % (test_function, input_error_template), input_error_list + [("%d", "errno_cp")])

    code_template += """
                    return 1;
                };
                /** testing mathcall */
                if (test_mode.matherr_check) {
                    if (comp_result.matherr_call != matherr_call_expected) {
                        /** exception comparison */
    """
    code_template += "         %s;\n" % generate_printf(""" matherr call error for test %%s\\n for %s(%s), expected matherr_call=0x%%s, got 0x%%s\\n""" % (test_function, input_error_template), [("%d", "i")] + input_error_list + [("%d", "comp_result.matherr_call"), ("%d", "matherr_call_expected")])

    code_template += """
                        return 1;
                    };
                    if (matherr_call_expected) {
                        /** matherr errno and type checking */
                        if (comp_result.errno_f != matherr_err_expected) {
                            printf("exception->err error for test %d\\n", i);
                            printf("ERROR: e->err=%d was expected, got %d\\n", comp_result.errno_f, matherr_err_expected);
                            return 1;
                        };
                        if (comp_result.errtype != matherr_type_expected) {
                            printf("exception->type error for test %d\\n", i);
                            printf("ERROR: e->type=%d was expected, got %d\\n", comp_result.errtype, matherr_type_expected);
                            return 1;
                        };
                    }
                } else if (matherr_call_expected != 0) {
        """
    code_template += "         %s;\n" % generate_printf(""" matherr call error for test %%s\\n for %s(%s), expected matherr_call=0x%%s, got 0x%%s\\n""" % (test_function, input_error_template), [("%d", "i")] + input_error_list + [("%d", "comp_result.matherr_call"), ("%d", "matherr_call_expected")])

    code_template += """
                }
                #endif /* !KML_MODE */
                #endif /* RTL_RUN */
                #endif /* PERF_TEST */ 

            };
        };
    };
    return 0;
};"""
    return code_template


if __name__ == "__main__":
    str_format_map = {
        "fp32": ML_Binary32,
        "fp64": ML_Binary64,
    }
    test_mode_map = {
        "faithful": TEST_MODE_FAITHFUL,
        "cr"      : TEST_MODE_CR,
    }
    input_format_list = [str_format_map[i_format] for i_format in extract_option_value("--informat", "fp32").split(",")]
    result_format = str_format_map[extract_option_value("--outformat", "fp32")]
    test_function = extract_option_value("--test", "undefined")
    output_file   = extract_option_value("--output", "test_template.c")
    test_mode     = test_mode_map[extract_option_value("--test-mode", "faithful")]

    code = gen_template_test(input_format_list, result_format, test_function, test_mode = test_mode)

    output_stream = open(output_file, "w")
    output_stream.write(code)
    output_stream.close()
