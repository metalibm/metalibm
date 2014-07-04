#include <math.h>
#include <errno.h>
#include <test_lib.h>
#include <support_lib/ml_types.h>
#include <support_lib/ml_utils.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef __K1__
#include <cpu.h>
#endif /** __K1__ */



int fp32_compare(float expected, float result) {
    if (float_to_32b_encoding(expected) == float_to_32b_encoding(result)) return 1;
    else {
        if (ml_is_nanf(expected)) {
            return ml_is_nanf(result);
        }
        else return 0;
    }
}

int fp64_compare(double expected, double result) {
    if (double_to_64b_encoding(expected) == double_to_64b_encoding(result)) return 1;
    else {
        if (ml_is_nan(expected)) {
            return ml_is_nan(result);
        }
        else return 0;
    }
}

int fp32_is_faithful(float expected_rd, float expected_ru, float result) {
    return fp32_compare(expected_rd, result) || fp32_compare(expected_ru, result);
}

int fp64_is_faithful(double expected_rd, double expected_ru, double result) {
    return fp64_compare(expected_rd, result) || fp64_compare(expected_ru, result);
}

int fp32_is_cr(float expected_cr, float result) {
    return fp32_compare(expected_cr, result);
}

int fp64_is_cr(double expected_cr, double result) {
    return fp64_compare(expected_cr, result);
}

void set_rounding_mode(tb_round_mode_t rnd_mode) {
    #ifdef __k1__
    switch(rnd_mode) {
        case tb_round_nearest:
            __k1_fpu_set_rounding_mode(_K1_FPU_NEAREST_EVEN); break;
        case tb_round_toward_zero:
            __k1_fpu_set_rounding_mode(_K1_FPU_TOWARDS_ZERO); break;
        case tb_round_up:
            __k1_fpu_set_rounding_mode(_K1_FPU_TOWARDS_PLUS_INF); break;
        case tb_round_down:
            __k1_fpu_set_rounding_mode(_K1_FPU_TOWARDS_MINUS_INF); break;
    }
    #endif
}


extern int matherr_call_expected;
extern int matherr_type_expected;
extern int matherr_err_expected;
extern char* matherr_name_expected;

int matherr(struct exception *e) {
    matherr_call_expected = +1;
    /*if (e->err != matherr_err_expected) {
        printf("ERROR: e->err=%d was expected, got %d\n", matherr_err_expected, e->err);
        return -1;
    }
    if (e->type != matherr_type_expected) {
        printf("ERROR: e->type=%d was expected, got %d\n", matherr_type_expected, e->type);
        return -1;
    }*/
    /* specific to K1's newlib */
    #ifdef __K1__
    matherr_err_expected = e->err;
    #endif /** __K1__ */
    matherr_type_expected = e->type;
    if (strcmp(matherr_name_expected, e->name) != 0) {
        printf("ERROR: e->name=%s was expected, got %s\n", matherr_name_expected, e->name);
        return -1;
    }
    
    return 0; 
}
