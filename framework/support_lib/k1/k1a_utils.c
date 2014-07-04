#include <support_lib/ml_special_values.h>
#include <support_lib/ml_utils.h>
#include <support_lib/k1/k1a_utils.h>
#include <support_lib/ml_multi_prec_lib.h>
#include <HAL/hal/hal.h>
#include <cpu.h>
#include <inttypes.h>
#include <stdio.h>


double k1_fma_d4_rn(double x, double y, double z) {
    // saving context
    enum __k1_fpu_rounding_mode round_mode =  __k1_fpu_get_rounding_mode();
    unsigned int ev = __k1_fpu_get_exceptions();

    // round to nearest operation
    __k1_fpu_set_rounding_mode(_K1_FPU_NEAREST_EVEN);
    double result = ml_fma(x, y, z);

    // restoring context
    __k1_fpu_set_rounding_mode(round_mode);
    __k1_fpu_clear_exceptions(_K1_FPU_ALL_EXCEPTS);
    __k1_fpu_raise_exceptions(ev);

    return result;
}

ml_dd_t k1_fma_dd_d3_rn(double x, double y, double z) {
    // saving context
    enum __k1_fpu_rounding_mode round_mode =  __k1_fpu_get_rounding_mode();
    unsigned int ev = __k1_fpu_get_exceptions();

    // round to nearest operation
    __k1_fpu_set_rounding_mode(_K1_FPU_NEAREST_EVEN);
    ml_dd_t result = ml_fma_dd_d3(x, y, z);

    // restoring context
    __k1_fpu_set_rounding_mode(round_mode);
    __k1_fpu_clear_exceptions(_K1_FPU_ALL_EXCEPTS);
    __k1_fpu_raise_exceptions(ev);

    return result;
}

struct {
    enum __k1_fpu_rounding_mode rnd_mode;
    unsigned int ev;
} k1_fp_context;

void k1_save_fp_context() {
    k1_fp_context.rnd_mode = __k1_fpu_get_rounding_mode();
    k1_fp_context.ev       = __k1_fpu_get_exceptions();
}

void k1_restore_fp_context() {
    __k1_fpu_set_rounding_mode(k1_fp_context.rnd_mode);
    __k1_fpu_clear_exceptions(_K1_FPU_ALL_EXCEPTS);
    __k1_fpu_raise_exceptions(k1_fp_context.ev);
}



double k1_subnormalize_d_dd_i(ml_dd_t x, int scale_factor) {
    int ex = ml_exp_extraction_dirty_fp64(x.hi);
    int scaled_ex = ex + scale_factor;
    int delta = -1022 - scaled_ex;
    

    uint64_t hi_sign = (double_to_64b_encoding(x.hi) & 0x8000000000000000ull);

    uint64_t x_mant = (double_to_64b_encoding(x.hi) & 0x000fffffffffffffull) | ((uint64_t) 1 << 52);

    enum __k1_fpu_rounding_mode rnd_mode = __k1_fpu_get_rounding_mode();
    uint64_t lo_sign = ((double_to_64b_encoding(x.hi) ^ double_to_64b_encoding(x.lo)) >> 63) != 0;
    uint64_t sticky = ((x_mant << (64 - (delta-1))) != 0);
    uint64_t round_cst = delta <= 53 ? (uint64_t) 1 << (delta - 1) : 0;


    double rounded_value = -1;
    uint64_t round_bit = (x_mant & round_cst) != 0;
        
    if (rnd_mode == _K1_FPU_NEAREST_EVEN) {
        double pre_rounded_value = double_from_64b_encoding(double_to_64b_encoding(x.hi) + round_cst);

        uint64_t parity_bit = x_mant & (1 << delta);

        if ((sticky == 0 && parity_bit == 0 && x.lo == 0.0) || (sticky == 0 && lo_sign)) rounded_value = x.hi;
        else rounded_value = pre_rounded_value;

    } else {
        /*printf("round_cst: %"PRIx64"\n", round_cst);
        printf("round_bit: %d\n", round_bit);
        printf("sticky   : %d\n", sticky);
        printf("lo_sign  : %d\n", lo_sign);
        printf("rnd_mode : %d\n", rnd_mode);*/

        if (rnd_mode == _K1_FPU_TOWARDS_ZERO) {
            if (round_bit || sticky) rounded_value = x.hi;  
            else if (lo_sign && x.lo != 0.0) rounded_value = double_from_64b_encoding(double_to_64b_encoding(x.hi) - (round_cst << 1));
            else rounded_value = x.hi;
        } else if (rnd_mode == _K1_FPU_TOWARDS_PLUS_INF) {
            uint64_t inf_round_cst = delta <= 52 ? (round_cst << 1) : (round_cst);
            if (x.hi > 0) {
                if (round_bit || sticky || (x.lo != 0.0 && !lo_sign)) rounded_value = double_from_64b_encoding(double_to_64b_encoding(x.hi) + (inf_round_cst));
                else rounded_value = x.hi;
            } else {
                if (round_bit || sticky || x.lo == 0.0 || !lo_sign) rounded_value = x.hi;
                else rounded_value = double_from_64b_encoding(double_to_64b_encoding(x.hi) - (inf_round_cst));
            }
        } else if (rnd_mode == _K1_FPU_TOWARDS_MINUS_INF) {
            uint64_t inf_round_cst = delta <= 52 ? (round_cst << 1) : (round_cst);
            if (x.hi < 0) {
                if (round_bit || sticky || (x.lo != 0.0 && !lo_sign)) rounded_value = double_from_64b_encoding(double_to_64b_encoding(x.hi) + (inf_round_cst));
                else rounded_value = x.hi; 
            } else {
                if (round_bit || sticky || x.lo == 0.0 || !lo_sign) rounded_value = x.hi;
                else rounded_value = double_from_64b_encoding(double_to_64b_encoding(x.hi) - (inf_round_cst));
            }
        }
    }
    if (delta > 0 && (sticky != 0 || round_bit != 0 || x.lo != 0.0)) {
        __k1_fpu_raise_exceptions(_K1_FPU_UNDERFLOW);
    }
    int delta_s = delta >= 52 ? 52 : delta;
    return double_from_64b_encoding(((double_to_64b_encoding(rounded_value) >> delta_s) << delta_s) | hi_sign);
}


double k1_round_signed_overflow_fp64(double sign) {
    enum __k1_fpu_rounding_mode rnd_mode = __k1_fpu_get_rounding_mode();
    uint64_t usign = double_to_64b_encoding(sign) >> 63;
    switch(rnd_mode) {
        case _K1_FPU_NEAREST_EVEN:
            return usign ? fp64_sv_MinusInfty.d : fp64_sv_PlusInfty.d; 
        case _K1_FPU_TOWARDS_PLUS_INF:
            return usign ? fp64_sv_MinusOmega.d : fp64_sv_PlusInfty.d; 
        case _K1_FPU_TOWARDS_MINUS_INF:
            return usign ? fp64_sv_MinusInfty.d : fp64_sv_PlusOmega.d; 
        case _K1_FPU_TOWARDS_ZERO:
            return usign ? fp64_sv_MinusOmega.d : fp64_sv_PlusOmega.d; 
    };

    return fp64_sv_SNaN.d;
};

float  k1_round_signed_overflow_fp32(float  sign) {
    enum __k1_fpu_rounding_mode rnd_mode = __k1_fpu_get_rounding_mode();
    uint32_t usign = float_to_32b_encoding(sign) >> 31;
    switch(rnd_mode) {
        case _K1_FPU_NEAREST_EVEN:
            return usign ? fp32_sv_MinusInfty.f : fp32_sv_PlusInfty.f; 
        case _K1_FPU_TOWARDS_PLUS_INF:
            return usign ? fp32_sv_MinusOmega.f : fp32_sv_PlusInfty.f; 
        case _K1_FPU_TOWARDS_MINUS_INF:
            return usign ? fp32_sv_MinusInfty.f : fp32_sv_PlusOmega.f; 
        case _K1_FPU_TOWARDS_ZERO:
            return usign ? fp32_sv_MinusOmega.f : fp32_sv_PlusOmega.f; 
    };

    return fp32_sv_SNaN.f;
};
