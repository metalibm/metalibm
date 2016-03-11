#include <support_lib/ml_types.h>
#include <support_lib/ml_utils.h>
#include <support_lib/ml_multi_prec_lib.h>
#include <inttypes.h>
#include <stdio.h>


ml_dd_t ml_split_dd_d(double x) {
    /* Veltkamp split appied to p = 53, C = (1 << 27) + 1 */
    const uid_conv_t C = {.u = 0x41a0000002000000ull};
    double gamma = C.d * x;
    double delta = x - gamma;

    double xhi = gamma + delta;

    ml_dd_t result = { .hi = xhi, .lo = x - xhi};

    return result;
}

ml_dd_t ml_split_dd_d_safe(double x) {
    /* Veltkamp split appied to p = 53, C = (1 << 27) + 1 */
    const uid_conv_t C = {.u = 0x41a0000002000000ull};
    double gamma = C.d * x;
    double delta = x - gamma;

    double xhi = gamma + delta;

    ml_dd_t result = { .hi = xhi, .lo = x - xhi};


    return result;
}

ml_dd_t ml_mult_dd_d2(double x, double y) {
    ml_dd_t xs, ys;
    xs = ml_split_dd_d(x);
    ys = ml_split_dd_d(y);
    
    double r1, t1, t2, t3, r2;
    r1 = x * y;
    t1 = -r1 + xs.hi * ys.hi;
    t2 =  t1 + xs.hi * ys.lo;
    t3 =  t2 + xs.lo * ys.hi;
    r2 =  t3 + xs.lo * ys.lo;

    ml_dd_t result = {.hi = r1, .lo = r2};
    return result;
}

ml_dd_t ml_add_dd_d2(double x, double y) {
    /* implementation of the 2sum algorithm */
    double s = x + y;
    double xp = s - y;
    double yp = s - xp;
    double dx = x - xp;
    double dy = y - yp;
    double t = dx + dy;

    ml_dd_t result = {.hi = s, .lo = t};
    return result;
}

ml_dd_t ml_add_dd_d2_fast(double x, double y) {
    /** implementation of the fast2sum algorithm */
    double s = x + y;
    double z = s - x;
    double t = y - z;

    ml_dd_t result = {.hi = s, .lo = t};
    return result;
}

ml_dd_t ml_add_dd_d_dd(double x, ml_dd_t y) {
    /** ml_dd_t + double -> ml_dd_t 
     *  pre-conditions, y normalized and abs(y.hi) <= 2^(-2) abs(x)
     *
     * */
    ml_dd_t t1  = ml_add_dd_d2(x, y.hi);
    double t3 = t1.lo + y.lo;

    return ml_add_dd_d2(t1.hi, t3);
}


/** dummy implementation , TBD */
ml_dd_t ml_add_dd_dd2(ml_dd_t x, ml_dd_t y) {
    /** ml_dd_t + ml_dd_t -> ml_dd_t 
     *
     * */
    ml_dd_t t1  = ml_add_dd_d2(x.hi, y.hi);
    ml_dd_t t2   = ml_add_dd_d_dd(x.lo, t1);

    return ml_add_dd_d_dd(y.lo, t2);
}


double ml_fma(double x, double y, double z) {
    /** Implementend using Muller's Handbook of floating-point arithmetic, Dekker product, p135 */
    ml_dd_t mult_result = ml_mult_dd_d2(x, y);
    ml_dd_t acc_result = ml_add_dd_d_dd(z, mult_result);

    return acc_result.hi;
}


ml_dd_t ml_fma_dd_d3(double x, double y, double z) {
    /** Implementend using Muller's Handbook of floating-point arithmetic, Dekker product, p135 */
    ml_dd_t mult_result = ml_mult_dd_d2(x, y);
    ml_dd_t acc_result = ml_add_dd_d_dd(z, mult_result);

    return acc_result;
}


double ml_subnormalize_d_dd_i(ml_dd_t x, int scale_factor) {
    int ex = ml_exp_extraction_dirty_fp64(x.hi);
    int scaled_ex = ex + scale_factor;
    int delta = -1022 - scaled_ex;

    uint64_t round_cst = 1 << (delta - 1);
    double pre_rounded_value = double_from_64b_encoding(double_to_64b_encoding(x.hi) + round_cst);
    /*printf("round_cst:         %"PRIx64"\n", round_cst);
    printf("x.hi:              %"PRIx64"\n", x.hi);
    printf("pre_rounded_value: %"PRIx64"\n", pre_rounded_value);*/

    /*double round_cst = ml_sign_exp_insertion_fp64(ml_aligned_sign_extraction_fp64(x.hi), ex - 53 + delta);
    printf("round_cst: %"PRIx64"\n", double_to_64b_encoding(round_cst));
    double tmp = (round_cst + x.lo);
    printf("tmp: %"PRIx64"\n", tmp);
    double pre_rounded_value = tmp + x.hi; 
    printf("pre_rounded_value: %"PRIx64"\n", double_to_64b_encoding(pre_rounded_value));
    */

    uint64_t sticky = double_to_64b_encoding(x.hi) << (64 - (delta-1));
    uint64_t lo_sign = (double_to_64b_encoding(x.hi) ^ double_to_64b_encoding(x.lo)) >> 63;
    uint64_t parity_bit = double_to_64b_encoding(x.hi) & (1 << delta);
    //printf("sticky=%d, lo_sign=%d, parity_bit=%d\n", sticky != 0, lo_sign != 0, parity_bit != 0);

    double rounded_value = -1;
    if ((sticky == 0 && parity_bit == 0 && x.lo == 0.0) || (sticky == 0 && lo_sign)) rounded_value = x.hi;
    else rounded_value = pre_rounded_value;

    return double_from_64b_encoding((double_to_64b_encoding(rounded_value) >> delta) << delta);
}
