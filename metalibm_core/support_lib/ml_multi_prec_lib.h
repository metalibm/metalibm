#ifndef ML_MULTI_PREC_LIB
#define ML_MULTI_PREC_LIB

#include <stdint.h>
#include <support_lib/ml_types.h>


ml_dd_t ml_split_dd_d(double x);


ml_dd_t ml_mult_dd_d2(double x, double y);


ml_dd_t ml_add_dd_d2(double x, double y);


ml_dd_t ml_add_dd_d2_fast(double x, double y);


ml_dd_t ml_add_dd_d_dd(double x, ml_dd_t y);

/** dummy implementation , TBD */
ml_dd_t ml_add_dd_dd2(ml_dd_t x, ml_dd_t y);

double ml_fma(double x, double y, double z); 

ml_dd_t ml_fma_dd_d3(double x, double y, double z); 

/** assuming result exponent is exp(x) + scale_factor
 *  round field(x) accounting for subnormal cases */
double ml_subnormalize_d_dd_i(ml_dd_t x, int scale_factor);



#endif /* def ML_MULTI_PREC_LIB */
