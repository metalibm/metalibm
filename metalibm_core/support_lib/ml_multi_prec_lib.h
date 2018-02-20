#ifndef ML_MULTI_PREC_LIB
#define ML_MULTI_PREC_LIB

#include <support_lib/ml_types.h>


/** Split a double-precision number into two non over-lapping
 *  double precision number hi and lo , such than x = hi +lo
 *  and such both hi and lo significand size is half the size
 *  of a double precision mantissa
 *  @param x input value
 *  @return double double split result
 */
ml_dd_t ml_split_dd_d(double x);

/** Split a single precision number into two non over-lapping
 *  single precision number hi and lo , such than x = hi +lo
 *  @param x input value
 *  @return double single split result
 */
ml_ds_t ml_split_ds_s(float x);

/** Compute the exact result a @p x * @p y, using a double double
 *  representation to store the result */
ml_dd_t ml_mult_dd_d2(double x, double y);


/** Compute the exact result a @p x + @p y, using a double double
 *  representation to store the result. This function uses 
 *  the 2Sum algorithm (no hypothesis on operand order)
 */
ml_dd_t ml_add_dd_d2(double x, double y);

/** Compute the exact result a @p x + @p y, using a double double
 *  representation to store the result. This function uses 
 *  the Fast2Sum algorithm and assumes x >= y
 */
ml_dd_t ml_add_dd_d2_fast(double x, double y);


ml_dd_t ml_add_dd_d_dd(double x, ml_dd_t y);

/** dummy implementation , TBD */
ml_dd_t ml_add_dd_dd2(ml_dd_t x, ml_dd_t y);

double ml_fma(double x, double y, double z);

ml_dd_t ml_fma_dd_d3(double x, double y, double z);

/** assuming result exponent is exp(x) + scale_factor
 *  round field(x) accounting for subnormal cases */
double ml_subnormalize_d_dd_i(ml_dd_t x, int scale_factor);

/** Conversion from binary64 to compound
 *  double double format
 *  @param x conversion input
 *  @return result of conversion
 */
ml_dd_t ml_conv_dd_d(double x);


/** Unary negation
 *  @param x input value
 *  @return negated value
 */
ml_dd_t ml_neg_dd(ml_dd_t x);

#endif /* def ML_MULTI_PREC_LIB */
