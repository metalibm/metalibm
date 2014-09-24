/** set errno and call matherr with underflow parameter
 * according to runtime value of _LIB_VERSION 
 */
float ml_raise_libm_underflowf(float result, float arg, char* func_name);


/** set errno and call matherr with overflow parameter
 * according to runtime value of _LIB_VERSION 
 */
float ml_raise_libm_overflowf(float result, float arg, char* func_name);
