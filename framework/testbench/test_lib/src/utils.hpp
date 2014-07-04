#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <stdlib.h>
#include <stdint.h>
#include <gmp.h>
#include <mpfr.h>
#include <gmpxx.h>
#include <math.h>
#include <errno.h>
#include <test_lib.h>
#include <iostream>
#include <sstream>



/** enumerated type of IEEE-754 special values */
typedef enum {
    plus_infty,
    minus_infty,
    qnan,
    snan,
    plus_zero,
    minus_zero,
    plus_omega,
    minus_omega,
    close_omega,
    close_subnormal,
    close_zero,
    close_one
} ieee_type_value_t;



/** conversion from mpz (binary encoded fp32 number) to float type */ 
float mpz_to_float(mpz_class);


/** conversion from mpz (binary encoded fp64 number) to double type */ 
double mpz_to_double(mpz_class);


/** set mpz_class number from uint64_t value */
mpz_class set_mpz_from_uint64(uint64_t);


/** set uint64_t number from mpz_class value */
uint64_t set_uint64_from_mpz(mpz_class);


mpz_class set_mpz_from_double_encoding(double);

void clear_mpfr_exception(void);

/** test if normal result has been obtained by rounding up a subnormal result 
 *  (with unbounded precision) */
bool is_hidden_underflowf(float value, int ternary);


void set_mpfr_binary32_env();
void set_mpfr_binary64_env();


std::string hex_value64(uint64_t value);
std::string hex_value32(uint32_t value);


#endif
