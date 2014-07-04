#ifndef __RANDOM_GEN_HPP__
#define __RANDOM_GEN_HPP__

#include <stdlib.h>
#include <stdint.h>
#include <utils.hpp>
#include <gmp.h>
#include <mpfr.h>
#include <gmpxx.h>

mpz_class gen_n_bits_string(int n, gmp_randstate_t randstate);

#define FP_VALUES_TABLE_SIZE 18
static const uint32_t fp16_values[FP_VALUES_TABLE_SIZE] = {
	0x0000, 0x8000,
	0x7c00, 0xfc00,
	0x7e00, 0xfe00,
	0xfc01, 0xfc01,
	0x03ff, 0x83ff,
	0x0400, 0x8400,
	0xffff, 0x3c00,
	0xdead, 0xbeef,
	0xbfff, 0xdfff,
};


static const uint32_t fp32_values[] = {
	0x00000000, 0x80000000,
	0x7f800000, 0xff800000,
	0x7fc00000, 0xffc00000,
	0x7f800001, 0xff800001,
	0x007fffff, 0x807fffff,
	0x00800000, 0x80800000,
	0xffffffff, 0x3f800000,
	0xdeadbeef, 0xdeadbeef,
	0xbfffffff, 0xdfffffff,
};


static const uint64_t fp64_values[] = {
	0x0000000000000000, 0x8000000000000000,
	0x7ff0000000000000, 0xfff0000000000000,
	0x7ff8000000000000, 0xfff8000000000000,
	0x7ff0000000000001, 0xfff0000000000001,
	0x000fffffffffffff, 0x800fffffffffffff,
	0x0010000000000000, 0x8010000000000000,
	0xffffffffffffffff, 0x3ff0000000000000,
	0xbfffffffffffffff, 0xdfffffffffffffff,
	0xdeadbeefdeadbeef, 0xdeadbeefdeadbeef,
};

class ML_FloatingPointRNG {
    protected:
        gmp_randstate_t m_state;
    public:
        /** initialize and seed the RNG */
        void seed(int n);


        /** generate random IEEE subnormal number 
         * wE is the exponent field width
         * wF is the mantissa field width
         */
        mpz_class generateRandomIEEESubnormal(int wE, int wF);


        /** generate random IEEE normal number 
         * wE is the exponent field width
         * wF is the mantissa field width
         */
        mpz_class generateRandomIEEENormal(int wE, int wF);

        mpz_class generateTrickyValue(int wE, int wF, int& index);


        /** generate random floating-point number 
         * generation is randomized over several categories 
         * (subnormal, normal, static values, tricky values, bitfield uniform random)
         *  wE defines the format exponent width
         *  wF defines the format mantissa width
         *  case_id is the test id used to randomize the test case
         *  index is used to spread trickyValue generation 
         */
    	mpz_class generateIEEETestValue(int case_id, int wE, int wF, int &index);

        int generate_index();


        float generate_random_float_interval(float low, float high);

        float generate_random_float_interval_focus(float low, float high, int& gen_index);


        double generate_random_double_interval(double low, double high);

        double generate_random_double_interval_focus(double low, double high, int& gen_index);
};

#endif


