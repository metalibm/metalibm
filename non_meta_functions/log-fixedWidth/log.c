/* 
 * this function computes log, correctly rounded, 
 * using  fixed-point arithmetic
 * 
 * THIS IS EXPERIMENTAL SOFTWARE
 * 
 * author: Julien Le Maire
 * date: March 2015
 * 
 * This function not portable, but should work on any 64 bits platform with clang
 * or gcc. It require:
 *   - an equivalent of the types int128_t and uint128_t
 *   - a function clz64 for counting leading zeros in a uint64_t
 * It can compile on any platform that provides something equivalent to thoses.
 * For example, on the microsoft platform, it can be implemented with __int128,
 * __mul128 and _BitScanReverse64.
 */
#include <inttypes.h>
#include <stddef.h>
#include <stdio.h>
#include <assert.h>
//#include <x86intrin.h>  // for __lzcnt64 on x86 that support it

double log_default_rn(double x);

/*** NON STANDART FUNCTIONS AND TYPES TO MANIPULATE FIXED-WIDTH NUMBERS ***/

static inline int clz64 (uint64_t x) {
	#if defined(__GNUC__)
	return __builtin_clzll (x);
	#else
	/* fast fallback method: cost ~10 instructions (~ 15 cycles) */
	static const uint8_t clz_lkup[256] = {
		64, 63, 62, 62, 61, 61, 61, 61, 60, 60, 60, 60, 60, 60, 60, 60,
		59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59,
		58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
		58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
		57, 57, 57, 57, 57, 57,	57, 57, 57, 57,	57, 57, 57, 57, 57, 57,
		57, 57, 57, 57, 57, 57,	57, 57, 57, 57,	57, 57, 57, 57, 57, 57,
		57, 57, 57, 57, 57, 57,	57, 57, 57, 57,	57, 57, 57, 57, 57, 57,
		57, 57, 57, 57, 57, 57,	57, 57, 57, 57,	57, 57, 57, 57, 57, 57,
		56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
		56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
		56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
		56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
		56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
		56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
		56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
		56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56
    };
	int n = ((x)   >= (UINT64_C(1) << 32)) * 32;
	n += ((x >> n) >= (UINT64_C(1) << 16)) * 16;
	n += ((x >> n) >= (UINT64_C(1) <<  8)) *  8;
	return clz_lkup[x >> n] - n;
	#endif
}

/* the following works on Gcc >= 4.6, Clang >= ?, Icc >= 13.0
for older Gcc, concider using  int __attribute__((__mode__(__TI__)))
  (this is available since at least sep 2001, see https://gcc.gnu.org/ml/gcc/2000-09/msg00419.html)
for older Icc and Msvc, try looking at  __mul128 and __mul128i  and do the extended multiplication
*/
typedef __int128_t int128_t;
typedef __uint128_t uint128_t;

#define UINT128(a,b) (((uint128_t)((uint64_t)(a)) << 64) | ((uint64_t)(b)))

#define HI(a) ((uint64_t)( ((uint128_t)(a)) >> 64 ))
#define LO(a) ((uint64_t)( ((uint128_t)(a)) ))

static inline uint128_t fullmul (uint64_t a, uint64_t b) {
	return (uint128_t)a * b;
}
static inline int128_t fullimul (int a, uint64_t b) {
	return (int128_t)a * b;
}
static inline uint64_t highmul (uint64_t a, uint64_t b) {
	return HI(fullmul(a, b));
}



/*** TABLES AND CONSTANTS USED FOR THE LOG ***/

#define ARG_REDUC_1_PREC 6
#define ARG_REDUC_1_TABLESIZE 64
#define ARG_REDUC_1_SIZE 9

#define ARG_REDUC_2_PREC 12
#define ARG_REDUC_2_TABLESIZE 80
#define ARG_REDUC_2_SIZE 14

#define IMPLICIT_ZEROS (52+ARG_REDUC_1_SIZE+ARG_REDUC_2_SIZE-64)

#define POLYNOMIAL_PREC 55

/* define to 1 to skip the end of the first step (and evaluate the second step instead):
 skip the polynom evaluation, the reconstruction of the solution and the error evaluation */
#define SKIP_END_OF_FIRST_STEP 0

/* define to 1 to use comacted log tables:
 It will store the values R[i] in a bitfield along with log(1/R[i]).
 This increase the data locality (as we read the same entry in the table),
 and the data consumption (24 bytes per entry instead of 26: save around 8%). */
#define COMPACT_LOGTABLES 0

/* 2**52 */
#define two52 4.50359962737049600000000000000000000000000000000000e+15

/* log(2) in fixed width */
#define log2fw_high UINT64_C(6243314768165359)    // bits 0 to -53
#define log2fw_mid UINT64_C(3853177435625389662)  // bits -54 to -117
#define log2fw_low UINT64_C(15413603158392197910) // bits -118 to -181


/* Log table for the argument reductions: */
#if COMPACT_LOGTABLES
	typedef struct {
		uint16_t val : 11;
		uint64_t log_hi : 53; // size of the field: ceil(log(last_entry_of_the_table.log_hi)/log(2))
		uint64_t log_mid;
		uint64_t log_lo;
	} ArgReduc1_t;
	typedef struct {
		uint16_t val : 16;
		uint64_t log_hi : 48;
		uint64_t log_mid;
		uint64_t log_lo;
	} ArgReduc2_t;
	#define ARG_REDUC_1_GETVALUE(ri) (argReduc1[ri].val+0)
	#define ARG_REDUC_2_GETVALUE(si) (argReduc2[si].val+0)
	/* généré par le scripte generate_logtables_compacted.sollya
	 à remplacer par le vrai résultat du script dans une version finale */
	#include "/tmp/logtables"
#else/*COMPACT_LOGTABLES*/
	typedef struct {
		uint64_t log_hi, log_mid, log_lo;
	} ArgReduc_t;
	#define ARG_REDUC_1_GETVALUE(ri) (argReduc1_val[ri]+0)
	#define ARG_REDUC_2_GETVALUE(si) (argReduc2_val[si]+0)
	/* généré par le scripte generate_logtables.sollya
	 à remplacer par le vrai résultat du script dans une version finale */
	#include "/tmp/logtables"
#endif/*COMPACT_LOGTABLES*/


double log_rn_accurate (uint128_t cstpart, uint64_t dz, int xe, uint64_t ri_low, uint64_t si_low)
{
	if (dz == 0 && xe == 0)
		return 0.0;
#if PRINT_DEBUG
	fprintf(stderr, "\nx=%a, dz=%016llx, xe=%d, mode=%d, ri=%u, si=%u\n", x, dz, xe, mode, ri, si);
#endif

	static const uint128_t d1 = UINT128(UINT64_C(0xffffffffffffffff), UINT64_C(0xffffffffffffffcd));
	static const uint128_t d2 = UINT128(UINT64_C(0x7fffffffffffffff), UINT64_C(0xfffffffff95d2edd));
	static const uint128_t d3 = UINT128(UINT64_C(0x5555555555555555), UINT64_C(0x5555441a272f4617));
	static const uint128_t d4 = UINT128(UINT64_C(0x3fffffffffffffff), UINT64_C(0xf008fd9c53b98d52));
	static const uint128_t d5 = UINT128(UINT64_C(0x3333333333332c14), UINT64_C(0x6ef40a305d92ecdb));
	static const uint128_t d6 = UINT128(UINT64_C(0x2aaaaaaaa8f50afb), UINT64_C(0x4b3747e8df1787bc));
	static const uint128_t d7 = UINT128(UINT64_C(0x249248eaa1852944), UINT64_C(0x3fa0804867361e8a));
	static const uint128_t d8 = UINT128(UINT64_C(0x1ffc04207b251a52), UINT64_C(0x2ec1d8cc930ba858));
	
	const uint64_t z0 = HI(d8)+1;
	const uint64_t z1 = HI(d7 - (fullmul(dz, z0) >> IMPLICIT_ZEROS));
	const uint64_t z2 = HI(d6 - (fullmul(dz, z1) >> IMPLICIT_ZEROS));
	const uint64_t z3 = HI(d5 - (fullmul(dz, z2) >> IMPLICIT_ZEROS));
	
	const uint128_t z4 = d4 - (fullmul(dz, z3) >> IMPLICIT_ZEROS);
	const uint128_t z5 = d3 - ((fullmul(dz, HI(z4)) + highmul(dz, LO(z4))) >> IMPLICIT_ZEROS);
	const uint128_t z6 = d2 - ((fullmul(dz, HI(z5)) + highmul(dz, LO(z5))) >> IMPLICIT_ZEROS);
	const uint128_t z7 = d1 - ((fullmul(dz, HI(z6)) + highmul(dz, LO(z6))) >> IMPLICIT_ZEROS);
	
	const uint128_t zapprox_low_tmp = fullmul(dz, LO(z7));
	const uint128_t zapprox_high = fullmul(dz, HI(z7)) + HI(zapprox_low_tmp);
	const uint64_t  zapprox_low = LO(zapprox_low_tmp);
	/* polynomial evaluation is  zapprox_low/2^(128+IMPLICIT_ZEROS) + zapprox_high/2^(64+IMPLICIT_ZEROS) */
	
#if PRINT_DEBUG
	fprintf(stderr, "  za_hi = %016llx %016llx\n", HI(zapprox_high), LO(zapprox_high));
	fprintf(stderr, "  za_lo = %016llx %016llx\n", HI(zapprox_low), LO(zapprox_low));
#endif

	const int128_t cstpart_low = ri_low + fullimul(xe, log2fw_low) + si_low;
	
	/* add zapprox_low, zapprox_high, cstpart, cstpart_low (aligned with cstpart_low)
	  and keep only the highest 128 bits of the results (ie: @FIX(11-128)) */
	const int128_t result_low_tmp = cstpart_low + (zapprox_low >> (IMPLICIT_ZEROS+11)) + (LO(zapprox_high)<<(64-IMPLICIT_ZEROS-11));
	uint128_t result_final = cstpart + (zapprox_high >> (IMPLICIT_ZEROS+11)) + (int64_t)(result_low_tmp >> 64);
	
#if PRINT_DEBUG
	fprintf(stderr, "  final = %016llx %016llx %016llx\n", HI(final_high), LO(final_high), LO(tmp_low));
#endif
	
	uint64_t sign = - (HI(result_final) >> 63);
//	if (sign) final_high = - final_high;
	result_final ^= UINT128(sign, sign);

#if PRINT_DEBUG
	fprintf(stderr, "  final = %016llx %016llx %016llx, sign=%d\n", HI(final_high), LO(final_high), sign ^ LO(tmp_low), !!sign);
#endif
	
	int exponent = 11 - (clz64(HI(result_final)) + 1);
	uint64_t mantissa = HI(result_final << (11 - exponent));
	
#if PRINT_DEBUG
	fprintf(stderr, "  exponent=%d, mantissa=%016llx\n", exponent, mantissa);
#endif
	
	/* no need to test if result is precise enouth: we know it is */
	uint64_t resultbits = ((uint64_t)sign << 63)
		+ ((uint64_t)(exponent+1023) << 52)
		+ (mantissa >> 12)
		+ ((mantissa >> 11) & 1); /* round to nearest */
	double result = (union { uint64_t u; double d; }){ resultbits }.d;
	return result;
}

double log_rn (const double x)
{
	uint64_t xbits;
	int xe;

	uint8_t ri, si;
	uint64_t y, dz, p;

	uint128_t longres, cstpart, zpzpart;

	int exponent;
	uint64_t mantissa, sign; // sign is 0 if result is positive, ~0 otherwise
	
	double result;
	uint64_t maxAbsErr, maxRelErr, resultbits;

	/* reinterpret x to manipulate its bits more easily */
	xbits = ((union { double d; uint64_t u; }){x}).u;
	xe = xbits >> 52;
	
	/* filter the spacial cases: !(x is normalized and 0 < x < +Inf) */
	if (0x7FEu <= (unsigned)xe - 1u) {
		/* x = +- 0:    raise a DivideByzero, return -Inf */
		if ((xbits & ~(1ull << 63)) == 0) return -1.0/0.0;
		/* x < 0.0:     raise a InvalidOperation, return a qNaN */
		if ((xbits &  (1ull << 63)) != 0) return (x-x)/0;
		/* x = qNaN:    return a qNaN
		   x = sNaN:    raise a InvalidOperation, return a qNaN
		   x = +Inf:    return +Inf */
		if (xe != 0) return x+x;
		/* x subnormal: change x to a normalized number */
		else {
			int u = clz64(xbits) - 12;
			xbits <<= u + 1;
			xe -= u;
		}
	}

	/* X = 2^xe * (xbits/2^52) */
	xe -= 1023;
	xbits = (xbits & 0xFFFFFFFFFFFFFull) + (UINT64_C(1) << 52);
	
	/* X = 2^xe * (1/R) * Y,
	 with  Y = y/2^(52 + ARG_REDUC_1_SIZE)
	 and 1/R = argReduc1[ri].val/2^ARG_REDUC_1_SIZE */
	ri = (xbits >> (52 - ARG_REDUC_1_PREC)) - (1u << ARG_REDUC_1_PREC);
	y = ARG_REDUC_1_GETVALUE(ri) * xbits;
	
	/* Y = (1/S) * (1 + dZ),
	 with dZ = dz/2^(52 + ARG_REDUC_1_SIZE + ARG_REDUC_2_SIZE)
	 and 1/S = argReduc2[si].val/2^ARG_REDUC_2_SIZE */
	si = (y >> (52 + ARG_REDUC_1_SIZE - ARG_REDUC_2_PREC)) - (1u << ARG_REDUC_2_PREC);
	dz = ARG_REDUC_2_GETVALUE(si) * y; // the integer part of the fixed-point is removed by overflow

	/* Compute part of the result that don't depend on Z (xe*log(2) + log(1/Ri) + log(1/Si)) */
	cstpart = fullimul(xe, log2fw_mid)
	        + UINT128((int64_t)xe * log2fw_high + (argReduc1[ri].log_hi + argReduc2[si].log_hi), argReduc1[ri].log_mid)
	        + argReduc2[si].log_mid;
	
	#if SKIP_END_OF_FIRST_STEP
	return log_rn_extended (cstpart, dz, xe, argReduc1[ri].log_lo, argReduc2[si].log_lo);
	#else
	
	/* Polynomial approximation of log(1+Z)/Z ~= P(Z), and evaluate Z*P(Z) */
	p = (UINT64_C(0xffffffffffffffa4)
	     - (highmul(dz, UINT64_C(0x7ffffffffeabf9c5)
	                    - (highmul(dz, UINT64_C(0x555554f70538f907)
	                                   - (highmul(dz, UINT64_C(0x3ff8278711338aba)) >> IMPLICIT_ZEROS)
	                      ) >> IMPLICIT_ZEROS)
	       ) >> IMPLICIT_ZEROS)
	    );
	zpzpart = fullmul(dz, p);
	
	/* Assemble the two parts, compute the sign, mantissa and exponent */
	longres = cstpart + (zpzpart >> (11 + IMPLICIT_ZEROS));
	sign = - (HI(longres) >> 63);   // sign is 0 if result > 0, and ~0 otherwise
	// if sign != 0, this is longres = ~ longres: it approximate the absolute value (-a = ~a + 1)
	// to avoid the approximation, do: longres = ((int64_t)sign + longres) ^ UINT128(sign, sign);
	longres ^= UINT128(sign, sign);
	
	int u = clz64(HI(longres)) + 1;
	exponent = 11 - u;
	mantissa = HI(longres << u);
	
	/* Compute the maximal absolute error (aligned with longres):
	     + 2 + abs(xe)      for xe*log(2), log(1/Ri) and log(1/Si)
	     + 1 + zpzpart>>(POLYNOMIAL_PREC+IMPLICIT_ZEROS+11) for the polynomial
	 If result*(1 +- maxRelErr) are not rounded to the same number, we need more precision */
	maxAbsErr = 3 + abs(xe) + (HI(zpzpart) >> (POLYNOMIAL_PREC + IMPLICIT_ZEROS + 11 - 64));
	maxRelErr = (maxAbsErr >> (64 - u)) + 1;
	if (((mantissa + maxRelErr) ^ (mantissa - maxRelErr)) >> 11) {
		#if EVAL_PERF
		  crlibm_second_step_taken++;
		#endif
		return log_rn_accurate (cstpart, dz, xe, argReduc1[ri].log_lo, argReduc2[si].log_lo);
	}
	
	/* Assemble the computed result */
	resultbits = ((uint64_t)sign << 63)
		+ ((uint64_t)(exponent+1023) << 52)
		+ (mantissa >> 12)
		+ ((mantissa >> 11) & 1); /* round to nearest */
	result = (union { uint64_t u; double d; }){ resultbits }.d;
	return result;
	#endif/*SKIP_END_OF_FIRST_STEP*/
}
