/*******************************************************************************
* This file is part of Kalray's Metalibm tool
* Copyright (2020)
* All rights reserved
* created:          Jun 08, 2020
* last-modified:    Dec 08, 2020
*
* author(s): Nicolas Brunie (nicolas.brunie@kalrayinc.com)
*
* Description: single-header support library for metalibm
*
*******************************************************************************/
#ifndef __ML_SUPPORT_LIB_H__
#define __ML_SUPPORT_LIB_H__

#include <inttypes.h>
#include <stdbool.h>
#include <math.h>

/*******************************************************************************
 *                        Metalibm basic types
 ******************************************************************************/

/** conversion union for bitfield/binary32 cast */
typedef union {
    uint32_t u;
    int32_t i;
    float f;
} uif_conv_t;

/** conversion union for bitfield/binary64 cast */
typedef union {
    uint64_t u;
    int64_t i;
    double d;
} uid_conv_t;

typedef struct {
    double hi;
    double lo;
} ml_dd_t;

typedef struct {
    double hi;
    double me;
    double lo;
} ml_td_t;

typedef struct {
    float hi;
    float lo;
} ml_ds_t;

/*******************************************************************************
 *                        Metalibm special values
 ******************************************************************************/

/** binary32 special values */
static const uif_conv_t fp32_sv_PlusInfty  = {.u = 0x7f800000};
static const uif_conv_t fp32_sv_MinusInfty = {.u = 0xff800000};
static const uif_conv_t fp32_sv_PlusOmega  = {.u = 0x7f7fffff};
static const uif_conv_t fp32_sv_MinusOmega = {.u = 0xff7fffff};
static const uif_conv_t fp32_sv_PlusZero   = {.u = 0x00000000};
static const uif_conv_t fp32_sv_MinusZero  = {.u = 0x80000000};
static const uif_conv_t fp32_sv_QNaN       = {.u = 0xffffffff};
static const uif_conv_t fp32_sv_SNaN       = {.u = 0xffbfffff};


/** binary64 special values */
static const uid_conv_t fp64_sv_PlusInfty  = {.u = 0x7ff0000000000000ull};
static const uid_conv_t fp64_sv_MinusInfty = {.u = 0xfff0000000000000ull};
static const uid_conv_t fp64_sv_PlusOmega  = {.u = 0x7fefffffffffffffull};
static const uid_conv_t fp64_sv_MinusOmega = {.u = 0xffefffffffffffffull};
static const uid_conv_t fp64_sv_PlusZero   = {.u = 0x0000000000000000ull};
static const uid_conv_t fp64_sv_MinusZero  = {.u = 0x8000000000000000ull};
static const uid_conv_t fp64_sv_QNaN       = {.u = 0xffffffffffffffffull};
static const uid_conv_t fp64_sv_SNaN       = {.u = 0xfff7ffffffffffffull};

/*******************************************************************************
 *                        Metalibm utility function
 ******************************************************************************/

int ml_isnanf(float);


/** cast from a binary32 float to its uint32_t binary encoding */
//uint32_t float_to_32b_encoding(float v);
static inline uint32_t float_to_32b_encoding(float v) {
    uif_conv_t conv_tmp;
    conv_tmp.f = v;
    return conv_tmp.u;
}


/** cast from a uin32_t binary encoding to its binary32 float value */
//float float_from_32b_encoding(uint32_t v);
static inline float float_from_32b_encoding(uint32_t v) {
    uif_conv_t conv_tmp;
    conv_tmp.u = v;
    return conv_tmp.f;
}


static inline uint64_t double_to_64b_encoding(double v) {
    uid_conv_t conv_tmp;
    conv_tmp.d = v;
    return conv_tmp.u;
}

static inline double double_from_64b_encoding(uint64_t v) {
    uid_conv_t conv_tmp;
    conv_tmp.u = v;
    return conv_tmp.d;
}

/** Metalibm implementation of x * y + z with single rounding 
 * rely on HW FMA (dot not use on architecture not providing one)
 *  */
static inline float ml_fmaf(float x, float y, float z) {
    double dx = x;
    double dy = y;
    double dz = z;

    float result = (dx * dy) + dz;
    return result;
}

/** count leading zeroes */
#if defined(__GNUC__)
static inline int ml_count_leading_zeros_32b (uint32_t x) {
    return (x == 0) ? 0 : __builtin_clz(x);
}
static inline int ml_count_leading_zeros_64b (uint64_t x) {
    return (x == 0) ? 0 : __builtin_clzl(x);
}
#else
static const uint8_t ml_clz_lkup[256] = {
    64, 63, 62, 62, 61, 61, 61, 61, 60, 60, 60, 60, 60, 60, 60, 60,
    59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57,
    57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57,
    57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57,
    57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57,
    56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
    56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
    56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
    56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
    56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
    56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
    56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56,
    56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56
};
static inline int ml_count_leading_zeros_32b (uint32_t x) {
    int n = ((x)   >= (UINT32_C(1) << 16)) * 16;
    n += ((x >> n) >= (UINT32_C(1) <<  8)) *  8;
    return clz_lkup[x >> n] - n;
}
static inline int ml_count_leading_zeros_64b (uint64_t x) {
    int n = ((x)   >= (UINT64_C(1) << 32)) * 32;
    n += ((x >> n) >= (UINT64_C(1) << 16)) * 16;
    n += ((x >> n) >= (UINT64_C(1) <<  8)) *  8;
    return clz_lkup[x >> n] - n;
}
#endif



#define ml_exp_insertion_fp32(k) (float_from_32b_encoding(((k + 127) << 23) & 0x7f800000))
#define ml_exp_insertion_fp64(k) (double_from_64b_encoding(((uint64_t) (k + 1023) << 52) & 0x7ff0000000000000ull))

#define ml_exp_insertion_no_offset_fp32(k) (float_from_32b_encoding(((k) << 23)))
#define ml_exp_insertion_no_offset_fp64(k) (double_from_64b_encoding(((uint64_t) (k) << 52)))

#define ml_sign_exp_insertion_fp64(aligned_sign, k) (double_from_64b_encoding((((uint64_t) (k + 1023) << 52) & 0x7ff0000000000000ull)| (aligned_sign)))

#define ml_aligned_sign_extraction_fp64(x) (double_to_64b_encoding(x) & 0x8000000000000000ull)

#define ml_exp_extraction_dirty_fp32(x) (((float_to_32b_encoding(x) >> 23) & 0xff) - 127) 
#define ml_exp_extraction_dirty_fp64(x) (((double_to_64b_encoding(x) >> 52) & 0x7ff) - 1023)

#define ml_mantissa_extraction_fp32(x)  (float_from_32b_encoding((float_to_32b_encoding(x) & 0x807fffff) | 0x3f800000))
#define ml_mantissa_extraction_fp64(x)  (double_from_64b_encoding((double_to_64b_encoding(x) & 0x800fffffffffffffull) | 0x3ff0000000000000ull))

#define ml_raw_mantissa_extraction_fp32(x) (float_to_32b_encoding(x) & 0x007FFFFF) 
#define ml_raw_mantissa_extraction_fp64(x) (double_to_64b_encoding(x) & 0x000fffffffffffffull) 

#define ml_is_normal_positive_fp64(x) ((uint64_t)(double_to_64b_encoding(x) >> 52) - 1u < 0x7FEu)
#define ml_is_normal_positive_fp32(x) ((uint64_t)(double_to_32b_encoding(x) >> 23) - 1u < 0x0FEu)

#define ml_is_nan_or_inff(x) ((float_to_32b_encoding(x) & 0x7f800000u) == 0x7f800000)
#define ml_is_nan_or_inf(x) ((double_to_64b_encoding(x) & 0x7ff0000000000000ull) == 0x7ff0000000000000ull)

#define ml_is_nanf(x) (((float_to_32b_encoding(x) & 0x7f800000u) == 0x7f800000) && (float_to_32b_encoding(x) & 0x007fffffu) != 0)
#define ml_is_nan(x) (((double_to_64b_encoding(x) & 0x7ff0000000000000ull) == 0x7ff0000000000000ull) && ((double_to_64b_encoding(x) & 0x000fffffffffffffull) != 0))

#define ml_is_signaling_nanf(x) (ml_is_nanf(x) && ((float_to_32b_encoding(x) & 0x00400000) == 0))
#define ml_is_signaling_nan(x) (ml_is_nan(x) && ((double_to_64b_encoding(x) & 0x0008000000000000ull) == 0))

#define ml_is_quiet_nanf(x) (ml_is_nanf(x) && ((float_to_32b_encoding(x) & 0x00400000) != 0))
#define ml_is_quiet_nan(x) (ml_is_nan(x) && ((double_to_64b_encoding(x) & 0x0008000000000000ull) != 0))

#define ml_is_inff(x) ((float_to_32b_encoding(x) & 0x7fffffffu) == 0x7f800000)
#define ml_is_inf(x) ((double_to_64b_encoding(x) & 0x7fffffffffffffffull) == 0x7ff0000000000000ull)

#define ml_is_plus_inff(x) ((float_to_32b_encoding(x)) == 0x7f800000)
#define ml_is_plus_inf(x) ((double_to_64b_encoding(x)) == 0x7ff0000000000000ull)

#define ml_is_minus_inff(x) ((float_to_32b_encoding(x)) == 0xff800000)
#define ml_is_minus_inf(x) ((double_to_64b_encoding(x)) == 0xfff0000000000000ull)

#define ml_is_zerof(x) ((float_to_32b_encoding(x) & 0x7fffffffu) == 0)
#define ml_is_zero(x)  ((double_to_64b_encoding(x) & 0x7fffffffffffffffull) == 0)

#define ml_is_positivezerof(x) ((float_to_32b_encoding(x)) == 0)
#define ml_is_positivezero(x)  ((double_to_64b_encoding(x)) == 0)

#define ml_is_negativezerof(x) (float_to_32b_encoding(x) == 0x80000000u)
#define ml_is_negativezero(x)  (double_to_64b_encoding(x) == 0x8000000000000000ull)

#define ml_is_normalf(x) (float_to_32b_encoding(x) & 0x7f800000u)
#define ml_is_normal(x) (double_to_64b_encoding(x) & 0x7ff0000000000000ull)

#define ml_is_subnormalf(x) ((float_to_32b_encoding(x) & 0x7f800000u) == 0)
#define ml_is_subnormal(x) ((double_to_64b_encoding(x) & 0x7ff0000000000000ull) == 0)


#define ml_comp_signf(x, y) ((float_to_32b_encoding(x) ^ float_to_32b_encoding(y)) & 0x80000000u)
#define ml_comp_sign(x, y)  ((double_to_64b_encoding(x) ^ double_to_64b_encoding(y)) >> 63)

#define ml_copy_signf(x, y) (float_from_32b_encoding((float_to_32b_encoding(x) & 0x80000000u) | float_to_32b_encoding(y)))
#define ml_copy_sign(x, y) (double_from_64b_encoding((double_to_64b_encoding(x) & 0x8000000000000000ull) | double_to_64b_encoding(y)))

/*******************************************************************************
 *                   Metalibm vector format
 ******************************************************************************/
/** Return a pointer to the address of the vector field '_'
 *  of vector value @p x
 */
#define GET_VEC_FIELD_ADDR(x) (&(x))


// single precision vector format
typedef float ml_float2_t __attribute__ ((vector_size (8)));
typedef float ml_float4_t __attribute__ ((vector_size (16)));
typedef float ml_float8_t __attribute__ ((vector_size (32)));


// double precision vector format
typedef double ml_double2_t __attribute__ ((vector_size (16)));
typedef double ml_double4_t __attribute__ ((vector_size (32)));
typedef double ml_double8_t __attribute__ ((vector_size (64)));

// 32-b integer vector format
typedef int32_t ml_int2_t __attribute__ ((vector_size (8)));
typedef int32_t ml_int4_t __attribute__ ((vector_size (16)));
typedef int32_t ml_int8_t __attribute__ ((vector_size (32)));

// 32-b unsigned integer vector format
typedef uint32_t ml_uint2_t __attribute__ ((vector_size (8)));
typedef uint32_t ml_uint4_t __attribute__ ((vector_size (16)));
typedef uint32_t ml_uint8_t __attribute__ ((vector_size (32)));

// 64-b integer vector format
typedef int64_t ml_long2_t __attribute__ ((vector_size (16)));
typedef int64_t ml_long4_t __attribute__ ((vector_size (32)));
typedef int64_t ml_long8_t __attribute__ ((vector_size (64)));

// 64-b unsigned integer vector format
typedef uint64_t ml_ulong2_t __attribute__ ((vector_size (16)));
typedef uint64_t ml_ulong4_t __attribute__ ((vector_size (32)));
typedef uint64_t ml_ulong8_t __attribute__ ((vector_size (64)));

// boolean vector formats as 32-b integer vector format
typedef int32_t ml_bool2_t __attribute__ ((vector_size (8)));
typedef int32_t ml_bool4_t __attribute__ ((vector_size (16)));
typedef int32_t ml_bool8_t __attribute__ ((vector_size (32)));

// boolean vector formats as 64-b integer vector format
typedef int64_t ml_lbool2_t __attribute__ ((vector_size (16)));
typedef int64_t ml_lbool4_t __attribute__ ((vector_size (32)));
typedef int64_t ml_lbool8_t __attribute__ ((vector_size (64)));


/** Union type for 4x 32-b element vectors
 * TODO/FIXME: hazardous use of union to cast: accessing a same union with 2
 *             distinct accessors is an undefined-behavior (at least in c++)
 */
typedef union {
    __int128_t i128;
    ml_uint4_t u;
    ml_int4_t i;
    ml_float4_t f;
    ml_float2_t v2f[2];
    ml_int2_t v2i[2];
} ml_vec4_32b_uif;

/** Multi-precision vector format */
#define DEC_ML_MP2_VEC_FORMAT(FORMAT_NAME, FIELD_FORMAT) \
typedef struct { \
    FIELD_FORMAT hi, lo; \
} FORMAT_NAME;

DEC_ML_MP2_VEC_FORMAT(ml_dualfloat2_t, ml_float2_t)
DEC_ML_MP2_VEC_FORMAT(ml_dualfloat4_t, ml_float4_t)
DEC_ML_MP2_VEC_FORMAT(ml_dualfloat8_t, ml_float8_t)

DEC_ML_MP2_VEC_FORMAT(ml_dualdouble2_t, ml_double2_t)
DEC_ML_MP2_VEC_FORMAT(ml_dualdouble4_t, ml_double4_t)
DEC_ML_MP2_VEC_FORMAT(ml_dualdouble8_t, ml_double8_t)

#define DEC_ML_MP3_VEC_FORMAT(FORMAT_NAME, FIELD_FORMAT) \
typedef struct { \
    FIELD_FORMAT hi, me, lo; \
} FORMAT_NAME;

DEC_ML_MP3_VEC_FORMAT(ml_trifloat2_t, ml_float2_t)
DEC_ML_MP3_VEC_FORMAT(ml_trifloat4_t, ml_float4_t)
DEC_ML_MP3_VEC_FORMAT(ml_trifloat8_t, ml_float8_t)

DEC_ML_MP3_VEC_FORMAT(ml_tridouble2_t, ml_double2_t)
DEC_ML_MP3_VEC_FORMAT(ml_tridouble4_t, ml_double4_t)
DEC_ML_MP3_VEC_FORMAT(ml_tridouble8_t, ml_double8_t)


/*******************************************************************************
 *                   Metalibm vector library
 ******************************************************************************/

#define VECTORIZE_OP1(OP, r, x, size) {\
  unsigned i;\
  for (i = 0; i < size; ++i) (*(r))[i] = OP((x)[i]);\
}

static inline ml_float4_t v4si_cast2_v4sf(ml_int4_t v) {
    ml_vec4_32b_uif vconv = {.i = v};
    return vconv.f;
}

static inline ml_int4_t v4sf_cast2_v4si(ml_float4_t v) {
    ml_vec4_32b_uif vconv = {.f = v};
    return vconv.i;
}


#define DEF_ML_VECTOR_PRIMITIVES_OP3(FUNC_NAME, VECTOR_FORMAT, SCALAR_FORMAT, VECTOR_SIZE, SCALAR_OP0, SCALAR_OP1) \
static inline void FUNC_NAME(VECTOR_FORMAT *r, VECTOR_FORMAT vop0, VECTOR_FORMAT vop1, VECTOR_FORMAT vop2) {\
  unsigned i;\
  for (i = 0; i < VECTOR_SIZE; ++i) {\
    (*r)[i] = vop0[i] SCALAR_OP0 vop1[i] SCALAR_OP1 vop2[i];\
  };\
}
#define DEF_ML_VECTOR_PRIMITIVES_FUNC3(FUNC_NAME, VECTOR_FORMAT, SCALAR_FORMAT, VECTOR_SIZE, SCALAR_FUNC, SCALAR_OP0, SCALAR_OP1, SCALAR_OP2) \
static inline void FUNC_NAME(VECTOR_FORMAT *r, VECTOR_FORMAT vop0, VECTOR_FORMAT vop1, VECTOR_FORMAT vop2) {\
  unsigned i;\
  for (i = 0; i < VECTOR_SIZE; ++i) {\
    (*r)[i] = SCALAR_FUNC(SCALAR_OP0(vop0[i]), SCALAR_OP1(vop1[i]), SCALAR_OP2( vop2[i]));\
  };\
}
#define DEF_ML_VECTOR_PRIMITIVES_OP2(FUNC_NAME, VECTOR_FORMAT, SCALAR_FORMAT, VECTOR_SIZE, SCALAR_OP) \
static inline void FUNC_NAME(VECTOR_FORMAT *r, VECTOR_FORMAT vop0, VECTOR_FORMAT vop1) {\
  unsigned i;\
  for (i = 0; i < VECTOR_SIZE; ++i) {\
    (*r)[i] = vop0[i] SCALAR_OP vop1[i];\
  };\
}
#define DEF_ML_VECTOR_PRIMITIVES_OP1(FUNC_NAME, VECTOR_FORMAT, SCALAR_FORMAT, VECTOR_SIZE, SCALAR_OP) \
static inline void FUNC_NAME(VECTOR_FORMAT *r, VECTOR_FORMAT vop) {\
  unsigned i;\
  for (i = 0; i < VECTOR_SIZE; ++i) {\
    (*r)[i] = SCALAR_OP vop[i];\
  };\
}

#define ML_ASSEMBLE_VECTOR(vr, va, vb, size_a, size_b) {\
  int __k; \
  for(__k = 0; __k < (size_a); __k++) (*(vr))[__k] = (va)[__k];\
  for(__k = 0; __k < (size_b); __k++) (*(vr))[__k + (size_a)] = (vb)[__k];\
}


/** Vector Division */
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vdivf2, ml_float2_t, float, 2, /)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vdivf4, ml_float4_t, float, 4, /)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vdivf8, ml_float8_t, float, 8, /)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vdivd2, ml_double2_t, double, 2, /)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vdivd4, ml_double4_t, double, 4, /)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vdivd8, ml_double8_t, double, 8, /)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vdivi2, ml_int2_t, int32_t, 2, /)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vdivi4, ml_int4_t, int32_t, 4, /)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vdivi8, ml_int8_t, int32_t, 8, /)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vdivu2, ml_uint2_t, uint32_t, 2, /)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vdivu4, ml_uint4_t, uint32_t, 4, /)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vdivu8, ml_uint8_t, uint32_t, 8, /)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vdivl2, ml_long2_t, int64_t, 2, /)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vdivl4, ml_long4_t, int64_t, 4, /)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vdivl8, ml_long8_t, int64_t, 8, /)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vdivul2, ml_ulong2_t, uint64_t, 2, /)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vdivul4, ml_ulong4_t, uint64_t, 4, /)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vdivul8, ml_ulong8_t, uint64_t, 8, /)

/** Vector Modulo */
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vmodi2, ml_int2_t, int32_t, 2, %)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vmodi4, ml_int4_t, int32_t, 4, %)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vmodi8, ml_int8_t, int32_t, 8, %)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vmodu2, ml_uint2_t, uint32_t, 2, %)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vmodu4, ml_uint4_t, uint32_t, 4, %)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vmodu8, ml_uint8_t, uint32_t, 8, %)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vmodl2, ml_long2_t, int64_t, 2, %)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vmodl4, ml_long4_t, int64_t, 4, %)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vmodl8, ml_long8_t, int64_t, 8, %)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vmodul2, ml_ulong2_t, uint64_t, 2, %)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vmodul4, ml_ulong4_t, uint64_t, 4, %)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vmodul8, ml_ulong8_t, uint64_t, 8, %)


/** Vector Logic Left Shift */
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vslli2, ml_int2_t, int32_t, 2, <<)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vslli4, ml_int4_t, int32_t, 4, <<)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vslli8, ml_int8_t, int32_t, 8, <<)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsllu2, ml_uint2_t, uint32_t, 2, <<)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsllu4, ml_uint4_t, uint32_t, 4, <<)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsllu8, ml_uint8_t, uint32_t, 8, <<)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vslll2, ml_long2_t, int64_t, 2, <<)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vslll4, ml_long4_t, int64_t, 4, <<)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vslll8, ml_long8_t, int64_t, 8, <<)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsllul2, ml_ulong2_t, uint64_t, 2, <<)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsllul4, ml_ulong4_t, uint64_t, 4, <<)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsllul8, ml_ulong8_t, uint64_t, 8, <<)


/** Vector Logic Right Shift */
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsrli2, ml_int2_t, int32_t, 2, >>)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsrli4, ml_int4_t, int32_t, 4, >>)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsrli8, ml_int8_t, int32_t, 8, >>)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsrlu2, ml_uint2_t, uint32_t, 2, >>)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsrlu4, ml_uint4_t, uint32_t, 4, >>)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsrlu8, ml_uint8_t, uint32_t, 8, >>)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsrll2, ml_long2_t, int64_t, 2, >>)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsrll4, ml_long4_t, int64_t, 4, >>)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsrll8, ml_long8_t, int64_t, 8, >>)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsrlul2, ml_ulong2_t, uint64_t, 2, >>)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsrlul4, ml_ulong4_t, uint64_t, 4, >>)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsrlul8, ml_ulong8_t, uint64_t, 8, >>)

/** Vector Arithmethic Right Shift */
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsrai2, ml_int2_t, int32_t, 2, >>)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsrai4, ml_int4_t, int32_t, 4, >>)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsrai8, ml_int8_t, int32_t, 8, >>)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsrau2, ml_uint2_t, uint32_t, 2, >>)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsrau4, ml_uint4_t, uint32_t, 4, >>)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsrau8, ml_uint8_t, uint32_t, 8, >>)

/** Vector Fused Multiply and Add */
DEF_ML_VECTOR_PRIMITIVES_FUNC3(ml_vfmaf2, ml_float2_t, float, 2, fmaf, +, +, +)
DEF_ML_VECTOR_PRIMITIVES_FUNC3(ml_vfmaf4, ml_float4_t, float, 4, fmaf, +, +, +)
DEF_ML_VECTOR_PRIMITIVES_FUNC3(ml_vfmaf8, ml_float8_t, float, 8, fmaf, +, +, +)

DEF_ML_VECTOR_PRIMITIVES_FUNC3(ml_vfmad2, ml_double2_t, double, 2, fma, +, +, +)
DEF_ML_VECTOR_PRIMITIVES_FUNC3(ml_vfmad4, ml_double4_t, double, 4, fma, +, +, +)
DEF_ML_VECTOR_PRIMITIVES_FUNC3(ml_vfmad8, ml_double8_t, double, 8, fma, +, +, +)

DEF_ML_VECTOR_PRIMITIVES_OP3(ml_vfmai2, ml_int2_t, int32_t, 2, *, +)
DEF_ML_VECTOR_PRIMITIVES_OP3(ml_vfmai4, ml_int4_t, int32_t, 4, *, +)
DEF_ML_VECTOR_PRIMITIVES_OP3(ml_vfmai8, ml_int8_t, int32_t, 8, *, +)

DEF_ML_VECTOR_PRIMITIVES_OP3(ml_vfmau2, ml_uint2_t, uint32_t, 2, *, +)
DEF_ML_VECTOR_PRIMITIVES_OP3(ml_vfmau4, ml_uint4_t, uint32_t, 4, *, +)
DEF_ML_VECTOR_PRIMITIVES_OP3(ml_vfmau8, ml_uint8_t, uint32_t, 8, *, +)

DEF_ML_VECTOR_PRIMITIVES_FUNC3(ml_vfmsf2, ml_float2_t, float, 2, fmaf, +, +, -)
DEF_ML_VECTOR_PRIMITIVES_FUNC3(ml_vfmsf4, ml_float4_t, float, 4, fmaf, +, +, -)
DEF_ML_VECTOR_PRIMITIVES_FUNC3(ml_vfmsf8, ml_float8_t, float, 8, fmaf, +, +, -)

DEF_ML_VECTOR_PRIMITIVES_FUNC3(ml_vfmsd2, ml_double2_t, double, 2, fma, +, +, -)
DEF_ML_VECTOR_PRIMITIVES_FUNC3(ml_vfmsd4, ml_double4_t, double, 4, fma, +, +, -)
DEF_ML_VECTOR_PRIMITIVES_FUNC3(ml_vfmsd8, ml_double8_t, double, 8, fma, +, +, -)

DEF_ML_VECTOR_PRIMITIVES_FUNC3(ml_vfmsnf2, ml_float2_t, float, 2, fmaf, -, +, +)
DEF_ML_VECTOR_PRIMITIVES_FUNC3(ml_vfmsnf4, ml_float4_t, float, 4, fmaf, -, +, +)
DEF_ML_VECTOR_PRIMITIVES_FUNC3(ml_vfmsnf8, ml_float8_t, float, 8, fmaf, -, +, +)

DEF_ML_VECTOR_PRIMITIVES_FUNC3(ml_vfmsnd2, ml_double2_t, double, 2, fma, -, +, +)
DEF_ML_VECTOR_PRIMITIVES_FUNC3(ml_vfmsnd4, ml_double4_t, double, 4, fma, -, +, +)
DEF_ML_VECTOR_PRIMITIVES_FUNC3(ml_vfmsnd8, ml_double8_t, double, 8, fma, -, +, +)

/** Vector Negate */
DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnegf2, ml_float2_t, float, 2, -)
DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnegf4, ml_float4_t, float, 4, -)
DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnegf8, ml_float8_t, float, 8, -)

DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnegd2, ml_double2_t, double, 2, -)
DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnegd4, ml_double4_t, double, 4, -)
DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnegd8, ml_double8_t, double, 8, -)

DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnegi2, ml_int2_t, int32_t, 2, -)
DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnegi4, ml_int4_t, int32_t, 4, -)
DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnegi8, ml_int8_t, int32_t, 8, -)

DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnegu2, ml_uint2_t, uint32_t, 2, -)
DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnegu4, ml_uint4_t, uint32_t, 4, -)
DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnegu8, ml_uint8_t, uint32_t, 8, -)

DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnegl2, ml_long2_t, int64_t, 2, -)
DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnegl4, ml_long4_t, int64_t, 4, -)
DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnegl8, ml_long8_t, int64_t, 8, -)

DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnegul2, ml_ulong2_t, uint64_t, 2, -)
DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnegul4, ml_ulong4_t, uint64_t, 4, -)
DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnegul8, ml_ulong8_t, uint64_t, 8, -)

/** Vector logical negation */
DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnoti2, ml_int2_t, int32_t, 2, !)
DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnoti4, ml_int4_t, int32_t, 4, !)
DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnoti8, ml_int8_t, int32_t, 8, !)

DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnotu2, ml_uint2_t, uint32_t, 2, !)
DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnotu4, ml_uint4_t, uint32_t, 4, !)
DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnotu8, ml_uint8_t, uint32_t, 8, !)

DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnotb2, ml_bool2_t, uint32_t, 2, !)
DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnotb4, ml_bool4_t, uint32_t, 4, !)
DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnotb8, ml_bool8_t, uint32_t, 8, !)


/** Vector logical and */
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vandi2, ml_int2_t, int32_t, 2, &&)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vandi4, ml_int4_t, int32_t, 4, &&)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vandi8, ml_int8_t, int32_t, 8, &&)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vandu2, ml_uint2_t, int32_t, 2, &&)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vandu4, ml_uint4_t, int32_t, 4, &&)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vandu8, ml_uint8_t, int32_t, 8, &&)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vandb2, ml_bool2_t, int32_t, 2, &&)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vandb4, ml_bool4_t, int32_t, 4, &&)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vandb8, ml_bool8_t, int32_t, 8, &&)

/** Vector logical or */
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vori2, ml_int2_t, int32_t, 2, ||)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vori4, ml_int4_t, int32_t, 4, ||)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vori8, ml_int8_t, int32_t, 8, ||)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_voru2, ml_uint2_t, int32_t, 2, ||)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_voru4, ml_uint4_t, int32_t, 4, ||)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_voru8, ml_uint8_t, int32_t, 8, ||)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vorb2, ml_bool2_t, int32_t, 2, ||)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vorb4, ml_bool4_t, int32_t, 4, ||)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vorb8, ml_bool8_t, int32_t, 8, ||)

/** Vector bitwise and */
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vbwandi2, ml_int2_t, int32_t, 2, &)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vbwandi4, ml_int4_t, int32_t, 4, &)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vbwandi8, ml_int8_t, int32_t, 8, &)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vbwandu2, ml_uint2_t, int32_t, 2, &)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vbwandu4, ml_uint4_t, int32_t, 4, &)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vbwandu8, ml_uint8_t, int32_t, 8, &)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vbwandl2, ml_long2_t, int64_t, 2, &)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vbwandl4, ml_long4_t, int64_t, 4, &)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vbwandl8, ml_long8_t, int64_t, 8, &)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vbwandul2, ml_ulong2_t, uint64_t, 2, &)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vbwandul4, ml_ulong4_t, uint64_t, 4, &)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vbwandul8, ml_ulong8_t, uint64_t, 8, &)

/** Vector bitwise or */
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vbwori2, ml_int2_t, int32_t, 2, |)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vbwori4, ml_int4_t, int32_t, 4, |)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vbwori8, ml_int8_t, int32_t, 8, |)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vbworu2, ml_uint2_t, int32_t, 2,|)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vbworu4, ml_uint4_t, int32_t, 4,|)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vbworu8, ml_uint8_t, int32_t, 8,|)


DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vbworl2, ml_long2_t, int64_t, 2, |)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vbworl4, ml_long4_t, int64_t, 4, |)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vbworl8, ml_long8_t, int64_t, 8, |)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vbworul2, ml_ulong2_t, uint64_t, 2, |)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vbworul4, ml_ulong4_t, uint64_t, 4, |)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vbworul8, ml_ulong8_t, uint64_t, 8, |)
/** Vector bitwise not */
DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vbwnoti2, ml_int2_t, int32_t, 2, ~)
DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vbwnoti4, ml_int4_t, int32_t, 4, ~)
DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vbwnoti8, ml_int8_t, int32_t, 8, ~)

DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vbwnotu2, ml_uint2_t, int32_t, 2,~)
DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vbwnotu4, ml_uint4_t, int32_t, 4,~)
DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vbwnotu8, ml_uint8_t, int32_t, 8,~)

/** Comparison operations */
#define DEF_ML_VECTOR_COMPARATOR_OP2(FUNC_NAME, RESULT_FORMAT, VECTOR_FORMAT, VECTOR_SIZE, COMP_OP) \
static inline void FUNC_NAME(RESULT_FORMAT *r, VECTOR_FORMAT vop0, VECTOR_FORMAT vop1) {\
  unsigned i;\
  for (i = 0; i < VECTOR_SIZE; ++i) {\
    (*(r))[i] = vop0[i] COMP_OP vop1[i];\
  };\
}

/** 2-element vector comparison */
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_gt_f2, ml_bool2_t, ml_float2_t, 2, >)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ge_f2, ml_bool2_t, ml_float2_t, 2, >=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_lt_f2, ml_bool2_t, ml_float2_t, 2, <)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_le_f2, ml_bool2_t, ml_float2_t, 2, <=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_eq_f2, ml_bool2_t, ml_float2_t, 2, ==)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ne_f2, ml_bool2_t, ml_float2_t, 2, !=)

DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_gt_d2, ml_bool2_t, ml_double2_t, 2, >)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ge_d2, ml_bool2_t, ml_double2_t, 2, >=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_lt_d2, ml_bool2_t, ml_double2_t, 2, <)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_le_d2, ml_bool2_t, ml_double2_t, 2, <=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_eq_d2, ml_bool2_t, ml_double2_t, 2, ==)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ne_d2, ml_bool2_t, ml_double2_t, 2, !=)

DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_gt_i2, ml_bool2_t, ml_int2_t, 2, >)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ge_i2, ml_bool2_t, ml_int2_t, 2, >=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_lt_i2, ml_bool2_t, ml_int2_t, 2, <)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_le_i2, ml_bool2_t, ml_int2_t, 2, <=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_eq_i2, ml_bool2_t, ml_int2_t, 2, ==)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ne_i2, ml_bool2_t, ml_int2_t, 2, !=)

DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_gt_u2, ml_bool2_t, ml_uint2_t, 2, >)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ge_u2, ml_bool2_t, ml_uint2_t, 2, >=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_lt_u2, ml_bool2_t, ml_uint2_t, 2, <)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_le_u2, ml_bool2_t, ml_uint2_t, 2, <=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_eq_u2, ml_bool2_t, ml_uint2_t, 2, ==)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ne_u2, ml_bool2_t, ml_uint2_t, 2, !=)

/** 4-element vector comparison */
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_gt_f4, ml_bool4_t, ml_float4_t, 4, >)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ge_f4, ml_bool4_t, ml_float4_t, 4, >=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_lt_f4, ml_bool4_t, ml_float4_t, 4, <)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_le_f4, ml_bool4_t, ml_float4_t, 4, <=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_eq_f4, ml_bool4_t, ml_float4_t, 4, ==)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ne_f4, ml_bool4_t, ml_float4_t, 4, !=)

DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_gt_d4, ml_bool4_t, ml_double4_t, 4, >)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ge_d4, ml_bool4_t, ml_double4_t, 4, >=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_lt_d4, ml_bool4_t, ml_double4_t, 4, <)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_le_d4, ml_bool4_t, ml_double4_t, 4, <=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_eq_d4, ml_bool4_t, ml_double4_t, 4, ==)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ne_d4, ml_bool4_t, ml_double4_t, 4, !=)

DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_gt_i4, ml_bool4_t, ml_int4_t, 4, >)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ge_i4, ml_bool4_t, ml_int4_t, 4, >=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_lt_i4, ml_bool4_t, ml_int4_t, 4, <)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_le_i4, ml_bool4_t, ml_int4_t, 4, <=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_eq_i4, ml_bool4_t, ml_int4_t, 4, ==)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ne_i4, ml_bool4_t, ml_int4_t, 4, !=)

DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_gt_u4, ml_bool4_t, ml_uint4_t, 4, >)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ge_u4, ml_bool4_t, ml_uint4_t, 4, >=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_lt_u4, ml_bool4_t, ml_uint4_t, 4, <)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_le_u4, ml_bool4_t, ml_uint4_t, 4, <=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_eq_u4, ml_bool4_t, ml_uint4_t, 4, ==)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ne_u4, ml_bool4_t, ml_uint4_t, 4, !=)

DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_gt_l4, ml_bool4_t, ml_long4_t, 4, >)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ge_l4, ml_bool4_t, ml_long4_t, 4, >=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_lt_l4, ml_bool4_t, ml_long4_t, 4, <)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_le_l4, ml_bool4_t, ml_long4_t, 4, <=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_eq_l4, ml_bool4_t, ml_long4_t, 4, ==)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ne_l4, ml_bool4_t, ml_long4_t, 4, !=)

DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_gt_ul4, ml_bool4_t, ml_ulong4_t, 4, >)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ge_ul4, ml_bool4_t, ml_ulong4_t, 4, >=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_lt_ul4, ml_bool4_t, ml_ulong4_t, 4, <)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_le_ul4, ml_bool4_t, ml_ulong4_t, 4, <=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_eq_ul4, ml_bool4_t, ml_ulong4_t, 4, ==)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ne_ul4, ml_bool4_t, ml_ulong4_t, 4, !=)

/** 8-element vector comparison */
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_gt_f8, ml_bool8_t, ml_float8_t, 8, >)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ge_f8, ml_bool8_t, ml_float8_t, 8, >=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_lt_f8, ml_bool8_t, ml_float8_t, 8, <)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_le_f8, ml_bool8_t, ml_float8_t, 8, <=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_eq_f8, ml_bool8_t, ml_float8_t, 8, ==)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ne_f8, ml_bool8_t, ml_float8_t, 8, !=)

DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_gt_d8, ml_bool8_t, ml_double8_t, 8, >)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ge_d8, ml_bool8_t, ml_double8_t, 8, >=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_lt_d8, ml_bool8_t, ml_double8_t, 8, <)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_le_d8, ml_bool8_t, ml_double8_t, 8, <=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_eq_d8, ml_bool8_t, ml_double8_t, 8, ==)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ne_d8, ml_bool8_t, ml_double8_t, 8, !=)

DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_gt_i8, ml_bool8_t, ml_int8_t, 8, >)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ge_i8, ml_bool8_t, ml_int8_t, 8, >=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_lt_i8, ml_bool8_t, ml_int8_t, 8, <)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_le_i8, ml_bool8_t, ml_int8_t, 8, <=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_eq_i8, ml_bool8_t, ml_int8_t, 8, ==)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ne_i8, ml_bool8_t, ml_int8_t, 8, !=)

DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_gt_u8, ml_bool8_t, ml_uint8_t, 8, >)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ge_u8, ml_bool8_t, ml_uint8_t, 8, >=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_lt_u8, ml_bool8_t, ml_uint8_t, 8, <)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_le_u8, ml_bool8_t, ml_uint8_t, 8, <=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_eq_u8, ml_bool8_t, ml_uint8_t, 8, ==)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ne_u8, ml_bool8_t, ml_uint8_t, 8, !=)

DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_gt_l8, ml_bool8_t, ml_long8_t, 8, >)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ge_l8, ml_bool8_t, ml_long8_t, 8, >=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_lt_l8, ml_bool8_t, ml_long8_t, 8, <)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_le_l8, ml_bool8_t, ml_long8_t, 8, <=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_eq_l8, ml_bool8_t, ml_long8_t, 8, ==)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ne_l8, ml_bool8_t, ml_long8_t, 8, !=)

DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_gt_ul8, ml_bool8_t, ml_ulong8_t, 8, >)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ge_ul8, ml_bool8_t, ml_ulong8_t, 8, >=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_lt_ul8, ml_bool8_t, ml_ulong8_t, 8, <)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_le_ul8, ml_bool8_t, ml_ulong8_t, 8, <=)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_eq_ul8, ml_bool8_t, ml_ulong8_t, 8, ==)
DEF_ML_VECTOR_COMPARATOR_OP2(ml_comp_ne_ul8, ml_bool8_t, ml_ulong8_t, 8, !=)


/** Specific tests, new to convert boolean value to 0/-1 */
#define DEF_ML_VECTOR_TEST_FUNC_OP1(FUNC_NAME, RESULT_FORMAT, VECTOR_FORMAT, VECTOR_SIZE, SCALAR_TEST_FUNC) \
static inline void FUNC_NAME(RESULT_FORMAT *r, VECTOR_FORMAT vop) {\
  unsigned i;\
  for (i = 0; i < VECTOR_SIZE; ++i) {\
    (*(r))[i] = SCALAR_TEST_FUNC(vop[i]) ? -1 : 0;\
  };\
}

DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestf2_is_nan_or_inf, ml_bool2_t, ml_float2_t, 2, ml_is_nan_or_inff)
DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestf4_is_nan_or_inf, ml_bool4_t, ml_float4_t, 4, ml_is_nan_or_inff)
DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestf8_is_nan_or_inf, ml_bool8_t, ml_float8_t, 8, ml_is_nan_or_inff)

DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestd2_is_nan_or_inf, ml_lbool2_t, ml_double2_t, 2, ml_is_nan_or_inf)
DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestd4_is_nan_or_inf, ml_lbool4_t, ml_double4_t, 4, ml_is_nan_or_inf)
DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestd8_is_nan_or_inf, ml_lbool8_t, ml_double8_t, 8, ml_is_nan_or_inf)

DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestf2_is_nan, ml_bool2_t, ml_float2_t, 2, ml_is_nanf)
DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestf4_is_nan, ml_bool4_t, ml_float4_t, 4, ml_is_nanf)
DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestf8_is_nan, ml_bool8_t, ml_float8_t, 8, ml_is_nanf)

DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestd2_is_nan, ml_lbool2_t, ml_double2_t, 2, ml_is_nan)
DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestd4_is_nan, ml_lbool4_t, ml_double4_t, 4, ml_is_nan)
DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestd8_is_nan, ml_lbool8_t, ml_double8_t, 8, ml_is_nan)

DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestf2_is_inf, ml_bool2_t, ml_float2_t, 2, ml_is_inff)
DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestf4_is_inf, ml_bool4_t, ml_float4_t, 4, ml_is_inff)
DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestf8_is_inf, ml_bool8_t, ml_float8_t, 8, ml_is_inff)

DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestd2_is_inf, ml_lbool2_t, ml_double2_t, 2, ml_is_inf)
DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestd4_is_inf, ml_lbool4_t, ml_double4_t, 4, ml_is_inf)
DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestd8_is_inf, ml_lbool8_t, ml_double8_t, 8, ml_is_inf)

DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestf2_is_zero, ml_bool2_t, ml_float2_t, 2, ml_is_zerof)
DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestf4_is_zero, ml_bool4_t, ml_float4_t, 4, ml_is_zerof)
DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestf8_is_zero, ml_bool8_t, ml_float8_t, 8, ml_is_zerof)

DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestd2_is_zero, ml_lbool2_t, ml_double2_t, 2, ml_is_zero)
DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestd4_is_zero, ml_lbool4_t, ml_double4_t, 4, ml_is_zero)
DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestd8_is_zero, ml_lbool8_t, ml_double8_t, 8, ml_is_zero)

DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestf2_is_subnormal, ml_bool2_t, ml_float2_t, 2, ml_is_subnormalf)
DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestf4_is_subnormal, ml_bool4_t, ml_float4_t, 4, ml_is_subnormalf)
DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestf8_is_subnormal, ml_bool8_t, ml_float8_t, 8, ml_is_subnormalf)

DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestd2_is_subnormal, ml_lbool2_t, ml_double2_t, 2, ml_is_subnormal)
DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestd4_is_subnormal, ml_lbool4_t, ml_double4_t, 4, ml_is_subnormal)
DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestd8_is_subnormal, ml_lbool8_t, ml_double8_t, 8, ml_is_subnormal)

static inline int ml_is_vmask2_zero(ml_bool2_t vop) {
    return (vop[0] == 0) && (vop[1] == 0);
}
static inline int ml_is_vmask4_zero(ml_bool4_t vop) {
    return (vop[0] == 0) &&
         (vop[1] == 0) &&
         (vop[2] == 0) &&
         (vop[3] == 0);
}
static inline int ml_is_vmask8_zero(ml_bool8_t vop) {
    return (vop[0] == 0) &&
         (vop[1] == 0) &&
         (vop[2] == 0) &&
         (vop[3] == 0) &&
         (vop[4] == 0) &&
         (vop[5] == 0) &&
         (vop[6] == 0) &&
         (vop[7] == 0);
}

static inline int ml_is_vmask2_any_zero(ml_bool2_t vop) {
    return (vop[0] == 0) || (vop[1] == 0);
}
static inline int ml_is_vmask4_any_zero(ml_bool4_t vop) {
    return (vop[0] == 0) ||
         (vop[1] == 0) ||
         (vop[2] == 0) ||
         (vop[3] == 0);
}
static inline int ml_is_vmask8_any_zero(ml_bool8_t vop) {
    return (vop[0] == 0) ||
         (vop[1] == 0) ||
         (vop[2] == 0) ||
         (vop[3] == 0) ||
         (vop[4] == 0) ||
         (vop[5] == 0) ||
         (vop[6] == 0) ||
         (vop[7] == 0);
}

static inline int ml_is_vmask2_not_any_zero(ml_bool2_t vop) {
    return (vop[0] != 0) && (vop[1] != 0);
}
static inline int ml_is_vmask4_not_any_zero(ml_bool4_t vop) {
    return (vop[0] != 0) &&
         (vop[1] != 0) &&
         (vop[2] != 0) &&
         (vop[3] != 0);
}
static inline int ml_is_vmask8_not_any_zero(ml_bool8_t vop) {
    return (vop[0] != 0) &&
         (vop[1] != 0) &&
         (vop[2] != 0) &&
         (vop[3] != 0) &&
         (vop[4] != 0) &&
         (vop[5] != 0) &&
         (vop[6] != 0) &&
         (vop[7] != 0);
}

static inline int ml_is_vmask2_not_all_zero(ml_bool2_t vop) {
    return (vop[0] != 0) || (vop[1] != 0);
}
static inline int ml_is_vmask4_not_all_zero(ml_bool4_t vop) {
    return (vop[0] != 0) ||
         (vop[1] != 0) ||
         (vop[2] != 0) ||
         (vop[3] != 0);
}
static inline int ml_is_vmask8_not_all_zero(ml_bool8_t vop) {
    return (vop[0] != 0) ||
         (vop[1] != 0) ||
         (vop[2] != 0) ||
         (vop[3] != 0) ||
         (vop[4] != 0) ||
         (vop[5] != 0) ||
         (vop[6] != 0) ||
         (vop[7] != 0);
}

/** Vector Assembling functions **/
#define DEF_ML_VECTOR_ASSEMBLY_FUNC_1_2(FUNC_NAME, RESULT_FORMAT, SCALAR_FORMAT) \
static inline void FUNC_NAME(RESULT_FORMAT *r, SCALAR_FORMAT op1, SCALAR_FORMAT op2) {\
    (*(r))[0] = op1; (*(r))[1] = op2;\
}

#define DEF_ML_VECTOR_ASSEMBLY_FUNC_1_4(FUNC_NAME, RESULT_FORMAT, SCALAR_FORMAT) \
static inline void FUNC_NAME(RESULT_FORMAT *(r), SCALAR_FORMAT op1, SCALAR_FORMAT op2, SCALAR_FORMAT op3, SCALAR_FORMAT op4) {\
    (*(r))[0] = op1; (*(r))[1] = op2; (*(r))[2] = op3; (*(r))[3] = op4;\
}

#define DEF_ML_VECTOR_ASSEMBLY_FUNC_1_8(FUNC_NAME, RESULT_FORMAT, SCALAR_FORMAT) \
static inline void FUNC_NAME(RESULT_FORMAT *(r), SCALAR_FORMAT op1,\
                             SCALAR_FORMAT op2, SCALAR_FORMAT op3, SCALAR_FORMAT op4,\
                             SCALAR_FORMAT op5, SCALAR_FORMAT op6, SCALAR_FORMAT op7,\
                             SCALAR_FORMAT op8\
                             ) {\
    (*(r))[0] = op1; (*(r))[1] = op2; (*(r))[2] = op3; (*(r))[3] = op4;\
    (*(r))[4] = op5; (*(r))[5] = op6; (*(r))[6] = op7; (*(r))[7] = op8;\
}

#define DEF_ML_VECTOR_ASSEMBLY_FUNC_2_4(FUNC_NAME, RESULT_FORMAT, VECTOR_FORMAT) \
static inline void FUNC_NAME(RESULT_FORMAT *(r), VECTOR_FORMAT vop1, VECTOR_FORMAT vop2) {\
  (*(r))[0] = vop1[0]; (*(r))[1] = vop1[1] ; (*(r))[2] = vop2[0] ; (*(r))[3] = vop2[1] ;\
}

#define DEF_ML_VECTOR_ASSEMBLY_FUNC_2_8(FUNC_NAME, RESULT_FORMAT, VECTOR_FORMAT) \
static inline void FUNC_NAME(RESULT_FORMAT *(r), VECTOR_FORMAT vop1, VECTOR_FORMAT vop2, VECTOR_FORMAT vop3, VECTOR_FORMAT vop4) {\
  (*(r))[0] = vop1[0]; (*(r))[1] = vop1[1] ; (*(r))[2] = vop2[0] ; (*(r))[3] = vop2[1] ;\
  (*(r))[4] = vop3[0]; (*(r))[5] = vop3[1] ; (*(r))[6] = vop4[0] ; (*(r))[7] = vop4[1] ;\
}

#define DEF_ML_VECTOR_ASSEMBLY_FUNC_4_8(FUNC_NAME, RESULT_FORMAT, VECTOR_FORMAT) \
static inline void FUNC_NAME(RESULT_FORMAT *(r), VECTOR_FORMAT vop1, VECTOR_FORMAT vop2) {\
  (*(r))[0] = vop1[0]; (*(r))[1] = vop1[1] ; (*(r))[2] = vop1[2] ; (*(r))[3] = vop1[3] ;\
  (*(r))[4] = vop2[0]; (*(r))[5] = vop2[1] ; (*(r))[6] = vop2[2] ; (*(r))[7] = vop2[3] ;\
}

DEF_ML_VECTOR_ASSEMBLY_FUNC_1_2(ml_vec_assembling_1_2_float, ml_float2_t, float)
DEF_ML_VECTOR_ASSEMBLY_FUNC_1_2(ml_vec_assembling_1_2_int, ml_int2_t, int32_t)
DEF_ML_VECTOR_ASSEMBLY_FUNC_1_2(ml_vec_assembling_1_2_bool, ml_bool2_t, bool)
DEF_ML_VECTOR_ASSEMBLY_FUNC_1_2(ml_vec_assembling_1_2_double, ml_double2_t, double)

DEF_ML_VECTOR_ASSEMBLY_FUNC_1_4(ml_vec_assembling_1_4_float, ml_float4_t, float)
DEF_ML_VECTOR_ASSEMBLY_FUNC_1_4(ml_vec_assembling_1_4_int, ml_int4_t, int32_t)
DEF_ML_VECTOR_ASSEMBLY_FUNC_1_4(ml_vec_assembling_1_4_bool, ml_bool4_t, bool)
DEF_ML_VECTOR_ASSEMBLY_FUNC_1_4(ml_vec_assembling_1_4_double, ml_double4_t, double)

DEF_ML_VECTOR_ASSEMBLY_FUNC_1_8(ml_vec_assembling_1_8_float, ml_float8_t, float)
DEF_ML_VECTOR_ASSEMBLY_FUNC_1_8(ml_vec_assembling_1_8_double, ml_double8_t, double)
DEF_ML_VECTOR_ASSEMBLY_FUNC_1_8(ml_vec_assembling_1_8_bool, ml_bool8_t, bool)

DEF_ML_VECTOR_ASSEMBLY_FUNC_2_4(ml_vec_assembling_2_4_float, ml_float4_t, ml_float2_t)
DEF_ML_VECTOR_ASSEMBLY_FUNC_2_4(ml_vec_assembling_2_4_int, ml_int4_t, ml_int2_t)
DEF_ML_VECTOR_ASSEMBLY_FUNC_2_4(ml_vec_assembling_2_4_bool, ml_bool4_t, ml_bool2_t)

DEF_ML_VECTOR_ASSEMBLY_FUNC_2_8(ml_vec_assembling_2_8_float, ml_float8_t, ml_float2_t)
DEF_ML_VECTOR_ASSEMBLY_FUNC_2_8(ml_vec_assembling_2_8_int, ml_int8_t, ml_int2_t)
DEF_ML_VECTOR_ASSEMBLY_FUNC_2_8(ml_vec_assembling_2_8_bool, ml_bool8_t, ml_bool2_t)

DEF_ML_VECTOR_ASSEMBLY_FUNC_4_8(ml_vec_assembling_4_8_float, ml_float8_t, ml_float4_t)
DEF_ML_VECTOR_ASSEMBLY_FUNC_4_8(ml_vec_assembling_4_8_int, ml_int8_t, ml_int4_t)
DEF_ML_VECTOR_ASSEMBLY_FUNC_4_8(ml_vec_assembling_4_8_bool, ml_bool8_t, ml_bool4_t)

DEF_ML_VECTOR_ASSEMBLY_FUNC_4_8(ml_vec_assembling_4_8_double, ml_double8_t, ml_double4_t)
DEF_ML_VECTOR_ASSEMBLY_FUNC_4_8(ml_vec_assembling_4_8_lbool, ml_lbool8_t, ml_lbool4_t)

/** Single Argument function with non-uniform formats */
#define DEF_ML_VECTOR_NONUN_FUNC_OP1(FUNC_NAME, RESULT_FORMAT, VECTOR_FORMAT, VECTOR_SIZE, SCALAR_TEST_FUNC) \
static inline void FUNC_NAME(RESULT_FORMAT *r, VECTOR_FORMAT vop) {\
  unsigned i;\
  for (i = 0; i < VECTOR_SIZE; ++i) {\
    (*(r))[i] = SCALAR_TEST_FUNC(vop[i]);\
  };\
}

DEF_ML_VECTOR_NONUN_FUNC_OP1(ml_vnearbyintd2, ml_long2_t, ml_double2_t, 2, nearbyint)
DEF_ML_VECTOR_NONUN_FUNC_OP1(ml_vnearbyintd4, ml_long4_t, ml_double4_t, 4, nearbyint)
DEF_ML_VECTOR_NONUN_FUNC_OP1(ml_vnearbyintd8, ml_long8_t, ml_double8_t, 8, nearbyint)

DEF_ML_VECTOR_NONUN_FUNC_OP1(ml_vnearbyintf2, ml_int2_t, ml_float2_t, 2, nearbyintf)
DEF_ML_VECTOR_NONUN_FUNC_OP1(ml_vnearbyintf4, ml_int4_t, ml_float4_t, 4, nearbyintf)
DEF_ML_VECTOR_NONUN_FUNC_OP1(ml_vnearbyintf8, ml_int8_t, ml_float8_t, 8, nearbyintf)

DEF_ML_VECTOR_NONUN_FUNC_OP1(ml_vrintf2, ml_float2_t, ml_float2_t, 2, rintf)
DEF_ML_VECTOR_NONUN_FUNC_OP1(ml_vrintf4, ml_float4_t, ml_float4_t, 4, rintf)
DEF_ML_VECTOR_NONUN_FUNC_OP1(ml_vrintf8, ml_float8_t, ml_float8_t, 8, rintf)


DEF_ML_VECTOR_NONUN_FUNC_OP1(ml_vrintd2, ml_double2_t, ml_double2_t, 2, rint)
DEF_ML_VECTOR_NONUN_FUNC_OP1(ml_vrintd4, ml_double4_t, ml_double4_t, 4, rint)
DEF_ML_VECTOR_NONUN_FUNC_OP1(ml_vrintd8, ml_double8_t, ml_double8_t, 8, rint)

/** Exponent insertion */
DEF_ML_VECTOR_NONUN_FUNC_OP1(ml_vexp_insertion_f2, ml_float2_t, ml_int2_t, 2, ml_exp_insertion_fp32)
DEF_ML_VECTOR_NONUN_FUNC_OP1(ml_vexp_insertion_f4, ml_float4_t, ml_int4_t, 4, ml_exp_insertion_fp32)
DEF_ML_VECTOR_NONUN_FUNC_OP1(ml_vexp_insertion_f8, ml_float8_t, ml_int8_t, 8, ml_exp_insertion_fp32)

DEF_ML_VECTOR_NONUN_FUNC_OP1(ml_vexp_insertion_d2, ml_double2_t, ml_long2_t, 2, ml_exp_insertion_fp64)
DEF_ML_VECTOR_NONUN_FUNC_OP1(ml_vexp_insertion_d4, ml_double4_t, ml_long4_t, 4, ml_exp_insertion_fp64)
DEF_ML_VECTOR_NONUN_FUNC_OP1(ml_vexp_insertion_d8, ml_double8_t, ml_long8_t, 8, ml_exp_insertion_fp64)

/** Exponent extraction */
DEF_ML_VECTOR_NONUN_FUNC_OP1(ml_vexp_extraction_f2, ml_int2_t, ml_float2_t, 2, ml_exp_extraction_dirty_fp32)
DEF_ML_VECTOR_NONUN_FUNC_OP1(ml_vexp_extraction_f4, ml_int4_t, ml_float4_t, 4, ml_exp_extraction_dirty_fp32)
DEF_ML_VECTOR_NONUN_FUNC_OP1(ml_vexp_extraction_f8, ml_int8_t, ml_float8_t, 8, ml_exp_extraction_dirty_fp32)

DEF_ML_VECTOR_NONUN_FUNC_OP1(ml_vexp_extraction_d2, ml_long2_t, ml_double2_t, 2, ml_exp_extraction_dirty_fp64)
DEF_ML_VECTOR_NONUN_FUNC_OP1(ml_vexp_extraction_d4, ml_long4_t, ml_double4_t, 4, ml_exp_extraction_dirty_fp64)
DEF_ML_VECTOR_NONUN_FUNC_OP1(ml_vexp_extraction_d8, ml_long8_t, ml_double8_t, 8, ml_exp_extraction_dirty_fp64)

/** Mantissa extraction */
DEF_ML_VECTOR_NONUN_FUNC_OP1(ml_vmantissa_extraction_f2, ml_float2_t, ml_float2_t, 2, ml_mantissa_extraction_fp32)
DEF_ML_VECTOR_NONUN_FUNC_OP1(ml_vmantissa_extraction_f4, ml_float4_t, ml_float4_t, 4, ml_mantissa_extraction_fp32)
DEF_ML_VECTOR_NONUN_FUNC_OP1(ml_vmantissa_extraction_f8, ml_float8_t, ml_float8_t, 8, ml_mantissa_extraction_fp32)

DEF_ML_VECTOR_NONUN_FUNC_OP1(ml_vmantissa_extraction_d2, ml_double2_t, ml_double2_t, 2, ml_mantissa_extraction_fp64)
DEF_ML_VECTOR_NONUN_FUNC_OP1(ml_vmantissa_extraction_d4, ml_double4_t, ml_double4_t, 4, ml_mantissa_extraction_fp64)
DEF_ML_VECTOR_NONUN_FUNC_OP1(ml_vmantissa_extraction_d8, ml_double8_t, ml_double8_t, 8, ml_mantissa_extraction_fp64)


/** Vector element-wise selection */
#define ML_VSELECT(result,test,op0,op1,size) {\
  unsigned __k; for (__k = 0; __k < size; ++__k) (*(result))[__k] = (test)[__k] ? (op0)[__k] : (op1)[__k]; };

/** Vector element-wise load (gather) */
#define ML_VLOAD(result,table,addr,size) {\
  unsigned __k; for (__k = 0; __k < size; ++__k) (*(result))[__k] = table[(addr)[__k]]; };
/** Vector element-wise load (gather) for 2D table */
#define ML_VLOAD2D(result,table,addr0,addr1,size) {\
  unsigned __k; for (__k = 0; __k < size; ++__k) (*(result))[__k] = table[(addr0)[__k]][(addr1)[__k]]; };
/** Implicit vector conversion */
#define ML_VCONV(dst,src,size) {\
  unsigned __k; for (__k = 0; __k < size; ++__k) (*(dst))[__k] = (src)[__k]; };

#endif // define __ML_SUPPORT_LIB_H__
