/*******************************************************************************
* This file is part of Kalray's Metalibm tool
* Copyright (2016-2020)
* All rights reserved
* created:          Feb  2nd, 2016
* last-modified:    Jan 23rd, 2020
*
* author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
*******************************************************************************/
#ifndef __ML_VECTOR_FORMAT_H__
#define __ML_VECTOR_FORMAT_H__

#include <stdint.h>

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

#endif /** ifdef __ML_VECTOR_FORMAT_H__ */
