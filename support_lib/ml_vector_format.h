/*******************************************************************************
* This file is part of Kalray's Metalibm tool
* Copyright (2016)
* All rights reserved
* created:          Feb  2nd, 2016
* last-modified:    Feb  2nd, 2016
*
* author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
*******************************************************************************/
#include <stdint.h>

#ifndef __ML_VECTOR_FORMAT_H__
#define __ML_VECTOR_FORMAT_H__

#define DEC_ML_FORMAT(FORMAT_NAME, FIELD_FORMAT, SIZE) \
typedef struct {\
  FIELD_FORMAT _[SIZE];\
} FORMAT_NAME;

// single precision vector format
DEC_ML_FORMAT(ml_float2_t, float, 2)
DEC_ML_FORMAT(ml_float4_t, float, 4)
DEC_ML_FORMAT(ml_float8_t, float, 8)

// double precision vector format
DEC_ML_FORMAT(ml_double2_t, double, 2)
DEC_ML_FORMAT(ml_double4_t, double, 4)
DEC_ML_FORMAT(ml_double8_t, double, 8)

// 32-b integer vector format
DEC_ML_FORMAT(ml_int2_t, int32_t, 2)
DEC_ML_FORMAT(ml_int4_t, int32_t, 4)
DEC_ML_FORMAT(ml_int8_t, int32_t, 8)

// 32-b unsigned integer vector format
DEC_ML_FORMAT(ml_uint2_t, uint32_t, 2)
DEC_ML_FORMAT(ml_uint4_t, uint32_t, 4)
DEC_ML_FORMAT(ml_uint8_t, uint32_t, 8)

// 32-b integer vector format
DEC_ML_FORMAT(ml_long2_t, int64_t, 2)
DEC_ML_FORMAT(ml_long4_t, int64_t, 4)
DEC_ML_FORMAT(ml_long8_t, int64_t, 8)

// 32-b unsigned integer vector format
DEC_ML_FORMAT(ml_ulong2_t, uint64_t, 2)
DEC_ML_FORMAT(ml_ulong4_t, uint64_t, 4)
DEC_ML_FORMAT(ml_ulong8_t, uint64_t, 8)

// boolean vector formats
DEC_ML_FORMAT(ml_bool2_t, int, 2)
DEC_ML_FORMAT(ml_bool4_t, int, 4)
DEC_ML_FORMAT(ml_bool8_t, int, 8)

#endif /** ifdef __ML_VECTOR_FORMAT_H__ */
