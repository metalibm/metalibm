/*******************************************************************************
* This file is part of Kalray's Metalibm tool
* Copyright (2013)
* All rights reserved
* created:          Dec 23rd, 2013
* last-modified:    Dec 23rd, 2013
*
* author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
*******************************************************************************/
#include "ml_types.h"

#ifndef __ML_SPECIAL_VALUES_H__
#define __ML_SPECIAL_VALUES_H__



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


#endif /** __ML_SPECIAL_VALUES_H__ */
