/*******************************************************************************
* This file is part of Kalray's Metalibm tool
* Copyright (2013)
* All rights reserved
* created:          Dec 23rd, 2013
* last-modified:    Dec 23rd, 2013
*
* author(s): Nicolas Brunie (nicolas.brunie@kalray.eu)
*******************************************************************************/
#include <stdint.h>

#ifndef __ML_TYPES_H__
#define __ML_TYPES_H__

/** boolean true value */
#ifndef ML_TRUE
#define ML_TRUE 1
#endif
/** boolean false value */
#ifndef ML_FALSE
#define ML_FALSE 0
#endif



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

#endif /** __ML_TYPES_H__ */
