#include "ml_vector_format.h"


#define DEF_ML_VECTOR_PRIMITIVES_OP2(FUNC_NAME, VECTOR_FORMAT, SCALAR_FORMAT, VECTOR_SIZE, SCALAR_OP) \
static inline FUNC_NAME(VECTOR_FORMAT *r, VECTOR_FORMAT vop0, VECTOR_FORMAT vop1) {\
  unsigned i;\
  for (i = 0; i < VECTOR_SIZE; ++i) {\
    (*r)._[i] = vop0._[i] SCALAR_OP vop1._[i];\
  };\
}
#define DEF_ML_VECTOR_PRIMITIVES_OP1(FUNC_NAME, VECTOR_FORMAT, SCALAR_FORMAT, VECTOR_SIZE, SCALAR_OP) \
static inline FUNC_NAME(VECTOR_FORMAT *r, VECTOR_FORMAT vop) {\
  unsigned i;\
  for (i = 0; i < VECTOR_SIZE; ++i) {\
    (*r)._[i] = SCALAR_OP vop._[i];\
  };\
}


/** Vector Addition */
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vaddf2, ml_float2_t, float, 2, +)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vaddf4, ml_float4_t, float, 4, +)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vaddf8, ml_float8_t, float, 8, +)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vaddd2, ml_double2_t, double, 2, +)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vaddd4, ml_double4_t, double, 4, +)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vaddd8, ml_double8_t, double, 8, +)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vaddi2, ml_int2_t, int32_t, 2, +)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vaddi4, ml_int4_t, int32_t, 4, +)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vaddi8, ml_int8_t, int32_t, 8, +)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vaddu2, ml_uint2_t, uint32_t, 2, +)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vaddu4, ml_uint4_t, uint32_t, 4, +)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vaddu8, ml_uint8_t, uint32_t, 8, +)

/** Vector Subtraction */
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsubf2, ml_float2_t, float, 2, -)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsubf4, ml_float4_t, float, 4, -)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsubf8, ml_float8_t, float, 8, -)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsubd2, ml_double2_t, double, 2, -)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsubd4, ml_double4_t, double, 4, -)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsubd8, ml_double8_t, double, 8, -)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsubi2, ml_int2_t, int32_t, 2, -)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsubi4, ml_int4_t, int32_t, 4, -)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsubi8, ml_int8_t, int32_t, 8, -)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsubu2, ml_uint2_t, uint32_t, 2, -)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsubu4, ml_uint4_t, uint32_t, 4, -)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vsubu8, ml_uint8_t, uint32_t, 8, -)

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

/** Vector logical negation */
DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnoti2, ml_int2_t, int32_t, 2, !)
DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnoti4, ml_int4_t, int32_t, 4, !)
DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnoti8, ml_int8_t, int32_t, 8, !)

DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnotu2, ml_uint2_t, uint32_t, 2, !)
DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnotu4, ml_uint4_t, uint32_t, 4, !)
DEF_ML_VECTOR_PRIMITIVES_OP1(ml_vnotu8, ml_uint8_t, uint32_t, 8, !)

/** Vector Multiplication */
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vmulf2, ml_float2_t, float, 2, *)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vmulf4, ml_float4_t, float, 4, *)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vmulf8, ml_float8_t, float, 8, *)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vmuld2, ml_double2_t, double, 2, *)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vmuld4, ml_double4_t, double, 4, *)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vmuld8, ml_double8_t, double, 8, *)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vmuli2, ml_int2_t, int32_t, 2, *)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vmuli4, ml_int4_t, int32_t, 4, *)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vmuli8, ml_int8_t, int32_t, 8, *)

DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vmulu2, ml_uint2_t, uint32_t, 2, *)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vmulu4, ml_uint4_t, uint32_t, 4, *)
DEF_ML_VECTOR_PRIMITIVES_OP2(ml_vmulu8, ml_uint8_t, uint32_t, 8, *)


/** Comparison operations */
#define DEF_ML_VECTOR_COMPARATOR_OP2(FUNC_NAME, RESULT_FORMAT, VECTOR_FORMAT, VECTOR_SIZE, COMP_OP) \
static inline FUNC_NAME(RESULT_FORMAT *r, VECTOR_FORMAT vop0, VECTOR_FORMAT vop1) {\
  unsigned i;\
  for (i = 0; i < VECTOR_SIZE; ++i) {\
    (*r)._[i] = vop0._[i] COMP_OP vop1._[i];\
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


/** Specific tests */
#define DEF_ML_VECTOR_TEST_FUNC_OP1(FUNC_NAME, RESULT_FORMAT, VECTOR_FORMAT, VECTOR_SIZE, SCALAR_TEST_FUNC) \
static inline FUNC_NAME(RESULT_FORMAT *r, VECTOR_FORMAT vop) {\
  unsigned i;\
  for (i = 0; i < VECTOR_SIZE; ++i) {\
    (*r)._[i] = SCALAR_TEST_FUNC(vop._[i]);\
  };\
}

DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestf2_is_nan_or_inf, ml_bool2_t, ml_float2_t, 2, ml_is_nan_or_inff)
DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestf4_is_nan_or_inf, ml_bool4_t, ml_float4_t, 4, ml_is_nan_or_inff)
DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestf8_is_nan_or_inf, ml_bool8_t, ml_float8_t, 8, ml_is_nan_or_inff)

DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestd2_is_nan_or_inf, ml_bool2_t, ml_double2_t, 2, ml_is_nan_or_inf)
DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestd4_is_nan_or_inf, ml_bool4_t, ml_double4_t, 4, ml_is_nan_or_inf)
DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestd8_is_nan_or_inf, ml_bool8_t, ml_double8_t, 8, ml_is_nan_or_inf)

DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestf2_is_nan, ml_bool2_t, ml_float2_t, 2, ml_is_nanf)
DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestf4_is_nan, ml_bool4_t, ml_float4_t, 4, ml_is_nanf)
DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestf8_is_nan, ml_bool8_t, ml_float8_t, 8, ml_is_nanf)

DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestd2_is_nan, ml_bool2_t, ml_double2_t, 2, ml_is_nan)
DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestd4_is_nan, ml_bool4_t, ml_double4_t, 4, ml_is_nan)
DEF_ML_VECTOR_TEST_FUNC_OP1(ml_vtestd8_is_nan, ml_bool8_t, ml_double8_t, 8, ml_is_nan)

static inline int is_vmask2_zero(ml_bool2_t vop) {
  return (vop._[0] == 0) && (vop._[1] == 0);
}
static inline int is_vmask4_zero(ml_bool4_t vop) {
  return (vop._[0] == 0) && 
         (vop._[1] == 0) && 
         (vop._[2] == 0) && 
         (vop._[3] == 0);
}
static inline int is_vmask8_zero(ml_bool8_t vop) {
  return (vop._[0] == 0) && 
         (vop._[1] == 0) && 
         (vop._[2] == 0) && 
         (vop._[3] == 0) && 
         (vop._[4] == 0) && 
         (vop._[5] == 0) && 
         (vop._[6] == 0) && 
         (vop._[7] == 0);
}

static inline int is_vmask2_non_zero(ml_bool2_t vop) {
  return (vop._[0] != 0) || (vop._[1] != 0);
}
static inline int is_vmask4_non_zero(ml_bool4_t vop) {
  return (vop._[0] != 0) || 
         (vop._[1] != 0) || 
         (vop._[2] != 0) || 
         (vop._[3] != 0);
}
static inline int is_vmask8_non_zero(ml_bool8_t vop) {
  return (vop._[0] != 0) || 
         (vop._[1] != 0) || 
         (vop._[2] != 0) || 
         (vop._[3] != 0) || 
         (vop._[4] != 0) || 
         (vop._[5] != 0) || 
         (vop._[6] != 0) || 
         (vop._[7] != 0);
}
