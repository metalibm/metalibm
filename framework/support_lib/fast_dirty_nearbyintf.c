#include <support_lib/ml_types.h>


float fast_nearbyintf(float v) {
    uif_conv_t x = {.f = v};
    uint32_t exp = (x.u >> 23) & 0xff;
    uint32_t sign = x.u & 0x80000000u;
    uint32_t round_cst = 1 << (22 - (exp - 127));
    uint32_t mask = 0xffffffffu << (22 - (exp - 127) + 1);
    mask |= 0xff800000;
    uint32_t pre_result = exp == 126 ? (sign | 0x3f800000u) : 0x0;

    uif_conv_t round_x = {.u = exp >= 127 ? (x.u + round_cst) & mask : pre_result};

    return round_x.f;
}
