#include <support_lib/ml_special_values.h>
#include <support_lib/ml_utils.h>
#include <support_lib/k1/k1b_utils.h>
#include <support_lib/ml_multi_prec_lib.h>
#include <HAL/hal/hal.h>
#include <cpu.h>
#include <inttypes.h>
#include <stdio.h>

ml_dd_t k1b_ml_split_dd_d(double x) {
    /* Veltkamp split appied to p = 53, C = (1 << 27) + 1 */
    const uid_conv_t C = {.u = 0x41a0000002000000ull};
    double gamma;
    // double gamma = C.d * x;
    // to bypass K1B fma
    asm("fmuld %0 = %1, %2" : "=r"(gamma) : "r"(C.d), "r"(x) :);
    double delta = x - gamma;

    double xhi = gamma + delta;

    ml_dd_t result = { .hi = xhi, .lo = x - xhi};

    return result;
}
