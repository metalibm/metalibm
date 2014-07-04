#include "utils.hpp"
#include <support_lib/ml_utils.h>
#include <support_lib/ml_types.h>


mpz_class set_mpz_from_uint64(uint64_t v) {
    uint64_t high = v >> 32;
    uint64_t low = v & 0xffffffffull;
    uint32_t high32 = high;
    uint32_t low32 =   low;

    mpz_class result = (high << 32) + low;
    return result;
}


uint64_t set_uint64_from_mpz(mpz_class x) {
    mpz_class mask = set_mpz_from_uint64(0xffffffffull);
    mpz_class low_mpz = (x & mask);
    uint64_t low = low_mpz.get_ui();
    mpz_class high_mpz = (x >> 32);
    uint64_t high = high_mpz.get_ui();

    return (high << 32) | low;
}


double mpz_to_double(mpz_class v) {
    uid_conv_t result;
    result.u = set_uint64_from_mpz(v);

    return result.d;
}


float mpz_to_float(mpz_class v) {
    uif_conv_t result;
    result.u = v.get_ui();

    return result.f;
}


mpz_class set_mpz_from_double_encoding(double v) {
    uid_conv_t conv_tmp;
    conv_tmp.d = v;

    return set_mpz_from_uint64(conv_tmp.u);
}

void clear_mpfr_exception(void) {
    mpfr_clear_underflow();
    mpfr_clear_overflow();
    mpfr_clear_nanflag();
    mpfr_clear_inexflag();
    mpfr_clear_erangeflag();
    mpfr_clear_divby0();
}

bool is_hidden_underflowf(float value, int ternary) {
   uif_conv_t tmp;
   tmp.f = value;
   const uint32_t lowest_normal_p = 0x00800000, lowest_normal_n = 0x80800000;
   if (tmp.u == lowest_normal_p && ternary > 0) return true;
   else if (tmp.u == lowest_normal_n && ternary < 0) return true;
   else return false;
}



void set_mpfr_binary32_env() {
    mpfr_set_default_prec(24);
    mpfr_set_emin(-(1 << 7) +3 - 23);
    mpfr_set_emax(1 << 7);
}

std::string hex_value32(uint32_t value) {
    std::stringstream s;
    s << std::hex << "0x" << value << "u";
    return s.str();
}


void set_mpfr_binary64_env() {
    mpfr_set_default_prec(53);
    mpfr_set_emin((-(1 << 10)) + 3 - 52);
    mpfr_set_emax(1 << 10);
}

std::string hex_value64(uint64_t value) {
    std::stringstream s;
    s << std::hex << "0x" << value << "ull";
    return s.str();
}
