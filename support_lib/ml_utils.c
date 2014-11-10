#include <support_lib/ml_utils.h>
#include <support_lib/ml_types.h>

/*uint32_t float_to_32b_encoding(float v) {
    uif_conv_t conv_tmp;
    conv_tmp.f = v;
    return conv_tmp.u;
}


float float_from_32b_encoding(uint32_t v) {
    uif_conv_t conv_tmp;
    conv_tmp.u = v;
    return conv_tmp.f;
}*/

/*int ml_isnanf(float v) {
    uif_conv_t conv_tmp;
    conv_tmp.f = v;
    int exp1     = (conv_tmp.u & 0x7f800000u) == 0x7f800000u;
    int not0mant = (conv_tmp.u & 0x007fffffu) != 0;
    return (exp1 && not0mant);
}*/
