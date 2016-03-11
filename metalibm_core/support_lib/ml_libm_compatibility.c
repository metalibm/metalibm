#include <math.h>
#include <errno.h>


float ml_raise_libm_underflowf(float result, float arg, char* func_name) {
    float return_value = result;
    if (_LIB_VERSION == _IEEE_) {
        return return_value;
    } else if (_LIB_VERSION == _POSIX_) {
        errno = ERANGE;
        return return_value;
    } else if (_LIB_VERSION == _XOPEN_) {
        struct exception e_struct = {.type = UNDERFLOW, .name = func_name, .arg1 = arg, .retval = return_value};
        matherr(&e_struct);
        errno = ERANGE;
        return e_struct.retval;
    } else if (_LIB_VERSION == _SVID_) {
        errno = ERANGE;
        struct exception e_struct = {.type = UNDERFLOW, .name = func_name, .arg1 = arg, .retval = return_value};
        matherr(&e_struct);
        return e_struct.retval;
    } else {
        return return_value;
    }

    return return_value;
}


float ml_raise_libm_overflowf(float result, float arg, char* func_name) {
    float return_value = HUGE_VALF;
    if (_LIB_VERSION == _IEEE_) {
        return return_value;
    } else if (_LIB_VERSION == _POSIX_) {
        errno = ERANGE;
        return return_value;
    } else if (_LIB_VERSION == _XOPEN_) {
        struct exception e_struct = {.type = OVERFLOW, .name = func_name, .arg1 = arg, .retval = return_value};
        matherr(&e_struct);
        errno = ERANGE;
        return e_struct.retval;
    } else if (_LIB_VERSION == _SVID_) {
        errno = ERANGE;
        struct exception e_struct = {.type = OVERFLOW, .name = func_name, .arg1 = arg, .retval = return_value};
        matherr(&e_struct);
        return e_struct.retval;
    } else {
        return return_value;
    }

    return return_value;
}
