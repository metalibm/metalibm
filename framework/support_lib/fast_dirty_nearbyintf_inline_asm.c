#include <support_lib/ml_types.h>


float fast_nearbyintf(float v) {
    float result;
    asm("extfz $r1 = %1, 23+8-1, 23\n\t"
    "make $r4 = -1\n\t"
    "make $r3 = 1\n\t"
    "and $r9 = %1, 31, 31 #/* input sign */\n\t"
    ";;\n\t"
    "sbf $r2 = $r1, 150\n\t"
    "sbf $r5 = $r1, 149\n\t"
    "comp.ge $r11 = $r1, 127  #/* exp >= 127 */\n\t"
    "or $r9 = $r9, 29, 23 #/* +/- 1.0f */\n\t"
    ";;\n\t"
    "sll $r6 = $r4, $r2\n\t"
    "sll $r7 = $r3, $r5\n\t"
    "comp.eq $r32 = $r1, 126  #/* exp == 126 */\n\t"
    ";;\n\t"
    "cmove.eqz $r9 = $r32, 0\n\t"
    "add %0 = $r7, %1\n\t"
    "or $r8 = $r6, 31, 23\n\t"
    ";;\n\t"
    "and %0 = %0, $r8\n\t"
    ";;\n\t"
    "cmove.eqz %0 = $r11, $r9\n\t"
    ";;\n" : "=r"(result) : "r"(v): "$r2", "$r3", "$r4", "$r5", "$r6", "$r7", "$r8", "$r9", "$r32");

    return result;
}

