	.file	"fast_dirty_nearbyintf.c"
	.text

	.align 8
	.globl fast_nearbyintf
	.type	fast_nearbyintf, @function
fast_nearbyintf:
	extfz $r1 = $r0, 23+8-1, 23
	make $r4 = -1
	make $r3 = 1
    and $r9 = $r0, 31, 31 #/* input sign */
	;;
	sbf $r2 = $r1, 150
	sbf $r5 = $r1, 149
    comp.ge $r11 = $r1, 127  #/* exp >= 127 */
    or $r9 = $r9, 29, 23 #/* +/- 1.0f */
    ;;
	sll $r6 = $r4, $r2
	sll $r7 = $r3, $r5
    comp.eq $r32 = $r1, 126  #/* exp == 126 */
	;;
    cmove.eqz $r9 = $r32, 0
	add $r0 = $r7, $r0
	or $r8 = $r6, 31, 23
	;;
	and $r0 = $r0, $r8
    ;;
    cmove.eqz $r0 = $r11, $r9
	ret
	;;
	.size	fast_nearbyintf, .-fast_nearbyintf
	.ident	"GCC: (GNU) 4.7.4 20130620 (prerelease) [Kalray Compiler unknown 8477b72-dirty]"
