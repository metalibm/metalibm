/** description:    Optimized version of payne hanek arg reduction for cosf (fp32)
 *  author:         Nicolas Brunie
 *  created:        02/13/2015 
 *  last modified:  02/13/2015
 */

/** function payne_hanek_fp32_asm
 *  double payne_hanek_fp32_asm(float)
 */
	.file	"payne_hanek_fp32_asm.S"


#define k 4
#define p 24
#define n 60
#define e 3

  .globl global_array
  .data

  .align 8
  .type global_array, @object
  .size global_array, 64
global_array:
  .long 0x7c22fa00
  .long 0xf3f92300
  .long 0xeb46f000
  .long 0x62a95000
  .long 0xd9762a00
  .long 0xcf382b00
  .long 0xc6322400
  .long 0x3c5b6300
  .long 0xb3d4ce00
  .long 0x2b710e00
  .long 0x22821000
  .long 0x98d74e00
  .long 0x8fa84300
  .long 0x76f1600
  .long 0x800f2470


	.text
.align 8
.globl payne_hanek_fp32_asm
.type	payne_hanek_fp32_asm, @function
/* $r0 contains fp32 <x> input */
payne_hanek_fp32_asm:
  $r1 = extfz $r0, 30, 23
  $r4 = and $r0, 0x80800000
  ;;
  $r2 = sbf $r1, (p-1+k)     /* msb_exp */
  $r3 = sbf $r1, (p - 1 - n) /* lsb_exp */
  $r4 = or $r4, 0x3f800000   /* $r4 = ml_mantissa_extraction_fp32(x) */
  ;;
  $r2 = srl $r2, 4           /* msb_index = msb_exp / 16 */
  make $r63 = global_array 
  make $r62 = 0x38000000     /* 0x1p-120 */
  ;;
  add $r62 = $r1, -120
  ;;
  sll $r62 = $r62, 23        /* scalingf */
  ;;
  ld.add.x4 $r32r33 = $r2[$r63] 
  ;;
  ld.add.x4 $r34r35 = $r2[$r63] 
  ;;
  fmul $r32 = $r32, $r62
  ;;
  fmul $r33 = $r33, $r62
  ;;
  fmul $r34 = $r34, $r62
  ;;
  fmul $r35 = $r35, $r62
  make $r44 = 0
  make $r45 = 0
  ;;
  ffmawd $r36r37 = $r44r45, $r4, $r32
  ;;
  ffmawd $r38r39 = $r44r45, $r4, $r33
  ;;
  ffmawd $r40r41 = $r44r45, $r4, $r34
  ;;
  ffmawd $r42r43 = $r44r45, $r4, $r35
  ;;
  /* red_to_exp chunk[0] */
  extfz $r32 = $r37, 30, 20
  make  $r33 = 0xffffffff
  ;;
  sbf $r32 = $r32, (53 + 3 + 127)
  ;;
  add $r44 = $r32, -52
  ;;
  cmove.gez $r32 = $r44, 64
  ;; 
  slld $r34:$r35 = $r33:$r33, $r32
  ;;
  andd $r34:$r35 = $r36:$r37, $r34:$r35
  ;; 
  fsbfd $r36r37 = $r34r35, $r36r37
  ;;
  /* red_to_exp chunk[1] */
  extfz $r32 = $r39, 30, 20
  ;;
  sbf $r32 = $r32, (53 + 3 + 127)
  ;;
  add $r44 = $r32, -52
  ;;
  cmove.gez $r32 = $r44, 64
  ;; 
  slld $r34:$r35 = $r33:$r33, $r32
  ;;
  andd $r34:$r35 = $r38:$r39, $r34:$r35
  ;; 
  fsbfd $r38r39 = $r34r35, $r38r39
  ;;
  /* red_to_exp chunk[2] */
  extfz $r32 = $r41, 30, 20
  ;;
  sbf $r32 = $r32, (53 + 3 + 127)
  ;;
  add $r44 = $r32, -52
  ;;
  cmove.gez $r32 = $r44, 64
  ;; 
  slld $r34:$r35 = $r33:$r33, $r32
  ;;
  andd $r34:$r35 = $r40:$r41, $r34:$r35
  ;; 
  fsbfd $r40r41 = $r34r35, $r40r41
  ;;
  /* red_to_exp chunk[3] */
  extfz $r32 = $r43, 30, 20
  ;;
  sbf $r32 = $r32, (53 + 3 + 127)
  ;;
  add $r44 = $r32, -52
  ;;
  cmove.gez $r32 = $r44, 64
  ;; 
  slld $r34:$r35 = $r33:$r33, $r32
  ;;
  andd $r34:$r35 = $r42:$r43, $r34:$r35
  ;; 
  fsbfd $r42r43 = $r34r35, $r42r43
  ;;
  faddd $r36r37 = $r36r37, $r38r39
  ;;
  faddd $r36r37 = $r36r37, $r40r41
  ;;
  faddd $r36r37 = $r36r37, $r42r43
  ;;
  fabsd $r0:$r1 = $r36:$r37
	ret
	;;
	.size	payne_hanek_fp32_asm, .-payne_hanek_fp32_asm
	.ident	"Nicolas Brunie, Copyright(2015) Kalray"


