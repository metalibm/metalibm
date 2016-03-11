#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <support_lib/ml_utils.h>

double new_ut_payne_hanek(double x);

#define TEST_NUM 10

double test_input[TEST_NUM] = {
  0x1.d09b7c184f85ep576,
  0x1.bed72098ea07ep224,
  0x1.efd9a2716c3d5p202,
  0x1.349465dae1e6cp27,
  0x1.626b347099a37p438,
  0x1.ae79d3c16db90p942,
  0x1.780b5609879aep838,
  0x1.eba4bb73ec7f2p122,
  0x1.b34801c742c28p942,
  0x1.2a31973d9542ap635

};

double expected[TEST_NUM] = {
  0xf.5ec547da1b4a8p0, 
  0xd.954f9a9626668p0, 
  0xf.84523ceea6118p0, 
  0xe.1ec4f3235b1cp0, 
  0x7.eb9a08df8c268p0, 
  0xd.14ae025ed2bb8p0, 
  0x1.ae371fa63afdp4, 
  0x1.39867b7c8d6bfp4, 
  0x1.419006f417531p4, 
  0x1.4c48bd89baa4bp4
};

int main(int argc, char** argv) {

  unsigned i;
  for (i = 0; i < TEST_NUM; i++) {
    double result = new_ut_payne_hanek(test_input[i]);
    if (result != expected[i]) {
      printf("ERROR test %d: %"PRIx64" vs %"PRIx64"[exp]\n", i, double_to_64b_encoding(result), double_to_64b_encoding(expected[i]));
      return 1;
    }
  }

  printf("TEST SUCCESS\n");

  return 0;
}
