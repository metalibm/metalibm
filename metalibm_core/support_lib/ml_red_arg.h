
typedef struct {
  unsigned data_size, cst_msb, k, chunk_size;
  unsigned exp_offset;
  float cst_data[];
} ph_data_t;

float payne_hanek_fp32(ph_data_t* ph_data, float x, unsigned n);

double payne_hanek_fp32_ext(ph_data_t* ph_data, float x, unsigned n);

