

float payne_hanek(ph_data_t* ph_data, float x, unsigned n);

typedef struct {
  unsigned data_size, cst_msb, k, chunk_size;
  float[] cst_data;
} ph_data_t;
