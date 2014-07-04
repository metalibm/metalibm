#include <random_gen.hpp>

#include <iostream>


void ML_FloatingPointRNG::seed(int n) {
    // initializing and seeding m_state
    gmp_randinit_mt(m_state);
    gmp_randseed_ui(m_state, n);

    // seeding stdlib rand functions
    srand(n);
}


mpz_class gen_n_bits_string(int n, gmp_randstate_t randstate) {
    mpz_class o;
    mpz_urandomb(o.get_mpz_t(), randstate, n);
    return o;
}


mpz_class ML_FloatingPointRNG::generateRandomIEEESubnormal(int wE, int wF) {
    mpz_class mantissa, sign_field, result;

    // generate uniformly randomized mantissa 
    // or mantissa with uniformly randomize width
    if (rand() % 2 == 0) {
        mantissa = gen_n_bits_string(wF, m_state);
    } else {
        mantissa = gen_n_bits_string(rand() % (wF+1), m_state);
    }

    sign_field = rand() % 2;
    result = (sign_field << (wF+wE)) + mantissa;

    return result;
}


mpz_class ML_FloatingPointRNG::generateRandomIEEENormal(int wE, int wF) {
    mpz_class mantissa, exp_field, sign_field, result;

    // generate uniformly randomized mantissa 
    // or mantissa with uniformly randomize width
    if (rand() % 2 == 0) {
        mantissa = gen_n_bits_string(wF, m_state);
    } else {
        mantissa = gen_n_bits_string(rand() % (wF+1), m_state);
    }

    exp_field = gen_n_bits_string(wE, m_state);
    sign_field = rand() % 2;
    result = (sign_field << (wF+wE)) + (exp_field << wF) + mantissa;

    return result;
}

mpz_class ML_FloatingPointRNG::generateTrickyValue(int wE, int wF, int& index) {
	mpz_class tmp;
	mpz_class exp1 = (mpz_class(1) << wE) - 1;
	mpz_class sig1 = (mpz_class(1) << wF) - 1;
	mpz_class limitOne = mpz_class(1) << (wF - 1);
	switch (index++ % 20) {
		case plus_infty:
            {
                mpz_class sign = 0; 
                mpz_class mantissa = 0; 
                mpz_class exp = (1 << wE) - 1;
                return (sign << (wF+wE)) | (exp << wF) | mantissa;
            }
		case minus_infty:
            {
                mpz_class sign = 1; 
                mpz_class mantissa = 0; 
                mpz_class exp = (1 << wE) - 1;
                return (sign << (wF+wE)) | (exp << wF) | mantissa;
            }
		case plus_omega:
            {
                mpz_class sign = 0; 
                mpz_class mantissa = 0; 
                mpz_class exp = (1 << wE) - 2;
                return (sign << (wF+wE)) | (exp << wF) | mantissa;
            }
		case minus_omega:
            {
                mpz_class sign = 1; 
                mpz_class mantissa = 0; 
                mpz_class exp = (1 << wE) - 2;
                return (sign << (wF+wE)) | (exp << wF) | mantissa;
            }
		case qnan:
            {
                mpz_class sign = rand() % 2;
                mpz_class mantissa = gen_n_bits_string(wF-1, m_state) + 1;
                mpz_class qbit = ((mpz_class) 1) << (wF-1);
                mpz_class exp = (1 << wE) - 1;
                return (sign << (wF+wE)) | (exp << wF) | qbit | mantissa;
            }
        case snan:
            {
                mpz_class sign = rand() % 2;
                mpz_class mantissa = gen_n_bits_string(wF-1, m_state) + 1;
                mpz_class exp = (1 << wE) - 1;
                return (sign << (wF+wE)) | (exp << wF) | mantissa;
            }
        case plus_zero: 
            return 0;
        case minus_zero:
            return ((mpz_class) 1) << (wF+wE+1);
		case close_one:
			return (((1 << (wE-1)) -1 ) << wF) + gen_n_bits_string(rand() % wF, m_state);
		default: 
			return gen_n_bits_string(wE+wF+1, m_state);
	};
}


#define RANDOMGEN_NBTC 18

mpz_class ML_FloatingPointRNG::generateIEEETestValue(int case_id, int wE, int wF, int &index) {
	const int numberCase = RANDOMGEN_NBTC;
	switch(case_id % numberCase) {
		case 0:
			// totally random case
			return gen_n_bits_string(wE+wF+1, m_state); 
            break; // better safe than sorry
		case 1:
		case 6:
		case 9:
			// subnormal ieee case
			return generateRandomIEEESubnormal(wE,wF); 
            break;
		case 2:
		case 7:
		case 10:
			// small exponent
			{
				mpz_class sgn = gen_n_bits_string(1, m_state) << (wE+wF);
				mpz_class field = gen_n_bits_string(wF, m_state);
				mpz_class exp = gen_n_bits_string(wE/3, m_state) << wF;
				return sgn + exp + field;
			};
            break;
		case 3:
		case 8:
		case 11:
			// big exponent 
			{
				mpz_class sgn = gen_n_bits_string(1, m_state) << (wE+wF);
				mpz_class field = gen_n_bits_string(wF, m_state);
				mpz_class exp = ((mpz_class(1) << wE) - (gen_n_bits_string(wE/3, m_state)+1)) << wF;
				return sgn + exp + field;
			};
		case 4:
			// special values case
		case 5:
			// tricky value case
			return generateTrickyValue(wE, wF, index);
            // static values
		case 12:
		case 13:
			{
				if (wE+wF+1 == 32) {
					mpz_class tmp = fp32_values[(case_id / RANDOMGEN_NBTC) % FP_VALUES_TABLE_SIZE];
					return tmp;
				} else if (wE+wF+1 == 64) {
					mpz_class tmp = set_mpz_from_uint64(fp64_values[(case_id / RANDOMGEN_NBTC) % FP_VALUES_TABLE_SIZE]);
					return tmp;
                } else if (wE+wF+1 == 16) {
                    mpz_class tmp = fp16_values[(case_id / RANDOMGEN_NBTC) % FP_VALUES_TABLE_SIZE];
                    return tmp;
				} else {
					std::cout << "ERROR: FP_VALUES only support fp16, fp32 and fp64 formats" << std::endl;
				}
			}
			break;
        case 14:
            {
                // exp is always even, sign is always positive
                mpz_class sign_exp = gen_n_bits_string(wE, m_state) | 1;
                uint32_t field_size = rand() % (wF/2);
                mpz_class field = gen_n_bits_string(field_size, m_state) | 1;
                mpz_class square = field * field;
                while ((square >> wF) != 1) {
                    square = (square << 1);
                }
                
                mpz_class input = (sign_exp << (wF)) | (square);
                return input;
            };
            break;
        case 15:
            {
                // exp is always even
                mpz_class sign_exp = gen_n_bits_string(wE, m_state);
                uint32_t field_size = rand() % (wF+1);
                mpz_class field = gen_n_bits_string(field_size, m_state);
                return (sign_exp << (wF+1)) | (field << (wF - field_size));
            };
            break;
		default: 
			// default random case
			return gen_n_bits_string(wE+wF+1, m_state); break;
    }
}

float ML_FloatingPointRNG::generate_random_float_interval(float low, float high) {
    float ratio = (float) rand() / (float) RAND_MAX;

    return ratio * (high - low) + low;
}

float ML_FloatingPointRNG::generate_random_float_interval_focus(float low, float high, int& gen_index) {
    //static int gen_index = -1;
    static int tricky_index = 0;
    gen_index++;
    switch (gen_index % 2) {
        case 0: 
            {
                float ratio = (float) rand() / (float) RAND_MAX;
                //return ratio * (high - low) + low;
                return (ratio * high + low) - ratio * low;
            }
        case 1:
            {
                mpz_class rvalue = generateIEEETestValue(gen_index / 2, 8, 23, tricky_index);
                return mpz_to_float(rvalue);
            }
    };
}

double ML_FloatingPointRNG::generate_random_double_interval(double low, double high) {
    double ratio = (double) rand() / (double) RAND_MAX;

    return ratio * (high - low) + low;
}


double ML_FloatingPointRNG::generate_random_double_interval_focus(double low, double high, int& gen_index) {
    //static int gen_index = -1;
    static int tricky_index = 0;
    gen_index++;
    switch (gen_index % 2) {
        case 0: 
            {
                double ratio = (double) rand() / (double) RAND_MAX;
                //return ratio * (high - low) + low;
                return (ratio * high + low) - ratio * low;
            }
        case 1:
            {
                mpz_class rvalue = generateIEEETestValue(gen_index / 2, 11, 52, tricky_index);
                return mpz_to_double(rvalue);
            }
    };
}

int ML_FloatingPointRNG::generate_index() {
    return rand();
}
