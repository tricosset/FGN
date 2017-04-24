#ifndef fgn_dr_mp_h
#define fgn_dr_mp_h

#define tstbit(x, n)  ((x << (63 - n)) >> 63 )

#define DEFAULT_SAMPLES_PER_BUFFER (1U << 15)

#include <stdint.h>
#include <list>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <climits>
#include <gmp.h>
#include <mpfr.h>
#include <string.h>
#include <tuple>
#include <typeinfo>
#include <assert.h>
#include "fastrandombytes.h"
#include "fgn_utils.hpp"


template<class in_class, class out_class, unsigned _lu_depth, unsigned _number_of_centers>
class fgn_dr_mp {
  typedef output<in_class, out_class> output_t;

  
private:
  unsigned int _bit_precision;
  unsigned int _word_precision;
  unsigned int _number_of_barriers;
  unsigned int _lu_size;
  unsigned int _flag_ctr1;
  double _sigma;
  mpfr_t _const_sigma;
  double _dp_const_sigma;
  unsigned int _security;
  unsigned int _samples;
  double _tail_bound;
  bool _verbose;
  bool _lazy;
  mpfr_t _mp_scaled_norm_fact;
  double _dp_norm_fact;
  unsigned int _low_precision_accuracy;
  unsigned int _lazy_bound;

  in_class **barriers[_number_of_centers];
  output_t *lu_table[_number_of_centers];

  unsigned int _samples_per_buffer;
  in_class *_noise_buffer, *_noise_init_ptr;
  uint64_t _innoise_bytesize, _used_words, _innoise_words;

  void check_template_params();
  void precomputeParameters();
  void precomputeBarrierValues(const mpfr_t center, in_class** &target_barriers); 
  void buildUncertaintyLookupTables();
  void initNoiseBuffer();
  void nn_gaussian_law(mpfr_t rop, const mpfr_t x_fr, const mpfr_t center);
  void compute_real_barrier(in_class* rop, mpfr_t real_center, out_class val);
  double compute_dp_barrier(mpfr_t real_center, out_class val);
  int cmp(in_class *op1, in_class *op2);
  int noise_cmp(unsigned *ctr, in_class *op1, in_class *op2);


public:
  static const unsigned int default_k;
  fgn_dr_mp(double sigma, unsigned int security, unsigned int samples, unsigned int samples_per_buffer = (1U << 10), bool verbose = false, bool lazy = true);
  ~fgn_dr_mp();
  void getNoise(out_class * const rand_data2out, uint64_t rlen, mpfr_t real_center); 
};


// Constructors
template<class in_class, class out_class, unsigned _lu_depth, unsigned _number_of_centers>
fgn_dr_mp<in_class, out_class, _lu_depth, _number_of_centers>::fgn_dr_mp( double sigma,
									  unsigned int security, unsigned int samples, unsigned int samples_per_buffer, bool verbose /*=false*/, bool lazy /*=true*/):
  _sigma(sigma),
  _security(security),
  _samples(samples),
  _verbose(verbose),
  _samples_per_buffer(samples_per_buffer),
  _lazy(lazy)
{

  // Check template parameters
  check_template_params();
 
  // General initialization functions
  precomputeParameters();

  mpfr_t center;
  mpfr_init_set_ui(center, 0, MPFR_RNDN);
  mpfr_t epsilon;
  mpfr_init_set_ui(epsilon, 1, MPFR_RNDN);
  mpfr_div_ui(epsilon, epsilon, _number_of_centers, MPFR_RNDN);

  for (unsigned i = 0; i < _number_of_centers; i++)
    {
      precomputeBarrierValues(center, barriers[i]);
      mpfr_add(center, center, epsilon, MPFR_RNDN);
    }

  buildUncertaintyLookupTables();
  if(_verbose) std::cout << 
		 "fgn_dr_mp: Uncertainty barriers computed and lookup tables created" << std::endl;
  initNoiseBuffer();

  //#define PRINT_UNCERTAINTY
#ifdef PRINT_UNCERTAINTY
  // Print differences between uncertainty barriers
  mpz_t int_value, int_value2;
  mpz_inits(int_value, int_value2, NULL);
  in_class* print = (in_class*) calloc(_bit_precision/8,sizeof(in_class));
  if(mpfr_get_d(_epsilon, MPFR_RNDN) != 0.0)
    for (int i = 0; i < (int)_number_of_barriers; i++)
      {
	memset(print, 0, sizeof(in_class)*_bit_precision/8);
	mpz_import(int_value, _bit_precision/8, 1, 1, 1,0, _barriers[i]);
	mpz_import(int_value2, _bit_precision/8, 1, 1, 1,0, _uncertainty_barriers[i]);
	mpz_sub(int_value, int_value, int_value2);
	mpz_export((void *) (print + ((int)_word_precision - 
				      (int)ceil( (double)mpz_sizeinbase(int_value, 256)/sizeof(in_class) ))), NULL,
		   1, sizeof(in_class), 0, 0, int_value);
	if (_verbose)
	  {
	    printf("Uncertainty[%2u] = ", i);
	    if (sizeof(in_class) == 1) for (unsigned j = 0 ; j < _word_precision; j++)
					 printf("%.2x", print[j]);
	    if (sizeof(in_class) == 2) for (unsigned j = 0 ; j < _word_precision; j++)
					 printf("%.4x", print[j]);
	    std::cout << std::endl;
	  }

      }
  free(print);
  mpz_clears(int_value, int_value2, NULL);
#endif
}


// Check template parameters
template<class in_class, class out_class, unsigned _lu_depth, unsigned _number_of_centers>
void fgn_dr_mp<in_class, out_class, _lu_depth, _number_of_centers>::check_template_params()
{

  // Uncertainty functions only work if _lu_depth == 2
  if(_lu_depth != 1)
    {
      std::cout << "fgn_dr_mp: CRITICAL _lu_depth must be 1" << std::endl;
      exit(2);
    }

  // Lookup tables can only have uint8_t or uint16_t indexes
  if (typeid(in_class) == typeid(uint8_t))
    _lu_size = 1<<8;
  else if(typeid(in_class) == typeid(uint16_t))
    _lu_size = 1<<16;
  else
    {
      std::cout << "fgn_dr_mp: CRITICAL in_class must be uint8_t or uint16_t" << std::endl;
      exit(3);
    }
}


// Compute some values (precision, number of barriers and outputs)
template<class in_class, class out_class, unsigned _lu_depth, unsigned _number_of_centers>
void fgn_dr_mp<in_class, out_class, _lu_depth, _number_of_centers>::precomputeParameters()
{
  double epsi, k;

  /* Lemma 1: 
   * tail_bound >= sqrt(1 + 2*log(tail_bound) + 2*k*log(2))  
   * IMPLIES delta_tailbound < 2**(-k)
   *
   * -epsi <= -k - log2(2 * tail_bound * sigma)) 
   * IMPLIES 2 * tail_bound * sigma * 2**(-epsi) <= 2**(-k)
   * 
   * Lemma 2 (and Lemma 1): 
   * 2 * tail_bound * sigma * 2**(-epsi) <= 2**(-k)
   * IMPLIES  delta_epsilon < 2**(-k)
   */

  // THUS setting 
  k = _security + 1 + ceil(log(_samples)/log(2));
  // IMPLIES (delta_tailbound + delta_epsilon)*_samples < 2**(-_security)
  // We can thus generate vectors of _samples samples securely

  // To compute the tail bound we use Newton-Raphson with three digits of precision
  // WE HAVE tail_bound >= sqrt(1 + 2*log(tail_bound) + 2*k*log(2)) 
  // IS EQUIV TO tail_bound**2 -2*log(tail_bound) - 1- 2*k*log(2) >= 0
  int digits = 3;
#ifdef BOOST_RAPHSON
  double max_guess = 1+2*k*log(2);
  double min_guess = sqrt(1 + 2*k*log(2)), guess = min_guess;
  _tail_bound = boost::math::tools::newton_raphson_iterate(funct(k),guess, min_guess, max_guess, digits);
#else
  double min_guess = sqrt(1 + 2*k*log(2));
  _tail_bound = newton_raphson(k, min_guess, digits);
#endif
  // We now can compute the precision needed 
  epsi = k + log2(2 * _tail_bound * _sigma);
  _bit_precision = ceil(epsi);
  _word_precision = ceil(_bit_precision/(8.0*sizeof(in_class)));
  // For the same cost we get a simpler situation and more precision
  _bit_precision = _word_precision*8*sizeof(in_class);
  // And the number of probabilities that must be computed
  // Only half of them are computed (others are symmetric)
  // but all values are stored to speedup noise generation
  _number_of_barriers = 2+2*ceil(_tail_bound*_sigma);
  if ( ((uint64_t)_number_of_barriers >> (sizeof(out_class) * 8 - 1)) != 0)
    std::cout << "fgn_dr_mp: WARNING out_class too small to contain some of the (signed) results" << std::endl;
  if ( ((uint64_t)_number_of_barriers >> (sizeof(int) * 8 - 1)) != 0)
    std::cout << "fgn_dr_mp: WARNING outputs are above 2**31, unexpected results" << std::endl;

  // We compute double precision accuracy and lazy bound

  double error = 0.2*pow(_sigma,2)*pow(_tail_bound,3)*pow(2,-51);
  if (-floor(log2(error))<0) _low_precision_accuracy = 0;
  else _low_precision_accuracy = (unsigned)-floor(log2(error));
  if (_verbose) std::cout << "low_precision_accuracy: " << _low_precision_accuracy << std::endl;
  _lazy_bound = (_low_precision_accuracy-8*sizeof(in_class))/(8*sizeof(in_class));

  // Finally we precompute 1/(2*sigma**2) to accelerate things
  mpfr_inits2(_bit_precision, _const_sigma, NULL);
  mpfr_set_d(_const_sigma, _sigma, MPFR_RNDN);
  mpfr_sqr(_const_sigma, _const_sigma, MPFR_RNDN);
  mpfr_mul_ui(_const_sigma, _const_sigma, 2, MPFR_RNDN); 
  mpfr_ui_div(_const_sigma, 1, _const_sigma, MPFR_RNDN);
  _dp_const_sigma = mpfr_get_d(_const_sigma, MPFR_RNDN);

  // Give some feedback
  if (_verbose) std::cout << "fgn_dr_mp: " << _number_of_barriers <<
		  " barriers with " << _bit_precision << 
		  " bits of precision will be computed" << std::endl;
}


template<class in_class, class out_class, unsigned _lu_depth, unsigned _number_of_centers>
void fgn_dr_mp<in_class, out_class, _lu_depth, _number_of_centers>::precomputeBarrierValues(const mpfr_t center, in_class** &target_barriers)
{
  // Declare and init mpfr vars
  mpfr_t sum, tmp, tmp2;
  mpfr_t *mp_barriers;
  mpfr_inits2(_bit_precision, sum, tmp, tmp2, NULL);

  // This var is used to export mpfr values
  mpz_t int_value;
  mpz_init2(int_value, _bit_precision);

  // Init SUM = \sum_{k=-ceil(tail_bound*sigma)}^{ceil(tail_bound*sigma)} exp(-(k+round(c)-c)^2/(2*sigma^2))
  // and compute on the loop with the barriers
  mpfr_set_ui(sum, 0, MPFR_RNDN);

  // Allocate memory for the barrier pointers
  target_barriers = (in_class **) malloc(_number_of_barriers*sizeof(in_class *)); 
  mp_barriers = (mpfr_t *) malloc(_number_of_barriers*sizeof(mpfr_t)); 

  // Now loop over the barriers
  for (int i = 0; i < (int)_number_of_barriers; i++)
    {
      // Init mpfr var
      mpfr_init2(mp_barriers[i], _bit_precision);

      // Compute the barrier value (without normalization)
      mpfr_set_si(tmp2, i-((int)_number_of_barriers)/2, MPFR_RNDN);
      nn_gaussian_law(tmp, tmp2, center);
      //mpfr_out_str(stdout, 10, 0, tmp2, MPFR_RNDN);
      //std::cout << std::endl;
      if (i==0) mpfr_set(mp_barriers[0], tmp, MPFR_RNDN);
      else 
	{
	  mpfr_add(mp_barriers[i], mp_barriers[i-1], tmp, MPFR_RNDN);
	}

      // Add the probability to the sum
      mpfr_add(sum, sum, tmp, MPFR_RNDN);
    }	
  
  // Invert the sum and scale it 
  mpfr_ui_div(sum, 1, sum, MPFR_RNDN);
  if (mpfr_cmp_ui(center,0)==0)
    {
      _dp_norm_fact = mpfr_get_d(sum, MPFR_RNDN);
    }
  mpfr_set_ui(tmp, 2, MPFR_RNDN);
  mpfr_pow_ui(tmp, tmp, _bit_precision, MPFR_RNDN);
  mpfr_sub_ui(tmp, tmp, 1, MPFR_RNDN);
  mpfr_mul(sum, sum, tmp, MPFR_RNDN);
  if (mpfr_cmp_ui(center,0)==0)
    {
      mpfr_init2(_mp_scaled_norm_fact,_bit_precision);
      mpfr_set(_mp_scaled_norm_fact, sum, MPFR_RNDN);
    }

  // Now that we got the inverted sum normalize and export
  for (unsigned i = 0; i < _number_of_barriers; i++)
    {
      // Allocate space
      target_barriers[i] = (in_class *) calloc(_word_precision,sizeof(in_class)); 
	  
      mpfr_mul(mp_barriers[i], mp_barriers[i], sum, MPFR_RNDN);  
      mpfr_get_z(int_value, mp_barriers[i], MPFR_RNDN);
      mpz_export((void *) (target_barriers[i] + ((int)_word_precision - 
						 (int)ceil( (double)mpz_sizeinbase(int_value, 256)/sizeof(in_class) ))), NULL, 1, sizeof(in_class), 0, 0, int_value);
#ifdef OUTPUT_BARRIERS
      mpz_out_str(stdout, 10, int_value);
      std::cout << " = Barriers[" << i << "] = " << std::endl;
      if (sizeof(in_class) == 1) for (unsigned j = 0 ; j < _word_precision; j++)
				   printf("%.2x", target_barriers[i][j]);
      if (sizeof(in_class) == 2) for (unsigned j = 0 ; j < _word_precision; j++)
				   printf("%.4x", target_barriers[i][j]);
      std::cout <<  std::endl;
#endif 
      mpfr_clear(mp_barriers[i]);
    }
  mpfr_clears(sum, tmp, tmp2, NULL);
  mpz_clear(int_value);
  mpfr_free_cache();
  free(mp_barriers);
}


//Build lookup tables used during noise generation
template<class in_class, class out_class, unsigned _lu_depth, unsigned _number_of_centers>
void fgn_dr_mp<in_class, out_class, _lu_depth, _number_of_centers>::buildUncertaintyLookupTables()
{
  _flag_ctr1 = 0;
  for (unsigned i = 0; i < _number_of_centers; i++)
    {
      unsigned lu_index1 = 0;
      unsigned flag_ctr1 = 0;

      // Allocate space for the lookup tables
      lu_table[i] = new output_t[_lu_size]();

      // We start building the first dimension of the lookup table
      // corresponding to the first in_class word of the barriers
      for (int64_t val = -((int)_number_of_barriers)/2, b_index = 0; val <= ((int)_number_of_barriers)/2 && lu_index1 < _lu_size;)
	{
	  while (lu_index1 < barriers[i][b_index][0] && lu_index1 < _lu_size)
	    {
	      lu_table[i][lu_index1].val = val;
	      lu_index1++;
	    }

	  // Flag the entry
	  lu_table[i][lu_index1].val = val;
	  lu_table[i][lu_index1].flag = true;
	  flag_ctr1++;
	  // If _lu_depth == 1 we have to list the barriers here
#ifdef OUTPUT_LUT_FLAGS
	  std::cout << "fgn_di_mp: flagged lu_table[" << i << "]["
		    << lu_index1 << "] for barriers " << val;
#endif
	  lu_table[i][lu_index1].nb_b_ptr = 0;
	  unsigned b_index_tmp = b_index;
	  while (b_index_tmp <_number_of_barriers && lu_index1 == barriers[i][b_index_tmp][0])
	    {
	      lu_table[i][lu_index1].nb_b_ptr++;
	      b_index_tmp++;
	    }
	  lu_table[i][lu_index1].a_b_ptr = new in_class*[lu_table[i][lu_index1].nb_b_ptr]();
	  unsigned a_b_ptr_index = 0;
	  // Prepare the first element of the chained list of barrier
	  lu_table[i][lu_index1].a_b_ptr[a_b_ptr_index++] = barriers[i][b_index++];
	  val++;
	  // If more that one barrier is present, we add them to the chained list
	  while ( (b_index<_number_of_barriers) && (lu_index1 == barriers[i][b_index][0]))
	    {
	      lu_table[i][lu_index1].a_b_ptr[a_b_ptr_index++] = barriers[i][b_index++];
	      val++;
#ifdef OUTPUT_LUT_FLAGS
	      std::cout << "fgn_di_mp: flagged lu_table[" << i << "][" << lu_index1 << "] for barriers " << val;
#endif
	    } // while
	  lu_index1++;
	}
      if (flag_ctr1 > _flag_ctr1) _flag_ctr1 = flag_ctr1;
    }
  // Give some feedback
  if (_verbose)
    {
      std::cout << "fgn_di_mp: Lookup tables built" << std::endl;
      std::cout << "lu_size: " << _lu_size << std::endl;
      std::cout << "out_class_bsize: " << sizeof(out_class)*8 << std::endl;
      std::cout << "bool_bsize: " << sizeof(bool)*8 << std::endl;
      std::cout << "in_class*_bsize: " << sizeof(in_class*)*8 << std::endl;
      if (_lu_depth == 2) std::cout << "flag_ctr1: " << _flag_ctr1 << std::endl;
      std::cout << "number_of_barriers: " << _number_of_barriers << std::endl;
      std::cout << "barrier_bsize: " << _bit_precision << std::endl;
      std::cout << "***********************************************" << std::endl;
      if (_lu_depth == 1) std::cout << "mem_bsize: " << _number_of_centers*_lu_size*(sizeof(out_class)*8 +  sizeof(bool)*8 + sizeof(in_class*)*8) + _number_of_barriers * _bit_precision << std::endl;
      std::cout << "***********************************************" << std::endl;
    }
}


template<class in_class, class out_class, unsigned _lu_depth, unsigned _number_of_centers>
void fgn_dr_mp<in_class, out_class, _lu_depth, _number_of_centers>::initNoiseBuffer()
{
  uint64_t computed_outputs;
  int64_t output1, output2;
  in_class *noise, *noise_init_ptr, *noise_tmp_ptr, input1, input2;
  double innoise_multiplier;

  // Expected number of input bytes per output byte
  innoise_multiplier = ((double)sizeof(in_class) + ((double)_flag_ctr1/(double)_lu_size) * log2(_sigma*sqrt(M_E*M_PI)) - (double)sizeof(in_class) + 1.0);
  _innoise_words = _samples_per_buffer * innoise_multiplier;
  _innoise_bytesize = sizeof(in_class) * _innoise_words;

  _noise_buffer = new in_class[_innoise_words];
  _noise_init_ptr = _noise_buffer;
  _used_words = 0;

  if (_verbose) std::cout << "fgn_di_mp: Using " << " " <<innoise_multiplier*sizeof(in_class)*8/std::min(log2(_number_of_barriers),(double)sizeof(out_class)*8) << " input bits per output bit" << std::endl;

  // Count time for uniform noise generation
  uint64_t start = rdtsc();
  fastrandombytes((uint8_t*)_noise_buffer, _innoise_bytesize);
  uint64_t stop = rdtsc();

  // Give some feedback
  if (_verbose) printf("fgn_di_mp: Uniform noise  cycles = %.2e bits = %.2e cycles/sample = %.2e\n", (double) (stop - start), (double) _innoise_bytesize * 8, (double)(stop-start)/(_samples_per_buffer));
}


template<class in_class, class out_class, unsigned _lu_depth, unsigned _number_of_centers>
void fgn_dr_mp<in_class, out_class, _lu_depth, _number_of_centers>::getNoise(out_class* const rand_outdata, uint64_t rlen, mpfr_t real_center)
{
  uint64_t computed_outputs, usedw1, usedw2, usedw3;
  int64_t output1, output2;
  int flagged;
  in_class *noise, *noise_tmp_ptr, input1;

  mpfr_t center_frac, tmp;
  mpfr_inits2(_bit_precision, center_frac, tmp, NULL);
  intmax_t center_floor = mpfr_get_sj(real_center,MPFR_RNDD);
  mpfr_sub_si(center_frac, real_center, center_floor, MPFR_RNDN);
  mpfr_mul_ui(tmp,center_frac,_number_of_centers,MPFR_RNDN);

  unsigned center_lu1 = mpfr_get_ui(tmp,MPFR_RNDZ);
  unsigned center_lu2 = (center_lu1+1)%_number_of_centers;
  unsigned step = (center_lu1+1)/_number_of_centers;
  unsigned b_index, cdf_ctr = 0, lazy_ctr = 0, max_ctr = 0;
  out_class min, cur, max;

  // Loop until all the outputs have been generated
  computed_outputs = 0;
  while (computed_outputs < rlen )
    {
      input1 = *_noise_buffer;
      _noise_buffer++;
      _used_words++;

      flagged = lu_table[center_lu1][input1].flag;

      noise_tmp_ptr = _noise_buffer;
      usedw1 = 0;

      // If flagged we have to look at the next in_class words
      if (flagged)
	{
	  min = 0, max = lu_table[center_lu1][input1].nb_b_ptr-1;
	  cur = (min+max)/2, b_index = 1;
	  while(max-min > 1)
	    {
	      if (*_noise_buffer > lu_table[center_lu1][input1].a_b_ptr[cur][b_index])
		{
		  min = cur;
		}
	      else if (*_noise_buffer < lu_table[center_lu1][input1].a_b_ptr[cur][b_index])
		{
		  max = cur;
		}
	      else
		{
		  b_index++;
		  _noise_buffer++;
		  usedw1++;
		}
	      cur = (min+max)/2;
	    }
	  if (min == 0)
	    {
	      if (*_noise_buffer < lu_table[center_lu1][input1].a_b_ptr[min][b_index])
		{
		  output1 = lu_table[center_lu1][input1].val;
		}
	      else
		{
		  output1 = lu_table[center_lu1][input1].val+1;
		}
	    }
	  else if (max == lu_table[center_lu1][input1].nb_b_ptr-1 && *_noise_buffer > lu_table[center_lu1][input1].a_b_ptr[max][b_index])
	    {
	      output1 = lu_table[center_lu1][input1].val+max+1;
	    }
	  else
	    {
	      output1 = lu_table[center_lu1][input1].val+max;
	    }
	  _noise_buffer++;
	  usedw1++;
	}
      else
	{
	  output1 = lu_table[center_lu1][input1].val;
	}

      flagged = lu_table[center_lu2][input1].flag;

      _noise_buffer = noise_tmp_ptr;
      usedw2 = 0;

      // If flagged we have to look at the next in_class word
      if (flagged)
	{
	  min = 0, max = lu_table[center_lu2][input1].nb_b_ptr-1;
	  cur = (min+max)/2, b_index = 1;
	  while(max-min > 1)
	    {
	      if (*_noise_buffer > lu_table[center_lu2][input1].a_b_ptr[cur][b_index])
		{
		  min = cur;
		}
	      else if (*_noise_buffer < lu_table[center_lu2][input1].a_b_ptr[cur][b_index])
		{
		  max = cur;
		}
	      else
		{
		  b_index++;
		  _noise_buffer++;
		  usedw2++;
		}
	      cur = (min+max)/2;
	    }
	  if (min == 0)
	    {
	      if (*_noise_buffer < lu_table[center_lu2][input1].a_b_ptr[min][b_index])
		{
		  output2 = lu_table[center_lu2][input1].val;
		}
	      else
		{
		  output2 = lu_table[center_lu2][input1].val+1;
		}
	    }
	  else if (max == lu_table[center_lu2][input1].nb_b_ptr-1 && *_noise_buffer > lu_table[center_lu2][input1].a_b_ptr[max][b_index])
	    {
	      output2 = lu_table[center_lu2][input1].val+max+1;
	    }
	  else
	    {
	      output2 = lu_table[center_lu2][input1].val+max;
	    }
	  _noise_buffer++;
	  usedw2++;
	}
      else
	{
	  output2 = lu_table[center_lu2][input1].val;
	}
      output2 += step;

      _noise_buffer = noise_tmp_ptr + (usedw1 > usedw2 ? usedw1 : usedw2);
      _used_words += (usedw1 > usedw2 ? usedw1 : usedw2);

      if (output1 == output2)
	{
	  _noise_buffer = noise_tmp_ptr + (usedw1 > usedw2 ? usedw1 : usedw2);
	  _used_words += (usedw1 > usedw2 ? usedw1 : usedw2);

	  // Add the obtained result to the list of outputs
	  rand_outdata[computed_outputs++] = output1 + (out_class) center_floor;
	}
      else
	{
	  cdf_ctr++;

	  bool high_prec_flag = true;

	  if (_lazy && _lazy_bound > 0)
	    {
	      _noise_buffer = noise_tmp_ptr;
	      usedw3 = 0;
	      double dp_bar = compute_dp_barrier(center_frac, output1);
	      dp_bar/=((double)(1<<(_bit_precision%(8*sizeof(in_class)))));
	      dp_bar*=(double)((1<<(8*sizeof(in_class))));
	      for (unsigned ctr = 0; ctr < _lazy_bound; ctr++)
		{
		  dp_bar*=(double)((1<<(8*sizeof(in_class))));
		  in_class ele = (in_class)dp_bar;
		  if(*(_noise_buffer) < ele)
		    {
		      rand_outdata[computed_outputs++] = output1 + (out_class) center_floor;
		      high_prec_flag = false;
		      lazy_ctr++;
		      break;
		    }
		  if(*(_noise_buffer) > ele)
		    {
		      rand_outdata[computed_outputs++] = output2 + (out_class) center_floor;
		      high_prec_flag = false;
		      lazy_ctr++;
		      break;
		    }
		  dp_bar -= (double)ele;
		  _noise_buffer++;
		  usedw3++;
		}
	    }

	  // TODO try interpolation before worst case

	  if (high_prec_flag)
	    {
	      _noise_buffer = noise_tmp_ptr;
	      unsigned ctr;
	      // This is the worst case, interpolation should come before this
	      in_class* real_barrier = (in_class *) calloc(_word_precision,sizeof(in_class));
	      compute_real_barrier(real_barrier, center_frac, output1);

	      if (noise_cmp(&ctr, _noise_buffer,real_barrier+1) < 0)
		{
		  // Add the obtained result to the list of outputs
		  rand_outdata[computed_outputs++] = output1 + (out_class) center_floor;
		}
	      else
		{
		  // Add the obtained result to the list of outputs
		  rand_outdata[computed_outputs++] = output2 + (out_class) center_floor;
		}
	      _used_words += ctr;
	      _noise_buffer += ctr;
	      if (ctr > max_ctr) max_ctr = ctr;
	      free(real_barrier);
	    }
	  else
	    {
	      _noise_buffer++;
	      _used_words += usedw3+1;
	    }
	}


#ifdef UNITTEST_ONEMILLION
      if ( (output > _sigma * 6) || (output < -_sigma*6) )
	{
	  std::cout << output << "fgn_di_mp: Unit test failed, this should happen once in a million. Uniform input leading to this is  ";
	  for (unsigned i = 0; i < _word_precision ; i++)
	    printf("%.2x", *(_noise_buffer-_word_precision+i));
	  std::cout << std::endl;
	}
#endif
#if 1
      // If too much noise has been used regenerate it
      if ((_used_words+_word_precision) >= _innoise_words)
	{
	  _noise_buffer = _noise_init_ptr;
	  _used_words = 0;
	  if (_verbose) std::cout << "fgn_di_mp: All the input bits have been used, regenerating them ..." << std::endl;
	  fastrandombytes((uint8_t*)_noise_buffer, _innoise_bytesize);
	}
#endif
    }
  if (_lazy && _verbose) std::cout << cdf_ctr-lazy_ctr << " multi-precision cdf computed and " << lazy_ctr <<" double-precision cdf computed for " << rlen << " samples" << std::endl;
  else if (_verbose) std::cout << cdf_ctr << " multi-precision cdf computed for " << rlen << " samples chosen with at most " << max_ctr << " words" << std::endl;
}


/* Compare two arrays word by word.
 * return 1 if op1 > op2, 0 if equals and -1 if op1 < op2 */
template<class in_class, class out_class, unsigned _lu_depth, unsigned _number_of_centers>
inline int fgn_dr_mp<in_class, out_class, _lu_depth, _number_of_centers>::cmp(in_class *op1, in_class *op2)
{

  for (int i = 0; i < (int)_word_precision; i++) 
    {

      if (op1[i] > op2[i]) return 1;

      else if (op1[i] < op2[i]) return -1;
    }
  return 0;
}


/* Compare two arrays word by word.
 * return 1 if op1 > op2, 0 if equals and -1 if op1 < op2 */
template<class in_class, class out_class, unsigned _lu_depth, unsigned _number_of_centers>
inline int fgn_dr_mp<in_class, out_class, _lu_depth, _number_of_centers>::noise_cmp(unsigned *ctr, in_class *op1, in_class *op2)
{
  (*ctr) = 0;
  for (int i = 0; i < (int)_word_precision; i++)
    {
      (*ctr)++;
      if (op1[i] > op2[i]) return 1;

      else if (op1[i] < op2[i]) return -1;
    }
  return 0;
}


// Compute exp(-(x-center)^2/(2*sigma^2)) this is not normalized ! (hence the nn)
template<class in_class, class out_class, unsigned _lu_depth, unsigned _number_of_centers>
void  inline fgn_dr_mp<in_class, out_class, _lu_depth, _number_of_centers>::nn_gaussian_law(mpfr_t rop, const mpfr_t x, const mpfr_t center)
{
  mpfr_sub(rop, x, center, MPFR_RNDN);
  mpfr_sqr(rop, rop, MPFR_RNDN);
  mpfr_neg(rop, rop, MPFR_RNDN);
  mpfr_mul(rop, rop, _const_sigma, MPFR_RNDN);
  mpfr_exp(rop, rop, MPFR_RNDN);
}


template<class in_class, class out_class, unsigned _lu_depth, unsigned _number_of_centers>
void  inline fgn_dr_mp<in_class, out_class, _lu_depth, _number_of_centers>::compute_real_barrier
(in_class* rop, mpfr_t real_center, out_class val)
{
  mpfr_t sum, tmp, tmp_val;
  mpfr_inits2(_bit_precision, sum, tmp, tmp_val, NULL);

  mpz_t int_value;
  mpz_init2(int_value, _bit_precision);

  mpfr_set_ui(sum, 0, MPFR_RNDN);

  for (out_class i = -((int)_number_of_barriers)/2 ; i <= val ; i++)
    {
      mpfr_set_si(tmp_val, i, MPFR_RNDN);
      nn_gaussian_law(tmp, tmp_val, real_center);
      mpfr_add(sum, sum, tmp, MPFR_RNDN);
    }

  mpfr_mul(sum, sum, _mp_scaled_norm_fact, MPFR_RNDN);
  mpfr_get_z(int_value, sum, MPFR_RNDN);
  mpz_export((void *) (rop + ((int)_word_precision - (int)ceil( (double)mpz_sizeinbase(int_value, 256)/(double)sizeof(in_class) ))), NULL, 1, sizeof(in_class), 0, 0, int_value);

  mpfr_clears(sum, tmp, tmp_val, NULL);
}


template<class in_class, class out_class, unsigned _lu_depth, unsigned _number_of_centers>
double  inline fgn_dr_mp<in_class, out_class, _lu_depth, _number_of_centers>::compute_dp_barrier
(mpfr_t real_center, out_class val)
{
  double center = mpfr_get_d(real_center, MPFR_RNDN);
  double rop = 0;
  for (out_class i = -((int)_number_of_barriers)/2 ; i <= val ; i++)
    {
      rop += exp( -((double)i-center) * ((double)i-center) * _dp_const_sigma);
    }

  rop *= _dp_norm_fact;

  return rop;
}


template<class in_class, class out_class, unsigned _lu_depth, unsigned _number_of_centers>
fgn_dr_mp<in_class, out_class, _lu_depth, _number_of_centers>::~fgn_dr_mp()
{
  for (unsigned i = 0; i < _number_of_centers; i++)
    {
      // Free allocated memory for the barriers
      for (unsigned ctr = 0; ctr < _number_of_barriers; ctr++)
	{
	  if (barriers[i][ctr] != NULL) free(barriers[i][ctr]);
	  barriers[i][ctr] = NULL;
	}
      if (barriers[i] != NULL) free(barriers[i]);
      barriers[i]=NULL;

      // Free allocated memory for look-up tables
      for (unsigned ctr = 0 ; ctr < _lu_size; ctr++)
	{
	  if (lu_table[i][ctr].a_b_ptr!=NULL) delete[](lu_table[i][ctr].a_b_ptr);
	}
      delete[](lu_table[i]);
    }

  // Free other variables
  mpfr_clears(_const_sigma,_mp_scaled_norm_fact,NULL);
  delete[] _noise_init_ptr;
}



#endif
