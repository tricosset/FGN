#include "fgn.hpp"

#define OUTPUT_NOISE_SIZE (1U << 20)
#define REPETITIONS 1

int main(int argc, const char *argv[])
{
	typedef int32_t T;
	fgn_dr_mp<uint8_t, T, 1, 32> rng(21, 128, OUTPUT_NOISE_SIZE*REPETITIONS, OUTPUT_NOISE_SIZE*REPETITIONS, true, true);

	T *noise = new T[OUTPUT_NOISE_SIZE]();
	bzero(noise, OUTPUT_NOISE_SIZE);
	mpfr_t real_center;
	mpfr_init_set_d(real_center, 3.0, MPFR_RNDN);

	uint64_t start = rdtsc();
	for (unsigned i = 0; i < REPETITIONS ; i++)
	{
		rng.getNoise(noise, OUTPUT_NOISE_SIZE, real_center);
	}
	uint64_t stop = rdtsc();

	printf("fgn_dr_mp: Gaussian noise cycles = %.2e samples = %.2e cycles/sample = %.2e\n", (float) (stop - start)/(REPETITIONS), (float) (OUTPUT_NOISE_SIZE), (float)(stop - start) / (OUTPUT_NOISE_SIZE*REPETITIONS));

#if(0)
	mpfr_t average;
	mpfr_init_set_si(average, noise[0], MPFR_RNDN);
	for (unsigned i = 1; i < OUTPUT_NOISE_SIZE; i++)
	{
		mpfr_add_si(average, average, noise[i], MPFR_RNDN);
	}
	mpfr_div_ui(average, average, OUTPUT_NOISE_SIZE, MPFR_RNDN);
	printf ("Average is ");
	mpfr_out_str (stdout, 10, 0, average, MPFR_RNDD);
	putchar ('\n');
#endif

#if(0)
  std::cout << "fgn_dr_mp: Noise generated below" << std::endl;
	std::cout << "[";
	for (unsigned int i = 0; i < (1 << 19); i++)
	{
		printf("%i,", noise[i]);
	}
	std::cout << "]" << std::endl;
#endif

	delete[] noise;

	return 0;
}






