#ifndef fgn_h
#define fgn_h

#include "fgn_dr_mp.hpp"

/*
   Multi-Precision Discrete Gaussian Sampler With Variable Center and Fixed Sigma:
   -  Offline:
        template<class in_class, class out_class, unsigned _lu_depth, unsigned _number_of_centers>
          fgn_dr_mp(double sigma, unsigned int security, unsigned int samples, bool verbose = true);
   -  Online:
        void fgn_dr_mp.getNoise(out_class * const rand_data2out, uint64_t rlen, mpfr_t real_center);
 */

#endif /* fgn_h */
