#ifndef fgn_utils_h
#define fgn_utils_h

#include <stdint.h>
#include <list>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <climits>
#include <gmpxx.h>
#include <mpfr.h>
#include <string.h>
#include <tuple>
#include <typeinfo>
#include <assert.h>
#include "fastrandombytes.h"

#ifdef BOOST_RAPHSON
#include <boost/math/tools/roots.hpp>
#endif

template<class in_class, class out_class>
struct output {
  out_class val;
  bool flag;
  unsigned nb_b_ptr;
  in_class** a_b_ptr;
};

template<class in_class, class out_class>
struct ct_output {
  out_class val;
  bool flag;
  std::list<in_class*> l_b_ptr;
};

template<class in_class>
struct element {
  in_class in;
  unsigned tar;

  element& operator=(element& a) {
    tar=a.tar;
    in=a.in;
    return *this;
  }
};

/* NOTATIONS (from paper Sampling from discrete Gaussians
 * for lattice-based cryptography on a constrainded device
 * Nagarjun C. Dwarakanath, Steven Galbraith
 * Journal : AAECC 2014 (25)
 *
 * m : lattice dimension, we set it to 1 in here (better results for same security) !!!!
 *
 * security : output distribution should be within 2^{-security}
 * statistical distance from the Gaussian distribution
 *
 * sigma : parameter of the Gaussian distribution (close
 * to the standard deviation but not equal)
 * SUM = \sum_{k=-\inf}^{\inf} exp(-(k-c)^2/(2*sigma^2))
 *
 * D_{sigma} : distribution such that for x\in \mathcal{Z}
 * Pr(x) = ro_{sigma}(x) = 1/SUM exp(-(x-c)^2/(2*sigma^2))
 *
 * Warning : can be replaced in some works by s = sqrt(2*pi)*sigma
 * if Pr(x) = 1/SUM exp(-pi*(x-c)^2/s^2)
 *
 * tail_bound : tail bound we only output points with
 * norm(point-c) < tail_bound*sigma
 *
 * bit_precision : bit precision when computing the probabilities
 *
 * delta_tailbound : statistical distance introduced
 * by the tail bound
 *
 * delta_epsi : statistical distance introduced by
 * probability approximation
 */



// HELPER FUNCTIONS

/* /!\ Warning only for x64 */
extern __inline__ uint64_t rdtsc(void)
{
  uint64_t a, d;
  __asm__ volatile ("rdtsc" : "=a" (a), "=d" (d));
  return (d<<32) | a;
}

// Function to be used in a Newton-Raphson solver
struct funct
{
  funct(double const& target) : k(target){}
  std::tuple<double, double> operator()(double const& x)
  {
    return std::make_tuple(x*x - 2*log(x) - 1 - 2*k*log(2), 2*x-2/x);
  }
private:
  double k;
};

static inline double newton_raphson(double k, double max_guess, int digits)
{
  unsigned max_counter = 1U<<15;
  std::tuple<double, double> values;
  double delta;
  double guess = max_guess;
  for (unsigned counter = 0 ; counter < max_counter; counter++)
    {
      values = funct(k)(guess);
      delta = std::get<0>(values)/std::get<1>(values);
      guess -= delta;
      if ( fabs(delta)/fabs(guess) < pow(10.0,-digits) ) break;
    }
  // In case there is a flat zone in the function
  while(0.95*guess*0.95*guess - 2*log(0.95*guess) - 1 - 2*k*log(2)>=0) guess*=0.95;
  // Test result
  if(guess*guess - 2*log(guess) - 1 - 2*k*log(2)<0)
    {
      std::cout << "fgn_c_dp: WARNING Newton-Raphson failed the generator is NOT secure" << std::endl;
    }
  return guess;
}

#endif
