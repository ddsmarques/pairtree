// Copyright (c) 2018 Daniel dos Santos Marques <danielsmarques7@gmail.com>
// License: BSD 3 clause

#include "BIT.h"


BIT::BIT(int64_t N) : N_(N+1), tree_(N+1, 0) {}


void BIT::update(int64_t inx, double value) {
  //inx++;
  while (inx < N_) {
    tree_[inx] += value;
    inx += (inx & -inx);
  }
}


double BIT::get(int64_t inx) {
  double sum = 0;
  while (inx > 0) {
    sum += tree_[inx];
    inx -= (inx & -inx);
  }
  return sum;
}
