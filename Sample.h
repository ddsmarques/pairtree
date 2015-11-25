// This module represents a dataset sample.
// 
// Author: ddsmarques
//
#pragma once
#include <vector>

class Sample {
public:
  Sample(int totAttributes, int totClasses);
  std::vector<int64_t> inxValue_;
  std::vector<double> benefit_;
};