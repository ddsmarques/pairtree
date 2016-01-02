#include "CompareUtils.h"

#include <cmath>

int CompareUtils::compare(const double& a, const double& b, double eps) {
  if (std::abs(a - b) <= eps) return 0;
  return a < b ? -1 : +1;
}