#include "CompareUtils.h"

#include <cmath>

int CompareUtils::compare(const long double& a, const long double& b, long double eps) {
  if (std::abs(a - b) <= eps) return 0;
  return a < b ? -1 : +1;
}