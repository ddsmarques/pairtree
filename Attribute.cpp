#include "Attribute.h"

template <>
bool Attribute<int64_t>::lessThan(const int64_t& a, const int64_t& b) {
  return (a < b);
}

template <>
bool Attribute<double>::lessThan(const double& a, const double& b) {
  double EPS = 1e-9;
  return (b - a > EPS);
}

template <>
bool Attribute<std::string>::lessThan(const std::string& a, const std::string& b) {
  return (a.compare(b) < 0);
}