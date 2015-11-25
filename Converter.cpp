#include "Converter.h"
#include "ErrorUtils.h"
#include <cstdlib>
#include <sstream>
#include <string>

bool Converter::isInteger(const std::string& str) {
  if (str == "") return false;
  for (int i = 0; i < str.size(); i++) {
    if ((str[i] == '+' || str[i] == '-') && (i > 0)) {
      return false;
    }
    if (str[i] < '0' || str[i] > '9') {
      return false;
    }
  }
  return true;
}

bool Converter::isDouble(const std::string& str) {
  std::istringstream iss(str);
  double d;
  char c;
  return iss >> d && !(iss >> c);
}

template <>
int64_t Converter::fromString<int64_t>(const std::string& str) {
  ErrorUtils::enforce(Converter::isInteger(str), "Not a valid integer");
  //return _atoi64(str.c_str());
  int64_t n;
  char c;
  int scanned = sscanf_s(str.c_str(), "%lld%c", &n, &c);
  if (scanned == 1) {
    return n;
  }
  else {
    return INT64_MIN;
  }
}

template <>
double Converter::fromString<double>(const std::string& str) {
  ErrorUtils::enforce(Converter::isDouble(str), "Not a valid double");
  return std::stod(str);
}

template <>
static std::string Converter::toString(const int64_t& x) {
  return std::to_string(x);
}
