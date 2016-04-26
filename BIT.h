#pragma once
#include <vector>


class BIT {
public:
  BIT(int64_t N);
  void update(int64_t inx, double value);
  double get(int64_t inx);

private:
  int64_t N_;
  std::vector<double> tree_;
};
