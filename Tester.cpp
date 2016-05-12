#include "Tester.h"

#include "CompareUtils.h"

#include <fstream>
#include <limits>
#include <iomanip>
#include <iostream>


Tester::TestResults Tester::test(std::shared_ptr<DecisionTreeNode> tree, DataSet& ds,
                                 std::string outputFileName) {
  TestResults result;
  result.score = 0;
  for (const auto& s : ds.samples_) {
    int64_t classInx = tree->classify(s);
    result.score += s->benefit_[classInx];
  }
  std::ofstream ofs(outputFileName, std::ofstream::app);
  ofs << "Score "
      << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
      << result.score << std::endl;

  auto best = ds.getBestClass();
  result.savings = std::numeric_limits<long double>::quiet_NaN();
  if (CompareUtils::compare(best.second, 0) != 0) {
    result.savings = 1 - result.score / best.second;
  }
  ofs << "Savings "
      << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
      << result.savings << std::endl;
  ofs << "Size "
      << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
      << tree->getSize() << std::endl;
  return result;
}
