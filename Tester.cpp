#include "Tester.h"

#include <fstream>
#include <limits>
#include <iomanip>
#include <iostream>


long double Tester::test(std::shared_ptr<DecisionTreeNode> tree, DataSet& ds,
                  std::string outputFileName) {
  long double score = 0;
  for (const auto& s : ds.samples_) {
    int64_t classInx = tree->classify(s);
    score += s->benefit_[classInx];
  }
  std::ofstream ofs(outputFileName, std::ofstream::app);
  ofs << "Score " << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << score << std::endl;
  return score;
}
