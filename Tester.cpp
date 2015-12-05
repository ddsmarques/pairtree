#include "Tester.h"

#include <fstream>
#include <iostream>

void Tester::test(std::shared_ptr<DecisionTreeNode> tree, DataSet& ds,
                  std::string outputFileName) {
  double score = 0;
  for (const auto& s : ds.samples_) {
    int64_t classInx = tree->classify(s);
    score += s->benefit_[classInx];
  }
  std::ofstream ofs(outputFileName, std::ofstream::app);
  ofs << "Score " << score << std::endl;
}
