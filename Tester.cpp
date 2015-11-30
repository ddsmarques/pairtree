#include "Tester.h"

#include <iostream>

void Tester::test(std::shared_ptr<DecisionTreeNode> tree, DataSet& ds) {
  double score = 0;
  for (const auto& s : ds.samples_) {
    int64_t classInx = tree->classify(s);
    score += s->benefit_[classInx];
  }
  std::cout << "Score " << score << std::endl;
}
