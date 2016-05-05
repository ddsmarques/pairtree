#pragma once
#include "DataSet.h"
#include "DecisionTreeNode.h"

#include <memory>

class Tester {
public:
  struct TestResults {
    long double score;
    long double savings;
  };
  TestResults test(std::shared_ptr<DecisionTreeNode> tree, DataSet& ds,
                   std::string outputFileName);
};
