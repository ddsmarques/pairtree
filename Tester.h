#pragma once
#include "DataSet.h"
#include "DecisionTreeNode.h"

#include <memory>

class Tester {
public:
  long double test(std::shared_ptr<DecisionTreeNode> tree, DataSet& ds,
                   std::string outputFileName);
};
