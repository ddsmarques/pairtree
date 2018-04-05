// Copyright (c) 2018 Daniel dos Santos Marques <danielsmarques7@gmail.com>
// License: BSD 3 clause

#pragma once
#include "DataSet.h"
#include "DecisionTreeNode.h"

#include <memory>

class Tester {
public:
  struct TestResults {
    long double score;
    long double savings;
    int64_t size;
  };
  TestResults test(std::shared_ptr<DecisionTreeNode> tree, DataSet& ds);

  void saveResult(TestResults result, std::string outputFileName);
};
