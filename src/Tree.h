// Copyright (c) 2018 Daniel dos Santos Marques <danielsmarques7@gmail.com>
// License: BSD 3 clause

#pragma once
#include "ConfigTree.h"
#include "DataSet.h"
#include "DecisionTreeNode.h"

#include <memory>

class Tree {
public:
  virtual std::shared_ptr<DecisionTreeNode> createTree(DataSet& ds, std::shared_ptr<ConfigTree> c) = 0;
};
