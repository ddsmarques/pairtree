// Copyright (c) 2018 Daniel dos Santos Marques <danielsmarques7@gmail.com>
// License: BSD 3 clause

#pragma once
#include "ConfigTree.h"
#include "DataSet.h"
#include "Tree.h"

#include <vector>

class ConfigGreedyDraw : public ConfigTree {
public:
  int totDraws;
  int64_t minLeaf;
};

class GreedyDrawTree : public Tree {
public:
  std::shared_ptr<DecisionTreeNode> createTree(DataSet& ds, std::shared_ptr<ConfigTree> c) override;

private:
  std::shared_ptr<DecisionTreeNode> createTreeRec(DataSet& ds, int64_t height,
                                                  int totDraws, int64_t minLeaf,
                                                  std::vector<bool> availableAttrib);
  bool isGoodAttribute(DataSet& ds, int64_t attribInx, int totDraws);
  double getRandomAttribute(DataSet& ds, int64_t attribInx);
  std::shared_ptr<DecisionTreeNode> createLeaf(DataSet& ds);

  // Order the attributes by smallest branching factor first
  std::vector<int64_t> attribOrder_;
};
