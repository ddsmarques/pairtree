#pragma once
#include "ConfigTree.h"
#include "DataSet.h"
#include "DecisionTreeNode.h"
#include "Tree.h"

#include <vector>

class ConfigGreedy : public ConfigTree {
public:
  int64_t minLeaf;
  int64_t percentiles;
};

class GreedyTree : public Tree {
public:
  GreedyTree();
  std::shared_ptr<DecisionTreeNode> createTree(DataSet& ds, std::shared_ptr<ConfigTree> c) override;

private:
  std::shared_ptr<DecisionTreeNode> createTreeRec(DataSet& ds, int64_t height,
                                                  int64_t minLeaf, int64_t percentiles,
                                                  std::vector<bool> availableAttrib);
  std::pair<long double, int64_t> getAttribScore(DataSet& ds, int64_t attribInx,
                                                 int64_t percentiles);
  long double getNominalScore(DataSet& ds, int64_t attribInx);
  std::pair<long double, int64_t> getOrderedScore(DataSet& ds, int64_t attribInx,
                                                  int64_t percentiles);
};
