#pragma once
#include "ConfigTree.h"
#include "DataSet.h"
#include "Tree.h"

#include <vector>

class ConfigGreedyDraw : public ConfigTree {
public:
  int totDraws;
};

class GreedyDrawTree : public Tree {
public:
  std::shared_ptr<DecisionTreeNode> createTree(DataSet& ds, std::shared_ptr<ConfigTree> c) override;

private:
  std::shared_ptr<DecisionTreeNode> createTreeRec(DataSet& ds, int64_t height,
                                                  int totDraws,
                                                  std::vector<bool> availableAttrib);
  bool isGoodAttribute(DataSet& ds, int64_t attribInx, int totDraws);
  double getRandomAttribute(DataSet& ds, int64_t attribInx);
  std::shared_ptr<DecisionTreeNode> createLeaf(DataSet& ds);

  // Order the attributes by smallest branching factor first
  std::vector<int64_t> attribOrder_;
};
