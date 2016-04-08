#pragma once
#include "ConfigTree.h"
#include "DataSet.h"
#include "DecisionTreeNode.h"
#include "Tree.h"

#include <vector>

class ConfigGreedy : public ConfigTree {
public:
  int64_t minLeaf;
};

class GreedyTree : public Tree {
public:
  GreedyTree();
  std::shared_ptr<DecisionTreeNode> createTree(DataSet& ds, std::shared_ptr<ConfigTree> c) override;

private:
  std::shared_ptr<DecisionTreeNode> createTreeRec(DataSet& ds, int64_t height,
                                                  int64_t minLeaf, std::vector<bool> availableAttrib);
};
