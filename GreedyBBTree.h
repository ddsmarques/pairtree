#pragma once
#include "ConfigTree.h"
#include "DataSet.h"
#include "DecisionTreeNode.h"
#include "Tree.h"

#include <vector>

class ConfigGreedyBB : public ConfigTree {
};

class GreedyBBTree : public Tree {
public:
  std::shared_ptr<DecisionTreeNode> createTree(DataSet& ds, std::shared_ptr<ConfigTree> c) override;

private:
  std::shared_ptr<DecisionTreeNode> createTreeRec(DataSet& ds, int64_t height, std::vector<bool> availableAttrib);
  bool nextState(DataSet& ds, std::vector<std::pair<int64_t, int64_t>>& state);
  std::pair<std::shared_ptr<DecisionTreeNode>, double> createBB(DataSet& ds, const std::vector<std::pair<int64_t, int64_t>>& state);
  std::vector<std::pair<int64_t, int64_t>> initState(std::vector<bool> availableAttrib, int64_t height);
};
