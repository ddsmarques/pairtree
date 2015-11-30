#pragma once
#include "DataSet.h"
#include "DecisionTreeNode.h"

#include <vector>

class GreedyTree {
public:
  GreedyTree();
  std::shared_ptr<DecisionTreeNode> createTree(DataSet& ds, int64_t height);

private:
  std::shared_ptr<DecisionTreeNode> createTreeRec(DataSet& ds, int64_t height, std::vector<bool> availableAttrib);
};
