#pragma once
#include "DataSet.h"
#include "DecisionTreeNode.h"

#include <vector>

class GreedyTree {
public:
  GreedyTree();
  std::shared_ptr<DecisionTreeNode> createBackBone(DataSet& ds, int64_t bbSize);

private:
  std::shared_ptr<DecisionTreeNode> createBackBoneRec(DataSet& ds, int64_t bbSize, std::vector<bool> availableAttrib);
};
