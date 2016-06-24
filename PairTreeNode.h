#pragma once
#include "DecisionTreeNode.h"

class PairTreeNode : public DecisionTreeNode {
public:
  PairTreeNode(NodeType type, int64_t attribCol = -1, int64_t splitValue = -1);

  void setAlpha(long double alpha);

  long double getAlpha();

  std::shared_ptr<DecisionTreeNode> getTree(long double targetAlpha);

private:
  long double alpha_;
};
