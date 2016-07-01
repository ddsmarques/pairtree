#pragma once
#include "DecisionTreeNode.h"

class PairTreeNode : public DecisionTreeNode {
public:
  PairTreeNode(NodeType type, int64_t attribCol = -1, int64_t splitValue = -1);

  void setAlpha(long double alpha);

  void setNumSamples(int64_t numSamples);

  long double getAlpha();

  long double getNumSamples();

  std::shared_ptr<DecisionTreeNode> getTree(long double targetAlpha,
                                            int64_t minSamples);

private:
  long double alpha_;
  int64_t numSamples_;
};
