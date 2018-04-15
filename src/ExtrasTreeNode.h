// Copyright (c) 2018 Daniel dos Santos Marques <danielsmarques7@gmail.com>
// License: BSD 3 clause

#pragma once
#include "DecisionTreeNode.h"

class ExtrasTreeNode : public DecisionTreeNode {
public:
  ExtrasTreeNode(NodeType type, int64_t attribCol = -1, int64_t splitValue = -1);

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
