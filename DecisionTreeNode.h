// This module implements a decision tree structure.
// 
// Author: ddsmarques
//
#pragma once
#include "Sample.h"

#include <cstdint>
#include <memory>
#include <vector>
#include <map>
#include <string>

class DecisionTreeNode {
public:
  enum class NodeType {REGULAR, LEAF};

  DecisionTreeNode(NodeType type, int64_t attribCol = -1);

  void setLeafValue(int64_t leafValue);

  void setName(std::string name);

  void printNode(std::string prefix = "");

  void addChild(std::shared_ptr<DecisionTreeNode> child, std::vector<int64_t> v);

  int64_t classify(std::shared_ptr<Sample> s);

private:
  std::string name_;
  NodeType type_;
  int64_t attribCol_;
  int64_t leafValue_;
  std::map<int64_t, std::shared_ptr<DecisionTreeNode>> children_;
};