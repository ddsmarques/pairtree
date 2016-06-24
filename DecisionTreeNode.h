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
#include <fstream>

class DecisionTreeNode {
public:
  enum class NodeType {REGULAR_NOMINAL, REGULAR_ORDERED, LEAF};

  DecisionTreeNode(NodeType type, int64_t attribCol = -1, int64_t splitValue = -1);

  void setLeafValue(int64_t leafValue);

  void setName(std::string name);

  void setAttribCol(int64_t attribCol);

  int64_t getAttribCol();

  NodeType getType();

  std::string getName();

  int64_t getLeafValue();

  int64_t getSeparator();

  std::shared_ptr<DecisionTreeNode> getLeftChild();

  std::shared_ptr<DecisionTreeNode> getRightChild();

  int64_t getSize();

  void addChild(std::shared_ptr<DecisionTreeNode> child, std::vector<int64_t> v);

  void addLeftChild(std::shared_ptr<DecisionTreeNode> child);

  void addRightChild(std::shared_ptr<DecisionTreeNode> child);

  int64_t classify(std::shared_ptr<Sample> s);

  bool isLeaf();

  std::map<int64_t, std::shared_ptr<DecisionTreeNode>> children_;

protected:
  std::map<int64_t, std::shared_ptr<DecisionTreeNode>>::iterator findChild(int64_t inxValue);

  std::string name_;
  NodeType type_;
  int64_t attribCol_;
  int64_t leafValue_;
  int64_t splitValue_;
};