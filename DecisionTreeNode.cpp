#include "DecisionTreeNode.h"
#include "ErrorUtils.h"

#include <iostream>

DecisionTreeNode::DecisionTreeNode(NodeType type, int64_t attribCol)
  : type_(type), attribCol_(attribCol), leafValue_(-1) {}

void DecisionTreeNode::setLeafValue(int64_t leafValue) {
  ErrorUtils::enforce(type_ == NodeType::LEAF, "Node is not a leaf.");
  leafValue_ = leafValue;
}

void DecisionTreeNode::setName(std::string name) {
  name_ = name;
}

void DecisionTreeNode::setAttribCol(int64_t attribCol) {
  ErrorUtils::enforce(type_ == NodeType::REGULAR, "Node is a leaf.");
  attribCol_ = attribCol;
}

int64_t DecisionTreeNode::getAttribCol() {
  return attribCol_;
}

DecisionTreeNode::NodeType DecisionTreeNode::getType() {
  return type_;
}

std::string DecisionTreeNode::getName() {
  return name_;
}

int64_t DecisionTreeNode::getLeafValue() {
  return leafValue_;
}

void DecisionTreeNode::addChild(std::shared_ptr<DecisionTreeNode> child, std::vector<int64_t> v) {
  ErrorUtils::enforce(type_ != NodeType::LEAF, "Can't add a child to a leaf node.");
  for (int64_t i = 0; i < v.size(); i++) {
    children_[v[i]] = child;
  }
}

int64_t DecisionTreeNode::classify(std::shared_ptr<Sample> s) {
  if (type_ == NodeType::LEAF) {
    return leafValue_;
  }

  ErrorUtils::enforce(attribCol_ < s->inxValue_.size(), "Sample doesn't have the required column");
  auto next = children_.find(s->inxValue_[attribCol_]);
  if (next == children_.end()) {
    return -1;
  }
  return children_[s->inxValue_[attribCol_]]->classify(s);
}

bool DecisionTreeNode::isLeaf() {
  return type_ == DecisionTreeNode::NodeType::LEAF;
}
