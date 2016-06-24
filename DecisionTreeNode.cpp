#include "DecisionTreeNode.h"
#include "ErrorUtils.h"

#include <iostream>

DecisionTreeNode::DecisionTreeNode(NodeType type, int64_t attribCol, int64_t splitValue)
  : type_(type), attribCol_(attribCol), splitValue_(splitValue), leafValue_(-1) {}

void DecisionTreeNode::setLeafValue(int64_t leafValue) {
  leafValue_ = leafValue;
}

void DecisionTreeNode::setName(std::string name) {
  name_ = name;
}

void DecisionTreeNode::setAttribCol(int64_t attribCol) {
  ErrorUtils::enforce(type_ == NodeType::REGULAR_NOMINAL, "Node must be REGULAR_NOMINAL.");
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

int64_t DecisionTreeNode::getSeparator() {
  return splitValue_;
}


std::shared_ptr<DecisionTreeNode> DecisionTreeNode::getLeftChild() {
  ErrorUtils::enforce(type_ == NodeType::REGULAR_ORDERED, "Node must be REGULAR_ORDERED type.");
  if (children_.find(0) != children_.end()) {
    return children_[0];
  }
  return nullptr;
}


std::shared_ptr<DecisionTreeNode> DecisionTreeNode::getRightChild() {
  ErrorUtils::enforce(type_ == NodeType::REGULAR_ORDERED, "Node must be REGULAR_ORDERED type.");
  if (children_.find(1) != children_.end()) {
    return children_[1];
  }
  return nullptr;
}


int64_t DecisionTreeNode::getSize() {
  if (type_ == NodeType::LEAF) {
    return 1;
  } else if (type_ == NodeType::REGULAR_ORDERED) {
    int64_t leftSize = 0;
    int64_t rightSize = 0;
    if (children_.find(0) != children_.end()) {
      leftSize = children_[0]->getSize();
    }
    if (children_.find(1) != children_.end()) {
      rightSize = children_[1]->getSize();
    }
    return 1 + leftSize + rightSize;
  } else {
    int64_t size = 1;
    for (auto it = children_.begin(); it != children_.end(); it++) {
      size += it->second->getSize();
    }
    return size;
  }
}


void DecisionTreeNode::addChild(std::shared_ptr<DecisionTreeNode> child, std::vector<int64_t> v) {
  ErrorUtils::enforce(type_ == NodeType::REGULAR_NOMINAL,
                      "Can only add a nominal child to a REGULAR_NOMINAL node.");
  for (int64_t i = 0; i < v.size(); i++) {
    children_[v[i]] = child;
  }
}


void DecisionTreeNode::addLeftChild(std::shared_ptr<DecisionTreeNode> child) {
  ErrorUtils::enforce(type_ == NodeType::REGULAR_ORDERED,
                      "Can only add a left child to REGULAR_ORDERED node.");
  children_[0] = child;
}


void DecisionTreeNode::addRightChild(std::shared_ptr<DecisionTreeNode> child) {
  ErrorUtils::enforce(type_ == NodeType::REGULAR_ORDERED,
                      "Can only add a right child to REGULAR_ORDERED node.");
  children_[1] = child;
}


int64_t DecisionTreeNode::classify(std::shared_ptr<Sample> s) {
  if (type_ == NodeType::LEAF) {
    return leafValue_;
  }

  ErrorUtils::enforce(attribCol_ < s->inxValue_.size(), "Sample doesn't have the required column");
  auto next = findChild(s->inxValue_[attribCol_]);
  if (next == children_.end()) {
    return -1;
  }
  return next->second->classify(s);
}

bool DecisionTreeNode::isLeaf() {
  return type_ == DecisionTreeNode::NodeType::LEAF;
}


std::map<int64_t, std::shared_ptr<DecisionTreeNode>>::iterator DecisionTreeNode::findChild(int64_t inxValue) {
  ErrorUtils::enforce(type_ != NodeType::LEAF, "Can't find a child on a LEAF node.");

  if (type_ == NodeType::REGULAR_NOMINAL) {
    return children_.find(inxValue);
  } else {
    if (inxValue <= splitValue_) {
      return children_.find(0);
    }
    return children_.find(1);
  }
}
