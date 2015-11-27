#include "DecisionTreeNode.h"
#include "ErrorUtils.h"

#include <iostream>

DecisionTreeNode::DecisionTreeNode(NodeType type, int64_t attribCol)
  : type_(type), attribCol_(attribCol) {}

void DecisionTreeNode::setLeafValue(int64_t leafValue) {
  leafValue_ = leafValue;
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

void DecisionTreeNode::setName(std::string name) {
  name_ = name;
}

void DecisionTreeNode::printNode(std::string prefix) {
  if (type_ == NodeType::LEAF) {
    std::cout << name_ << ", LEAF, " << leafValue_ << std::endl;
  } else {
    std::cout << name_ << ", REGULAR, " << attribCol_ << std::endl;
    prefix += "| ";
    for (auto it : children_) {
      std::cout << prefix << it.first << ": ";
      it.second->printNode(prefix);
    }
  }
}

