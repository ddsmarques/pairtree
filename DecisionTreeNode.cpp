#include "DecisionTreeNode.h"
#include "ErrorUtils.h"

#include <iostream>

DecisionTreeNode::DecisionTreeNode(NodeType type, int64_t attribCol)
  : type_(type), attribCol_(attribCol) {}

void DecisionTreeNode::setLeafValue(int64_t leafValue) {
  leafValue_ = leafValue;
}

void DecisionTreeNode::setName(std::string name) {
  name_ = name;
}

int64_t DecisionTreeNode::getAttribCol() {
  return attribCol_;
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

void DecisionTreeNode::print2File(std::string fileName) {
  std::ofstream ofs(fileName, std::ofstream::app);
  printNode(ofs);
}

void DecisionTreeNode::printNode(std::ofstream& ofs, std::string prefix) {
  if (type_ == NodeType::LEAF) {
    ofs << name_ << ", LEAF, " << leafValue_ << std::endl;
  } else {
    ofs << name_ << ", REGULAR, " << attribCol_ << std::endl;
    prefix += "| ";
    for (const auto& child : children_) {
      ofs << prefix << child.first << ": ";
      child.second->printNode(ofs, prefix);
    }
  }
}

bool DecisionTreeNode::isLeaf() {
  return type_ == DecisionTreeNode::NodeType::LEAF;
}
