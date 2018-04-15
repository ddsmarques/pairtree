// Copyright (c) 2018 Daniel dos Santos Marques <danielsmarques7@gmail.com>
// License: BSD 3 clause

#include "ExtrasTreeNode.h"

#include "CompareUtils.h"
#include "ErrorUtils.h"


ExtrasTreeNode::ExtrasTreeNode(NodeType type, int64_t attribCol, int64_t splitValue)
  : DecisionTreeNode(type, attribCol, splitValue), alpha_(1), numSamples_(0) {}


void ExtrasTreeNode::setAlpha(long double alpha) {
  alpha_ = alpha;
}


void ExtrasTreeNode::setNumSamples(int64_t numSamples) {
  numSamples_ = numSamples;
}


long double ExtrasTreeNode::getAlpha() {
  return alpha_;
}


long double ExtrasTreeNode::getNumSamples() {
  return numSamples_;
}


std::shared_ptr<DecisionTreeNode> ExtrasTreeNode::getTree(long double targetAlpha,
                                                        int64_t minSamples) {
  std::shared_ptr<DecisionTreeNode> node = nullptr;
  if (type_ == NodeType::LEAF || CompareUtils::compare(alpha_, targetAlpha) > 0
      || numSamples_ <= minSamples) {
    node = std::make_shared<DecisionTreeNode>(NodeType::LEAF);
    node->setLeafValue(leafValue_);
  } else {
    node = std::make_shared<DecisionTreeNode>(type_, attribCol_, splitValue_);
    if (node->getType() == NodeType::REGULAR_NOMINAL) {
      std::map<int64_t, std::shared_ptr<DecisionTreeNode>>::iterator it, itPrev;
      for (it = children_.begin(); it != children_.end(); it++) {
        bool found = false;
        for (itPrev = children_.begin(); itPrev != it; itPrev++) {
          if (it->second == itPrev->second) {
            found = true;
            node->addChild(std::static_pointer_cast<ExtrasTreeNode>(node->children_[itPrev->first]), { it->first });
            break;
          }
        }
        if (!found) {
          node->addChild(std::static_pointer_cast<ExtrasTreeNode>(it->second)->getTree(targetAlpha, minSamples),
                         { it->first });
        }
      }
    } else {
      node->addLeftChild(std::static_pointer_cast<ExtrasTreeNode>(getLeftChild())->getTree(targetAlpha, minSamples));
      node->addRightChild(std::static_pointer_cast<ExtrasTreeNode>(getRightChild())->getTree(targetAlpha, minSamples));
    }
  }
  return node;
}
