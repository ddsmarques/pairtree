#include "PairTreeNode.h"

#include "CompareUtils.h"
#include "ErrorUtils.h"


PairTreeNode::PairTreeNode(NodeType type, int64_t attribCol, int64_t splitValue)
  : DecisionTreeNode(type, attribCol, splitValue), alpha_(1) {}


void PairTreeNode::setAlpha(long double alpha) {
  alpha_ = alpha;
}


long double PairTreeNode::getAlpha() {
  return alpha_;
}


std::shared_ptr<DecisionTreeNode> PairTreeNode::getTree(long double targetAlpha) {
  std::shared_ptr<DecisionTreeNode> node = nullptr;
  if (type_ == NodeType::LEAF || CompareUtils::compare(alpha_, targetAlpha) > 0) {
    node = std::make_shared<DecisionTreeNode>(NodeType::LEAF);
    node->setLeafValue(leafValue_);
  } else {
    node = std::make_shared<DecisionTreeNode>(type_, attribCol_, splitValue_);
    if (node->getType() == NodeType::REGULAR_NOMINAL) {
      for (auto const& child : children_) {
        node->addChild(std::static_pointer_cast<PairTreeNode>(child.second)->getTree(targetAlpha),
                       { child.first });
      }
    } else {
      node->addLeftChild(std::static_pointer_cast<PairTreeNode>(getLeftChild())->getTree(targetAlpha));
      node->addRightChild(std::static_pointer_cast<PairTreeNode>(getRightChild())->getTree(targetAlpha));
    }
  }
  return node;
}
