#include "GreedyTree.h"

#include "CompareUtils.h"
#include "ErrorUtils.h"

#include <string>

GreedyTree::GreedyTree() {}

std::shared_ptr<DecisionTreeNode> GreedyTree::createTree(DataSet& ds, std::shared_ptr<ConfigTree> c) {
  std::shared_ptr<ConfigGreedy> config = std::static_pointer_cast<ConfigGreedy>(c);
  std::vector<bool> availableAttrib(ds.getTotAttributes(), true);
  return createTreeRec(ds, config->height, config->minLeaf, availableAttrib);
}

std::shared_ptr<DecisionTreeNode> GreedyTree::createTreeRec(DataSet& ds, int64_t height, int64_t minLeaf,
                                                            std::vector<bool> availableAttrib) {
  ErrorUtils::enforce(ds.getTotClasses() > 0, "Invalid data set.");

  if (height == 0 || (minLeaf > 0 && ds.samples_.size() <= minLeaf)) {
    auto best = ds.getBestClass();

    std::shared_ptr<DecisionTreeNode> leaf = std::make_shared<DecisionTreeNode>(DecisionTreeNode::NodeType::LEAF);
    leaf->setName("LEAF " + std::to_string(best.first));
    leaf->setLeafValue(best.first);

    return leaf;
  }

  int64_t bestAttrib = -1;
  double bestScore = 0;
  for (int64_t i = 0; i < ds.getTotAttributes(); i++) {
    if (availableAttrib[i]) {
      double attribScore = 0;
      for (int64_t j = 0; j < ds.getAttributeSize(i); j++) {
        auto bestClass = ds.getSubDataSet(i, j).getBestClass();
        attribScore += bestClass.second;
      }
      if (bestAttrib == -1
          || CompareUtils::compare(attribScore, bestScore) > 0) {
        bestAttrib = i;
        bestScore = attribScore;
      }
    }
  }
  availableAttrib[bestAttrib] = false;

  int64_t bestAttribSize = ds.getAttributeSize(bestAttrib);
  std::vector<DataSet> allDS(bestAttribSize);
  for (int64_t i = 0; i < bestAttribSize; i++) {
    allDS[i].initAllAttributes(ds);
  }
  for (const auto& s : ds.samples_) {
    allDS[s->inxValue_[bestAttrib]].addSample(s);
  }

  std::shared_ptr<DecisionTreeNode> node = std::make_shared<DecisionTreeNode>(DecisionTreeNode::NodeType::REGULAR_NOMINAL,
                                                                              bestAttrib);
  for (int64_t j = 0; j < bestAttribSize; j++) {
    node->addChild(createTreeRec(allDS[j], height - 1, minLeaf, availableAttrib), { j });
  }
  return node;
}
