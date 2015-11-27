#include "GreedyTree.h"

#include "ErrorUtils.h"

#include <string>

GreedyTree::GreedyTree() {}

std::shared_ptr<DecisionTreeNode> GreedyTree::createBackBone(DataSet& ds, int64_t bbSize) {
  std::vector<bool> availableAttrib(ds.getTotAttributes(), true);
  return createBackBoneRec(ds, bbSize, availableAttrib);
}

std::shared_ptr<DecisionTreeNode> GreedyTree::createBackBoneRec(DataSet& ds, int64_t bbSize, std::vector<bool> availableAttrib) {
  ErrorUtils::enforce(ds.getTotClasses() > 0, "Invalid data set.");

  if (bbSize == 1) {
    std::vector<double> classScore(ds.getTotClasses());
    for (auto s : ds.samples_) {
      for (int64_t i = 0; i < classScore.size(); i++) {
        classScore[i] += s->benefit_[i];
      }
    }
    int64_t best = 0;
    for (int64_t i = 1; i < classScore.size(); i++) {
      if (classScore[i] > classScore[best]) {
        best = i;
      }
    }

    std::shared_ptr<DecisionTreeNode> leaf = std::make_shared<DecisionTreeNode>(DecisionTreeNode::NodeType::LEAF);
    leaf->setName("LEAF " + std::to_string(best));
    leaf->setLeafValue(best);

    return leaf;
  }

  int64_t bestAttrib = -1;
  int64_t bestValue = -1;
  for (int64_t i = 0; i < ds.getTotAttributes(); i++) {
    if (availableAttrib[i]) {
      int64_t attribSize = ds.getAttributeSize(i);
      for (int64_t j = 0; j < attribSize; j++) {
        if (bestAttrib == -1 || ds.getAttributeFrequency(i, j) > ds.getAttributeFrequency(bestAttrib, bestValue)) {
          bestAttrib = i;
          bestValue = j;
        }
      }
    }
  }
  availableAttrib[bestAttrib] = false;

  int64_t bestAttribSize = ds.getAttributeSize(bestAttrib);
  std::vector<DataSet> allDS(bestAttribSize);
  for (int64_t i = 0; i < bestAttribSize; i++) {
    allDS[i].initAllAttributes(ds);
  }
  for (auto s : ds.samples_) {
    allDS[s->inxValue_[bestAttrib]].addSample(s);
  }

  std::shared_ptr<DecisionTreeNode> node = std::make_shared<DecisionTreeNode>(DecisionTreeNode::NodeType::REGULAR, bestAttrib);
  for (int64_t i = 0; i < bestAttribSize; i++) {
    if (i == bestValue) {
      node->addChild(createBackBoneRec(allDS[i], bbSize - 1, availableAttrib), { i });
    } else {
      node->addChild(createBackBoneRec(allDS[i], 1, availableAttrib), { i });
    }
  }
  return node;
}
