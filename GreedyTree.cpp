#include "GreedyTree.h"

#include "CompareUtils.h"
#include "ErrorUtils.h"
#include "Logger.h"

#include <string>

GreedyTree::GreedyTree() {}

std::shared_ptr<DecisionTreeNode> GreedyTree::createTree(DataSet& ds, std::shared_ptr<ConfigTree> c) {
  ErrorUtils::enforce(ds.getTotClasses() == 2, "Error! Number of classes must be 2.");
  std::shared_ptr<ConfigGreedy> config = std::static_pointer_cast<ConfigGreedy>(c);
  std::vector<bool> availableAttrib(ds.getTotAttributes(), true);
  return createTreeRec(ds, config->height, config->minLeaf, config->percentiles,
                       config->minGain);
}

std::shared_ptr<DecisionTreeNode> GreedyTree::createTreeRec(DataSet& ds, int64_t height, int64_t minLeaf,
                                                            int64_t percentiles, double minGain) {
  ErrorUtils::enforce(ds.getTotClasses() > 0, "Invalid data set.");

  if (height == 0 || (minLeaf > 0 && ds.samples_.size() <= minLeaf)) {
    return createLeaf(ds);
  }

  int64_t bestAttrib = -1;
  long double bestScore = 0;
  int64_t bestSeparator = -1;
  for (int64_t i = 0; i < ds.getTotAttributes(); i++) {
    auto attrib = getAttribScore(ds, i, percentiles);
    long double score = attrib.first;
    int64_t separator = attrib.second;
    if (bestAttrib == -1
        || CompareUtils::compare(score, bestScore) > 0) {
      bestAttrib = i;
      bestScore = score;
      bestSeparator = separator;
    }
  }
  if (bestAttrib == -1 || !minimumGain(ds, bestScore, minGain)) {
    return createLeaf(ds);
  }

  if (bestSeparator == -1) {
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
      node->addChild(createTreeRec(allDS[j], height - 1, minLeaf, percentiles, minGain), { j });
    }
    return node;
  } else {
    std::shared_ptr<DecisionTreeNode> node = std::make_shared<DecisionTreeNode>(DecisionTreeNode::NodeType::REGULAR_ORDERED, bestAttrib, bestSeparator);
    DataSet leftDS, rightDS;
    leftDS.initAllAttributes(ds);
    rightDS.initAllAttributes(ds);
    for (const auto& s : ds.samples_) {
      if (s->inxValue_[bestAttrib] <= bestSeparator) {
        leftDS.addSample(s);
      }
      else {
        rightDS.addSample(s);
      }
    }
    node->addLeftChild(createTreeRec(leftDS, height - 1, minLeaf, percentiles, minGain));
    node->addRightChild(createTreeRec(rightDS, height - 1, minLeaf, percentiles, minGain));
    return node;
  }
}


std::pair<long double, int64_t> GreedyTree::getAttribScore(DataSet& ds, int64_t attribInx,
                                                           int64_t percentiles) {
  if (ds.getAttributeType(attribInx) == AttributeType::STRING) {
    return std::make_pair(getNominalScore(ds, attribInx), -1);
  } else {
    return getOrderedScore(ds, attribInx, percentiles);
  }
}


long double GreedyTree::getNominalScore(DataSet& ds, int64_t attribInx) {
  long long score = 0;
  for (int64_t j = 0; j < ds.getAttributeSize(attribInx); j++) {
    auto best = ds.getSubDataSet(attribInx, j).getBestClass();
    score += best.second;
  }
  return score;
}


std::pair<long double, int64_t> GreedyTree::getOrderedScore(DataSet& ds, int64_t attribInx,
                                                            int64_t percentiles) {
  // Order attributes in asceding order according to this attribute
  struct Order {
    int64_t attribValue;
    double benefit[2];
  };
  std::vector<Order> ord(ds.samples_.size());
  int64_t count = 0;
  for (auto s : ds.samples_) {
    ord[count].attribValue = s->inxValue_[attribInx];
    ord[count].benefit[0] = s->benefit_[0];
    ord[count].benefit[1] = s->benefit_[1];
    count++;
  }
  std::sort(ord.begin(), ord.end(),
            [](const Order& a, const Order& b) { return a.attribValue < b.attribValue; });

  // Try all possible percentiles
  int64_t step = std::max(1.0, ds.getAttributeSize(attribInx) / (double)percentiles);
  long double rightScore[2] = { 0 }; // Benefit of all samples to the right
  for (auto s : ds.samples_) {
    rightScore[0] += s->benefit_[0];
    rightScore[1] += s->benefit_[1];
  }
  long double leftScore[2] = { 0 }; // Benefit of all samples to the left
  long double bestScore = std::max(rightScore[0], rightScore[1]);
  int64_t bestSeparator = ds.getAttributeSize(attribInx) - 1;
  int64_t limit = 0; // First element to the right
  while (limit < ord.size()) {
    if (CompareUtils::compare(bestScore, std::max(leftScore[0], leftScore[1])
                                         + std::max(rightScore[0], rightScore[1])) < 0) {
      bestScore = std::max(leftScore[0], leftScore[1])
                  + std::max(rightScore[0], rightScore[1]);
      bestSeparator = ord[limit].attribValue;
    }

    int64_t nextLimit = ord[limit].attribValue + step;
    while (limit < ord.size() && ord[limit].attribValue < nextLimit) {
      leftScore[0] += ord[limit].benefit[0];
      leftScore[1] += ord[limit].benefit[1];
      rightScore[0] -= ord[limit].benefit[0];
      rightScore[1] -= ord[limit].benefit[1];
      limit++;
    }
  }
  return std::make_pair(bestScore, bestSeparator-1);
}


std::shared_ptr<DecisionTreeNode> GreedyTree::createLeaf(DataSet& ds) {
  auto best = ds.getBestClass();

  std::shared_ptr<DecisionTreeNode> leaf = std::make_shared<DecisionTreeNode>(DecisionTreeNode::NodeType::LEAF);
  leaf->setName("LEAF " + std::to_string(best.first));
  leaf->setLeafValue(best.first);

  return leaf;
}


bool GreedyTree::minimumGain(DataSet& ds, long double score, long double minGain) {
  auto bestClass = ds.getBestClass();
  if (CompareUtils::compare(bestClass.second, 0) == 0) return false;

  long double gain = (score - bestClass.second) / -bestClass.second;
  return CompareUtils::compare(gain, minGain) >= 0;
}
