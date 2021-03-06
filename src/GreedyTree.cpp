// Copyright (c) 2018 Daniel dos Santos Marques <danielsmarques7@gmail.com>
// License: BSD 3 clause

#include "GreedyTree.h"

#include "CompareUtils.h"
#include "ErrorUtils.h"
#include "Logger.h"
#include "ExtrasTreeNode.h"

#include <string>

GreedyTree::GreedyTree() {}

std::shared_ptr<DecisionTreeNode> GreedyTree::createTree(DataSet& ds, std::shared_ptr<ConfigTree> c) {
  ErrorUtils::enforce(ds.getTotClasses() == 2, "Error! Number of classes must be 2.");
  std::shared_ptr<ConfigGreedy> config = std::static_pointer_cast<ConfigGreedy>(c);
  std::vector<bool> availableAttrib(ds.getTotAttributes(), true);
  return createTreeRec(ds, config->height, config->minLeaf, config->percentiles,
                       config->minGain, config->useNominalBinary);
}

std::shared_ptr<DecisionTreeNode> GreedyTree::createTreeRec(DataSet& ds, int64_t height, int64_t minLeaf,
                                                            int64_t percentiles, double minGain,
                                                            bool useNominalBinary) {
  ErrorUtils::enforce(ds.getTotClasses() > 0, "Invalid data set.");

  if (height == 0 || (minLeaf > 0 && ds.samples_.size() <= minLeaf)
      || isAllSameClass(ds)) {
    return createLeaf(ds);
  }

  int64_t bestAttrib = -1;
  long double bestScore = ds.getBestClass().second;
  int64_t bestSeparator = -1;
  for (int64_t i = 0; i < ds.getTotAttributes(); i++) {
    auto attrib = getAttribScore(ds, i, percentiles, useNominalBinary);
    long double score = attrib.first;
    int64_t separator = attrib.second;
    if (CompareUtils::compare(score, bestScore) > 0) {
      bestAttrib = i;
      bestScore = score;
      bestSeparator = separator;
    }
  }
  long double bestGain = calcGain(ds, bestScore);
  if (bestAttrib == -1 || CompareUtils::compare(bestGain, minGain) < 0) {
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

    std::shared_ptr<ExtrasTreeNode> node = std::make_shared<ExtrasTreeNode>(DecisionTreeNode::NodeType::REGULAR_NOMINAL,
      bestAttrib);
    node->setAlpha(1 - bestGain);
    node->setNumSamples(ds.samples_.size());
    node->setLeafValue(ds.getBestClass().first);
    for (int64_t j = 0; j < bestAttribSize; j++) {
      node->addChild(createTreeRec(allDS[j], height - 1, minLeaf, percentiles, minGain, useNominalBinary), { j });
    }
    return node;

  // Nominal attribute separating one value from the rest
  } else if(bestSeparator != -1 && ds.getAttributeType(bestAttrib) == AttributeType::STRING) {
    std::shared_ptr<ExtrasTreeNode> node = std::make_shared<ExtrasTreeNode>(DecisionTreeNode::NodeType::REGULAR_NOMINAL, bestAttrib);
    node->setAlpha(1 - bestGain);
    node->setNumSamples(ds.samples_.size());
    node->setLeafValue(ds.getBestClass().first);

    DataSet leftDS, rightDS;
    leftDS.initAllAttributes(ds);
    rightDS.initAllAttributes(ds);
    for (const auto& s : ds.samples_) {
      if (s->inxValue_[bestAttrib] == bestSeparator) {
        leftDS.addSample(s);
      }
      else {
        rightDS.addSample(s);
      }
    }

    node->addChild(createTreeRec(leftDS, height - 1, minLeaf, percentiles, minGain, useNominalBinary), { bestSeparator });
    std::vector<int64_t> rightInxs;
    for (int64_t i = 0; i < ds.getAttributeSize(bestAttrib); i++) {
      if (i != bestSeparator) rightInxs.push_back(i);
    }
    node->addChild(createTreeRec(rightDS, height - 1, minLeaf, percentiles, minGain, useNominalBinary), rightInxs);

    return node;

  } else {
    std::shared_ptr<ExtrasTreeNode> node = std::make_shared<ExtrasTreeNode>(DecisionTreeNode::NodeType::REGULAR_ORDERED, bestAttrib, bestSeparator);
    node->setAlpha(1 - bestGain);
    node->setNumSamples(ds.samples_.size());
    node->setLeafValue(ds.getBestClass().first);
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
    node->addLeftChild(createTreeRec(leftDS, height - 1, minLeaf, percentiles, minGain, useNominalBinary));
    node->addRightChild(createTreeRec(rightDS, height - 1, minLeaf, percentiles, minGain, useNominalBinary));
    return node;
  }
}


bool GreedyTree::isAllSameClass(DataSet& ds) {
  int64_t tot0 = 0;
  int64_t tot1 = 0;
  for (auto s : ds.samples_) {
    if (CompareUtils::compare(s->benefit_[0], s->benefit_[1]) >= 0) {
      tot0++;
    }
    else {
      tot1++;
    }
    if (tot0 > 0 && tot1 > 0) return false;
  }
  return tot0 == 0 || tot1 == 0;
}


std::pair<long double, int64_t> GreedyTree::getAttribScore(DataSet& ds, int64_t attribInx,
                                                           int64_t percentiles, bool useNominalBinary) {
  if (ds.getAttributeType(attribInx) == AttributeType::STRING) {
    return getNominalScore(ds, attribInx, useNominalBinary);
  } else {
    return getOrderedScore(ds, attribInx, percentiles);
  }
}


std::pair<long double, int64_t> GreedyTree::getNominalScore(DataSet& ds, int64_t attribInx, bool useNominalBinary) {
  long double score = 0;

  if (useNominalBinary) {
    std::vector<std::vector<long double>> subClassScore(ds.getAttributeSize(attribInx),
                                                        std::vector<long double>(2, 0));
    for (int64_t j = 0; j < ds.getAttributeSize(attribInx); j++) {
      subClassScore[j][0] = ds.getSubDataSet(attribInx, j).getClassBenefit(0);
      subClassScore[j][1] = ds.getSubDataSet(attribInx, j).getClassBenefit(1);
    }

    long double sumClass0 = 0;
    long double sumClass1 = 0;
    for (int64_t j = 0; j < ds.getAttributeSize(attribInx); j++) {
      sumClass0 += subClassScore[j][0];
      sumClass1 += subClassScore[j][1];
    }
    score = 0;
    int64_t bestSeparator = -1;
    long double currScore = 0;
    for (int64_t j = 0; j < ds.getAttributeSize(attribInx); j++) {
      long double currScore = std::max(subClassScore[j][0], subClassScore[j][1])
                              + std::max(sumClass0 - subClassScore[j][0],
                                         sumClass1 - subClassScore[j][1]);
      if (bestSeparator == -1 || CompareUtils::compare(score, currScore) < 0) {
        score = currScore;
        bestSeparator = j;
      }
    }

    return std::pair<long double, int64_t>(score, bestSeparator);
  }
  else {
    for (int64_t j = 0; j < ds.getAttributeSize(attribInx); j++) {
      auto best = ds.getSubDataSet(attribInx, j).getBestClass();
      score += best.second;
    }
    return std::pair<long double, int64_t>(score, -1);
  }
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


long double GreedyTree::calcGain(DataSet& ds, long double score) {
  auto bestClass = ds.getBestClass();
  if (CompareUtils::compare(bestClass.second, 0) == 0) return 0;

  return (score - bestClass.second) / std::abs(bestClass.second);
}
