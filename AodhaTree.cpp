// Copyright (c) 2018 Daniel dos Santos Marques <danielsmarques7@gmail.com>
// License: BSD 3 clause

#include "AodhaTree.h"

#include "CompareUtils.h"
#include "ErrorUtils.h"
#include "Logger.h"
#include "ExtrasTreeNode.h"

#include <string>


AodhaTree::AodhaTree():
  minValue_(std::numeric_limits<long double>::max()),
  maxValue_(std::numeric_limits<long double>::min()) {}


std::shared_ptr<DecisionTreeNode> AodhaTree::createTree(DataSet& ds, std::shared_ptr<ConfigTree> c) {
  ErrorUtils::enforce(ds.getTotClasses() == 2, "Error! Number of classes must be 2.");

  calcNormVars(ds);

  std::shared_ptr<ConfigAodha> config = std::static_pointer_cast<ConfigAodha>(c);
  return createTreeRec(ds, config->height, config->minLeaf, config->minGain,
                       config->useNominalBinary);
}


std::shared_ptr<DecisionTreeNode> AodhaTree::createTreeRec(DataSet& ds, int64_t height,
                                                           int64_t minLeaf, long double minGain,
                                                           bool useNominalBinary) {
  ErrorUtils::enforce(ds.getTotClasses() > 0, "Invalid data set.");
  if (height == 0 || (minLeaf > 0 && ds.samples_.size() <= minLeaf)
      || isAllSameClass(ds)) {
    return createLeaf(ds);
  }

  long double impurity = calcImpurity(ds);
  int64_t bestAttrib = -1;
  long double bestGain = 0;
  int64_t bestSeparator = -1;
  for (int64_t i = 0; i < ds.getTotAttributes(); i++) {
    AttribResult result = calcAttribGain(ds, i, impurity, useNominalBinary);

    if (CompareUtils::compare(result.gain, bestGain) > 0) {
      bestAttrib = i;
      bestGain = result.gain;
      bestSeparator = result.separator;
    }
  }

  if (bestAttrib == -1 || CompareUtils::compare(bestGain, minGain) < 0) {
    return createLeaf(ds);
  }
  
  // Nominal attribute
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
      node->addChild(createTreeRec(allDS[j], height - 1, minLeaf, minGain, useNominalBinary), { j });
    }

    return node;

  // Nominal attribute separating one value from the rest
  } else if (bestSeparator != -1 && ds.getAttributeType(bestAttrib) == AttributeType::STRING) {
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

    node->addChild(createTreeRec(leftDS, height - 1, minLeaf, minGain, useNominalBinary), { bestSeparator });
    std::vector<int64_t> rightInxs;
    for (int64_t i = 0; i < ds.getAttributeSize(bestAttrib); i++) {
      if (i != bestSeparator) rightInxs.push_back(i);
    }
    node->addChild(createTreeRec(rightDS, height - 1, minLeaf, minGain, useNominalBinary), rightInxs);

    return node;

  // Numeric attribute
  } else {
    std::shared_ptr<ExtrasTreeNode> node = std::make_shared<ExtrasTreeNode>(DecisionTreeNode::NodeType::REGULAR_ORDERED,
                                                                            bestAttrib, bestSeparator);
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
    node->addLeftChild(createTreeRec(leftDS, height - 1, minLeaf, minGain, useNominalBinary));
    node->addRightChild(createTreeRec(rightDS, height - 1, minLeaf, minGain, useNominalBinary));
    return node;
  }
}


bool AodhaTree::isAllSameClass(DataSet& ds) {
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


AodhaTree::AttribResult AodhaTree::calcAttribGain(DataSet& ds, int64_t attribInx,
                                                  long double parentImp,
                                                  bool useNominalBinary) {
  if (ds.getAttributeType(attribInx) == AttributeType::STRING) {
    return calcNominalGain(ds, attribInx, parentImp, useNominalBinary);
  } else {
    return calcNumericGain(ds, attribInx, parentImp);
  }
}


AodhaTree::AttribResult AodhaTree::calcNominalGain(DataSet& ds, int64_t attribInx,
                                                   long double parentImp, bool useNominalBinary) {
  AttribResult ans;

  if (useNominalBinary) {
    ImpSums allSums;
    std::vector<ImpSums> subSums(ds.getAttributeSize(attribInx));
    std::vector<int64_t> totSamples(ds.getAttributeSize(attribInx), 0);

    for (auto s : ds.samples_) {
      totSamples[s->inxValue_[attribInx]]++;

      long double benefit0 = normalizedValue(s->benefit_[0]);
      long double benefit1 = normalizedValue(s->benefit_[1]);

      subSums[s->inxValue_[attribInx]].sumS += std::abs(benefit0 - benefit1);
      allSums.sumS += std::abs(benefit0 - benefit1);

      if (CompareUtils::compare(benefit0, benefit1) > 0) {
        subSums[s->inxValue_[attribInx]].sumS0 += benefit0 - benefit1;
        subSums[s->inxValue_[attribInx]].sumSqS0 += (benefit0 - benefit1) * (benefit0 - benefit1);
        allSums.sumS0 += benefit0 - benefit1;
        allSums.sumSqS0 += (benefit0 - benefit1) * (benefit0 - benefit1);
      }
      else {
        subSums[s->inxValue_[attribInx]].sumS1 += benefit1 - benefit0;
        subSums[s->inxValue_[attribInx]].sumSqS1 += (benefit1 - benefit0) * (benefit1 - benefit0);
        allSums.sumS1 += benefit1 - benefit0;
        allSums.sumSqS1 += (benefit1 - benefit0) * (benefit1 - benefit0);
      }
    }

    ans.impurity = parentImp;
    ans.gain = 0;
    ans.separator = -1;
    for (int64_t j = 0; j < ds.getAttributeSize(attribInx); j++) {
      long double impLeft = applyFormula(subSums[j].sumS,
                                         subSums[j].sumS0,
                                         subSums[j].sumS1,
                                         subSums[j].sumSqS0,
                                         subSums[j].sumSqS1);
      long double impRight = applyFormula(allSums.sumS - subSums[j].sumS,
                                          allSums.sumS0 - subSums[j].sumS0,
                                          allSums.sumS1 - subSums[j].sumS1,
                                          allSums.sumSqS0 - subSums[j].sumSqS0,
                                          allSums.sumSqS1 - subSums[j].sumSqS1);
      long double impurity = impLeft * (totSamples[j] / ((long double)ds.samples_.size()))
                             + impRight * ((ds.samples_.size() - totSamples[j]) / ((long double)ds.samples_.size()));
      if (ans.separator == -1 || CompareUtils::compare(ans.impurity, impurity) > 0) {
        ans.impurity = impurity;
        ans.gain = parentImp - impurity;
        ans.separator = j;
      }
    }
  }
  else {
    long double impurity = 0;
    if (ds.samples_.size() > 0) {
      for (int i = 0; i < ds.getAttributeSize(attribInx); i++) {
        auto subDS = ds.getSubDataSet(attribInx, i);
        impurity += (subDS.samples_.size() / ((long double)ds.samples_.size())) * calcImpurity(subDS);
      }
    }
    ans.impurity = impurity;
    ans.gain = parentImp - impurity;
    ans.separator = -1;
  }
  return ans;
}


AodhaTree::AttribResult AodhaTree::calcNumericGain(DataSet& ds, int64_t attribInx, long double parentImp) {
  // Order attributes in asceding order according to this attribute
  struct Order {
    int64_t attribValue;
    std::shared_ptr<Sample> ptr;
  };
  std::vector<Order> ord(ds.samples_.size());
  int64_t count = 0;
  for (auto s : ds.samples_) {
    ord[count].attribValue = s->inxValue_[attribInx];
    ord[count].ptr = s;
    count++;
  }
  std::sort(ord.begin(), ord.end(),
    [](const Order& a, const Order& b) { return a.attribValue < b.attribValue; });

  // Calculate the sums for left and right children
  ImpSums leftSums;
  ImpSums rightSums;
  for (auto s : ds.samples_) {
    long double benefit0 = normalizedValue(s->benefit_[0]);
    long double benefit1 = normalizedValue(s->benefit_[1]);

    rightSums.sumS += std::abs(benefit0 - benefit1);

    if (CompareUtils::compare(benefit0, benefit1) > 0) {
      rightSums.sumS0 += benefit0 - benefit1;
      rightSums.sumSqS0 += (benefit0 - benefit1) * (benefit0 - benefit1);
    }
    else {
      rightSums.sumS1 += benefit1 - benefit0;
      rightSums.sumSqS1 += (benefit1 - benefit0) * (benefit1 - benefit0);
    }
  }

  // Tries all possible splitting parameters
  long double bestImpurity = parentImp;
  int64_t bestSeparator = -1;
  int64_t i = 0;
  while (i < ord.size()) {
    long double leftImp = applyFormula(leftSums.sumS, leftSums.sumS0,
                                       leftSums.sumS1, leftSums.sumSqS0,
                                       leftSums.sumSqS1);
    long double rightImp = applyFormula(rightSums.sumS, rightSums.sumS0,
                                        rightSums.sumS1, rightSums.sumSqS0,
                                        rightSums.sumSqS1);
    long double impurity = (i/((long double)ord.size())) * leftImp
                           + ((ord.size() - i)/((long double)ord.size())) * rightImp;
    if (CompareUtils::compare(impurity, bestImpurity) < 0) {
      bestImpurity = impurity;
      bestSeparator = ord[i-1].attribValue;
    }
    int64_t start = i;
    while (i < ord.size() && ord[i].attribValue == ord[start].attribValue) {
      // Decrease rightSums and add leftSums
      long double benefit0 = normalizedValue(ord[i].ptr->benefit_[0]);
      long double benefit1 = normalizedValue(ord[i].ptr->benefit_[1]);

      rightSums.sumS -= std::abs(benefit0 - benefit1);
      leftSums.sumS += std::abs(benefit0 - benefit1);
      if (CompareUtils::compare(benefit0, benefit1) > 0) {
        rightSums.sumS0 -= benefit0 - benefit1;
        rightSums.sumSqS0 -= (benefit0 - benefit1) * (benefit0 - benefit1);
        leftSums.sumS0 += benefit0 - benefit1;
        leftSums.sumSqS0 += (benefit0 - benefit1) * (benefit0 - benefit1);
      } else {
        rightSums.sumS1 -= benefit1 - benefit0;
        rightSums.sumSqS1 -= (benefit1 - benefit0) * (benefit1 - benefit0);
        leftSums.sumS1 += benefit1 - benefit0;
        leftSums.sumSqS1 += (benefit1 - benefit0) * (benefit1 - benefit0);
      }
      i++;
    }
  }

  AttribResult ans;
  ans.impurity = bestImpurity;
  ans.gain = parentImp - bestImpurity;
  ans.separator = bestSeparator;
  return ans;
}


std::shared_ptr<DecisionTreeNode> AodhaTree::createLeaf(DataSet& ds) {
  auto best = ds.getBestClass();

  std::shared_ptr<DecisionTreeNode> leaf = std::make_shared<DecisionTreeNode>(DecisionTreeNode::NodeType::LEAF);
  leaf->setName("LEAF " + std::to_string(best.first));
  leaf->setLeafValue(best.first);

  return leaf;
}


long double AodhaTree::calcImpurity(DataSet& ds) {
  if (ds.samples_.size() == 0) {
    return 0;
  }

  long double sumS = 0;
  long double sumS0 = 0;
  long double sumS1 = 0;
  long double sumSqS0 = 0;
  long double sumSqS1 = 0;

  for (auto s : ds.samples_) {
    long double benefit0 = normalizedValue(s->benefit_[0]);
    long double benefit1 = normalizedValue(s->benefit_[1]);

    sumS += std::abs(benefit0 - benefit1);

    if (CompareUtils::compare(benefit0, benefit1) > 0) {
      sumS0 += benefit0 - benefit1;
      sumSqS0 += (benefit0 - benefit1) * (benefit0 - benefit1);
    }
    else {
      sumS1 += benefit1 - benefit0;
      sumSqS1 += (benefit1 - benefit0) * (benefit1 - benefit0);
    }
  }

  return applyFormula(sumS, sumS0, sumS1, sumSqS0, sumSqS1);
}


long double AodhaTree::applyFormula(long double sumS, long double sumS0,
                                    long double sumS1, long double sumSqS0,
                                    long double sumSqS1) {
  if (CompareUtils::compare(sumS, 0) == 0) return 0;
  return 0.5*((sumSqS0 + sumSqS1) / sumS - (sumSqS0 * sumSqS0 + sumSqS1 * sumSqS1) / (sumS * sumS));
}


void AodhaTree::calcNormVars(DataSet& ds) {
  minValue_ = std::numeric_limits<long double>::max();
  maxValue_ = std::numeric_limits<long double>::min();
  for (auto s : ds.samples_) {
    minValue_ = std::min(minValue_, (long double)std::min(s->benefit_[0], s->benefit_[1]));
    maxValue_ = std::max(maxValue_, (long double)std::max(s->benefit_[0], s->benefit_[1]));
  }
}


long double AodhaTree::normalizedValue(long double value) {
  return (value - minValue_) / (maxValue_ - minValue_);
}
