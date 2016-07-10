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
  return createTreeRec(ds, config->height);
}


std::shared_ptr<DecisionTreeNode> AodhaTree::createTreeRec(DataSet& ds, int64_t height) {
  ErrorUtils::enforce(ds.getTotClasses() > 0, "Invalid data set.");

  if (height == 0) {
    return createLeaf(ds);
  }

  long double impurity = calcImpurity(ds);
  int64_t bestAttrib = -1;
  long double bestGain = 0;
  int64_t bestSeparator = -1;
  for (int64_t i = 0; i < ds.getTotAttributes(); i++) {
    AttribResult result = calcAttribGain(ds, i, impurity);

    if (CompareUtils::compare(result.gain, bestGain) > 0) {
      bestAttrib = i;
      bestGain = result.gain;
      bestSeparator = result.separator;
    }
  }

  if (bestAttrib == -1) {
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

    std::shared_ptr<DecisionTreeNode> node = std::make_shared<DecisionTreeNode>(DecisionTreeNode::NodeType::REGULAR_NOMINAL,
                                                                                bestAttrib);
    for (int64_t j = 0; j < bestAttribSize; j++) {
      node->addChild(createTreeRec(allDS[j], height - 1), { j });
    }

    return node;
  }
  // Numeric attribute
  else {
    std::shared_ptr<DecisionTreeNode> node = std::make_shared<DecisionTreeNode>(DecisionTreeNode::NodeType::REGULAR_ORDERED,
                                                                                bestAttrib, bestSeparator);
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
    node->addLeftChild(createTreeRec(leftDS, height - 1));
    node->addRightChild(createTreeRec(rightDS, height - 1));
    return node;
  }
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
    } else {
      sumS1 += benefit1 - benefit0;
      sumSqS1 += (benefit1 - benefit0) * (benefit1 - benefit0);
    }
  }

  return 0.5*((sumSqS0 + sumSqS1)/sumS - (sumSqS0 * sumSqS0 + sumSqS1 * sumSqS1)/(sumS * sumS));
}


AodhaTree::AttribResult AodhaTree::calcAttribGain(DataSet& ds, int64_t attribInx,
                                                  long double parentImp) {
  if (ds.getAttributeType(attribInx) == AttributeType::STRING) {
    return calcNominalGain(ds, attribInx, parentImp);
  } else {
    return calcNumericGain(ds, attribInx, parentImp);
  }
}


AodhaTree::AttribResult AodhaTree::calcNominalGain(DataSet& ds, int64_t attribInx, long double parentImp) {
  long double impurity = 0;
  for (int i = 0; i < ds.getAttributeSize(attribInx); i++) {
    impurity += calcImpurity(ds.getSubDataSet(attribInx, i));
  }

  AttribResult ans;
  ans.impurity = impurity;
  ans.gain = parentImp - impurity;
  ans.separator = -1;
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

  DataSet leftDS, rightDS;
  leftDS.initAllAttributes(ds);
  rightDS.initAllAttributes(ds);
  for (auto s : ds.samples_) {
    rightDS.addSample(s);
  }

  long double bestImpurity = parentImp;
  int64_t bestSeparator = -1;
  int64_t separator = 0;
  while (!rightDS.samples_.empty()) {
    long double leftImp = calcImpurity(leftDS);
    long double rightImp = calcImpurity(rightDS);
    long double impurity = leftImp + rightImp;
    if (CompareUtils::compare(impurity, bestImpurity) < 0) {
      bestImpurity = impurity;
      bestSeparator = separator;
    }

    int64_t curValue = rightDS.samples_.front()->inxValue_[attribInx];
    while (!rightDS.samples_.empty()
           && rightDS.samples_.front()->inxValue_[attribInx] == curValue) {
      leftDS.addSample(rightDS.samples_.front());
      rightDS.samples_.pop_front();
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
