#include "PairTree.h"

#include "Attribute.h"
#include "BIT.h"
#include "CompareUtils.h"
#include "Logger.h"
#include "PairTreeNode.h"

#include <cmath>


std::shared_ptr<DecisionTreeNode> PairTree::createTree(DataSet& ds, std::shared_ptr<ConfigTree> c) {
  ErrorUtils::enforce(ds.getTotClasses() == 2, "Error! Number of classes must be 2.");
  
  std::shared_ptr<ConfigPairTree> config = std::static_pointer_cast<ConfigPairTree>(c);
  return createTreeRec(ds, config->height, config->maxBound, config->minLeaf,
                       config->useScore, config->useNominalBinary);
}


std::shared_ptr<DecisionTreeNode> PairTree::createTreeRec(DataSet& ds, int height,
                                                          double maxBound,
                                                          int64_t minLeaf, bool useScore,
                                                          bool useNominalBinary) {
  if (height == 0 || (minLeaf > 0 && ds.samples_.size() <= minLeaf)) {
    return createLeaf(ds);
  }

  std::vector<SampleInfo> samplesInfo;
  initSampleInfo(ds, samplesInfo);

  int bestAttrib = -1;
  double bestBound = 1;
  int64_t bestSeparator = -1;
  long double bestScore = 0;
  for (int64_t i = 0; i < ds.getTotAttributes(); i++) {
    auto attribResult = testAttribute(ds, i, samplesInfo, useNominalBinary);
    // If bound satisfy maxBound then gets either greatest score or lowest bound
    if (CompareUtils::compare(attribResult.bound, maxBound) < 0
        && ((useScore && CompareUtils::compare(attribResult.score, bestScore) > 0)
            || (!useScore && CompareUtils::compare(attribResult.bound, bestBound) < 0))) {
      bestAttrib = i;
      bestBound = attribResult.bound;
      bestSeparator = attribResult.separator;
      bestScore = attribResult.score;
    }
  }

  if (bestAttrib == -1) {
    return createLeaf(ds);
  }

  // Nominal k-valued attribute creating k children
  if (bestSeparator == -1) {
    int64_t bestAttribSize = ds.getAttributeSize(bestAttrib);
    std::vector<DataSet> allDS(bestAttribSize);
    for (int64_t i = 0; i < bestAttribSize; i++) {
      allDS[i].initAllAttributes(ds);
    }
    for (const auto& s : ds.samples_) {
      allDS[s->inxValue_[bestAttrib]].addSample(s);
    }

    std::shared_ptr<PairTreeNode> node = std::make_shared<PairTreeNode>(DecisionTreeNode::NodeType::REGULAR_NOMINAL, bestAttrib);
    node->setAlpha(bestBound);
    node->setNumSamples(ds.samples_.size());
    node->setLeafValue(ds.getBestClass().first);
    for (int64_t j = 0; j < bestAttribSize; j++) {
      node->addChild(createTreeRec(allDS[j], height - 1, maxBound, minLeaf, useScore, useNominalBinary), { j });
    }
    return node;

  // Nominal attribute separating one value from the rest
  } else if (bestSeparator != -1 && ds.getAttributeType(bestAttrib) == AttributeType::STRING) {
    std::shared_ptr<PairTreeNode> node = std::make_shared<PairTreeNode>(DecisionTreeNode::NodeType::REGULAR_NOMINAL, bestAttrib);
    node->setAlpha(bestBound);
    node->setNumSamples(ds.samples_.size());
    node->setLeafValue(ds.getBestClass().first);

    DataSet leftDS, rightDS;
    leftDS.initAllAttributes(ds);
    rightDS.initAllAttributes(ds);
    for (const auto& s : ds.samples_) {
      if (s->inxValue_[bestAttrib] == bestSeparator) {
        leftDS.addSample(s);
      } else {
        rightDS.addSample(s);
      }
    }
    
    node->addChild(createTreeRec(leftDS, height - 1, maxBound, minLeaf, useScore, useNominalBinary), { bestSeparator });
    std::vector<int64_t> rightInxs;
    for (int64_t i = 0; i < ds.getAttributeSize(bestAttrib); i++) {
      if (i != bestSeparator) rightInxs.push_back(i);
    }
    node->addChild(createTreeRec(leftDS, height - 1, maxBound, minLeaf, useScore, useNominalBinary), rightInxs);

    return node;

  // Numeric Attribute
  } else {
    std::shared_ptr<PairTreeNode> node = std::make_shared<PairTreeNode>(DecisionTreeNode::NodeType::REGULAR_ORDERED, bestAttrib, bestSeparator);
    node->setAlpha(bestBound);
    node->setNumSamples(ds.samples_.size());
    node->setLeafValue(ds.getBestClass().first);
    DataSet leftDS, rightDS;
    leftDS.initAllAttributes(ds);
    rightDS.initAllAttributes(ds);
    for (const auto& s : ds.samples_) {
      if (s->inxValue_[bestAttrib] <= bestSeparator) {
        leftDS.addSample(s);
      } else {
        rightDS.addSample(s);
      }
    }
    node->addLeftChild(createTreeRec(leftDS, height - 1, maxBound, minLeaf, useScore, useNominalBinary));
    node->addRightChild(createTreeRec(rightDS, height - 1, maxBound, minLeaf, useScore, useNominalBinary));
    return node;
  }
}


std::shared_ptr<DecisionTreeNode> PairTree::createLeaf(DataSet& ds) {
  auto best = ds.getBestClass();

  std::shared_ptr<PairTreeNode> leaf = std::make_shared<PairTreeNode>(DecisionTreeNode::NodeType::LEAF);
  leaf->setName("LEAF " + std::to_string(best.first));
  leaf->setLeafValue(best.first);

  return leaf;
}


PairTree::AttribResult PairTree::testAttribute(DataSet& ds, int64_t attribInx,
                                               std::vector<PairTree::SampleInfo>& samplesInfo,
                                               bool useNominalBinary) {
  if (ds.getAttributeType(attribInx) == AttributeType::INTEGER
      || ds.getAttributeType(attribInx) == AttributeType::DOUBLE) {
    return testNumeric(ds, attribInx, samplesInfo);
  } else {
    return testNominal(ds, attribInx, samplesInfo, useNominalBinary);
  }
}


PairTree::AttribResult PairTree::testNominal(DataSet& ds, int64_t attribInx,
                                             std::vector<PairTree::SampleInfo>& samplesInfo,
                                             bool useNominalBinary) {
  AttribResult best;
  best.bound = 1;

  int64_t attribSize = ds.getAttributeSize(attribInx);
  // If wants to use binary splits then should test from -1 to attribSize.
  // Else only needs to test boxZero = -1
  int64_t maxSplits = useNominalBinary ? attribSize : 0;
  for (int64_t i = -1; i < maxSplits; i++) {
    int64_t boxZero = i;
    std::function<int64_t(int64_t)> valueBox = [boxZero](int64_t inxValue) {
      if (boxZero < 0) {
        return inxValue;
      }
      return inxValue == boxZero ? 0ll : 1ll;
    };

    auto scoreResult = calcNominalScore(ds, attribInx, valueBox, boxZero == -1 ? attribSize : 2, samplesInfo);
    long double bound = getAttribBound(scoreResult, attribInx, samplesInfo);

    if (CompareUtils::compare(bound, best.bound) < 0) {
      best.bound = bound;
      best.score = scoreResult.score;
      best.separator = boxZero;
    }
  }

  return best;
}


PairTree::AttribScoreResult PairTree::calcNominalScore(DataSet& ds, int64_t attribInx,
                                                       std::function<int64_t(int64_t)> valueBox,
                                                       int64_t attribSize,
                                                       std::vector<PairTree::SampleInfo>& samplesInfo) {
  std::vector<double> distrib(attribSize);
  for (int64_t i = 0; i < samplesInfo.size(); i++) {
    distrib[valueBox(samplesInfo[i].ptr->inxValue_[attribInx])]++;
  }
  for (int64_t j = 0; j < attribSize; j++) {
    distrib[j] = distrib[j] / samplesInfo.size();
  }

  // totalClass[0] = number of samples whose best class is 0
  std::vector<int64_t> totalClass(2, 0);
  // totalValueClass[j][c] = number of samples valued 'j' at 'attribInx' whose class is 'c'
  std::vector<std::vector<int64_t>> totalValueClass(attribSize, std::vector<int64_t>(2, 0));

  for (auto s : samplesInfo) {
    totalClass[s.bestClass]++;
    totalValueClass[valueBox(s.ptr->inxValue_[attribInx])][s.bestClass]++;
  }
  int64_t totPairs = totalClass[0] * totalClass[1];

  long double score = 0;
  for (auto s : samplesInfo) {
    int notBestClass = (s.bestClass + 1) % 2;
    score += s.diff * (totalClass[notBestClass] - totalValueClass[valueBox(s.ptr->inxValue_[attribInx])][notBestClass]);
    totalClass[s.bestClass]--;
    totalValueClass[valueBox(s.ptr->inxValue_[attribInx])][s.bestClass]--;
  }

  AttribScoreResult ans;
  ans.score = score;
  ans.distrib = distrib;
  ans.separator = -1;
  return ans;
}


PairTree::AttribResult PairTree::testNumeric(DataSet& ds, int64_t attribInx, std::vector<PairTree::SampleInfo>& samplesInfo) {
  // Calculate the sum on formula E[Gain(A)] = 2*p(1-p) * \sum_{i=1...N}{D(s_i) * TC^i_{notC}}
  // The following variables will be used later to calculate the bound for a splitting parameter
  auto randomScore = getRandomScore(samplesInfo, std::vector<double>{ 0.5, 0.5 });
  long double randomSum = 2 * randomScore.first;
  auto auxBound = calcXstarSumsq(attribInx, samplesInfo);
  long double xstar = auxBound.first;
  long double sumSq = auxBound.second;

  long double bestBound = 1;
  long double bestScore = 0;
  int64_t bestLeftSize = 0;
  int64_t bestSeparator = 0;
  int64_t totSamples = ds.samples_.size();

  // Sort all samples by attribute attribInx
  struct Order {
    int64_t attribValue;
    int64_t posDiff;
  };
  std::vector<Order> ordSamples(totSamples);
  for (int64_t i = 0; i < samplesInfo.size(); i++) {
    Order aux;
    aux.posDiff = i;
    aux.attribValue = samplesInfo[i].ptr->inxValue_[attribInx];
    ordSamples[i] = aux;
  }
  std::sort(ordSamples.begin(), ordSamples.end(), [](const Order& a, const Order& b) { return a.attribValue < b.attribValue; });

  std::vector<std::vector<BIT>> sumLeft(2, std::vector<BIT>(2, totSamples));
  std::vector<BIT> countLeft(2, totSamples);
  std::vector<std::vector<BIT>> sumRight(2, std::vector<BIT>(2, totSamples));
  std::vector<BIT> countRight(2, totSamples);

  // Put all samples to the right
  for (int64_t i = 0; i < ordSamples.size(); i++) {
    int64_t posDiff = ordSamples[i].posDiff;
    int64_t bestClass = samplesInfo[posDiff].bestClass;
    countRight[bestClass].update(posDiff + 1, 1);
    sumRight[bestClass][0].update(posDiff + 1, samplesInfo[posDiff].ptr->benefit_[0]);
    sumRight[bestClass][1].update(posDiff + 1, samplesInfo[posDiff].ptr->benefit_[1]);
  }
  
  long double score = 0;
  int64_t i = 0;
  for (int64_t i = 0; i < totSamples; i++) {
    int64_t posDiff = ordSamples[i].posDiff;
    int bestClass = samplesInfo[posDiff].bestClass;
    int worstClass = (bestClass + 1) % 2;

    // First part max{B(x,0), B(x,1)} + max{B(y,0), B(y,1)}
    score -= (countLeft[worstClass].get(totSamples) * samplesInfo[posDiff].ptr->benefit_[bestClass]
              + sumLeft[worstClass][worstClass].get(totSamples));
    score += (countRight[worstClass].get(totSamples) * samplesInfo[posDiff].ptr->benefit_[bestClass]
              + sumRight[worstClass][worstClass].get(totSamples));

    // Second part
    // C1 = {p | p in S_notC AND D(p) < D(s) AND Ai(p) < Ai(s)}
    //ans -= countLeft[notC].get(i - 1) * B(s, c)
    //ans -= sumLeft[notC][c].get(i - 1)
    score += countLeft[worstClass].get(posDiff) * samplesInfo[posDiff].ptr->benefit_[bestClass];
    score += sumLeft[worstClass][bestClass].get(posDiff);

    // C2 = {p | p in S_notC AND D(p) > D(s) AND Ai(p) < Ai(s)}
    //ans -= (countLeft[notC].get(N) - countLeft[notC].get(i)) * B(s, notC)
    //ans -= (sumLeft[notC][notC].get(N) - sumLeft[notC][notC].get(i))
    score += (countLeft[worstClass].get(totSamples) - countLeft[worstClass].get(posDiff+1)) * samplesInfo[posDiff].ptr->benefit_[worstClass];
    score += (sumLeft[worstClass][worstClass].get(totSamples) - sumLeft[worstClass][worstClass].get(posDiff+1));

    // C3 = {p | p in S_notC AND D(p) < D(s) AND Ai(p) > Ai(s)}
    //ans += countRight[notC].get(i - 1) * B(s, c)
    //ans += sumRight[notC][c].get(i - 1)
    score -= countRight[worstClass].get(posDiff) * samplesInfo[posDiff].ptr->benefit_[bestClass];
    score -= sumRight[worstClass][bestClass].get(posDiff);

    // C4 = {p | p in S_notC AND D(p) > D(s) AND Ai(p) > Ai(s)}
    //ans += (countRight[notC].get(N) - countRight[notC].get(i)) * B(s, notC)
    //ans += (sumRight[notC][notC].get(N) - sumRight[notC][notC].get(i))
    score -= (countRight[worstClass].get(totSamples) - countRight[worstClass].get(posDiff+1)) * samplesInfo[posDiff].ptr->benefit_[worstClass];
    score -= (sumRight[worstClass][worstClass].get(totSamples) - sumRight[worstClass][worstClass].get(posDiff+1));

    sumLeft[bestClass][0].update(posDiff + 1, samplesInfo[posDiff].ptr->benefit_[0]);
    sumLeft[bestClass][1].update(posDiff + 1, samplesInfo[posDiff].ptr->benefit_[1]);
    countLeft[bestClass].update(posDiff + 1, 1);
    sumRight[bestClass][0].update(posDiff + 1, -samplesInfo[posDiff].ptr->benefit_[0]);
    sumRight[bestClass][1].update(posDiff + 1, -samplesInfo[posDiff].ptr->benefit_[1]);
    countRight[bestClass].update(posDiff + 1, -1);

    if (i == totSamples - 1 || (ordSamples[i].attribValue != ordSamples[i + 1].attribValue)) {
      long double p = (i + 1) / ((long double)totSamples);
      long double expected = 2 * p * (1 - p) * randomSum;
      long double bound = applyBound(score - expected, xstar, sumSq);
      if (CompareUtils::compare(bound, bestBound) < 0) {
        bestSeparator = ordSamples[i].attribValue;
        bestBound = bound;
        bestScore = score;
        bestLeftSize = i + 1;
      }
    }
  }

  AttribResult ans;
  ans.score = bestScore;
  ans.bound = bestBound;
  ans.separator = bestSeparator;

  return ans;
}


long double PairTree::getAttribBound(AttribScoreResult& attribResult, int64_t attribInx,
                                     std::vector<PairTree::SampleInfo>& samplesInfo) {
  auto aux = getRandomScore(samplesInfo, attribResult.distrib);
  long double expected = aux.first;
  if (CompareUtils::compare(attribResult.score, expected, 1e-7) > 0) {
    return getProbBound(attribInx, samplesInfo, attribResult.score - expected);
  }
  return 1;
}


std::pair<long double, long double> PairTree::getRandomScore(std::vector<PairTree::SampleInfo>& samplesInfo,
                                                             std::vector<double>& distrib) {
  int64_t attribSize = distrib.size();
  std::vector<int64_t> totalClass(2, 0); // totalClass[0] = number of samples whose best class is 0
  for (auto s : samplesInfo) {
    totalClass[s.bestClass]++;
  }
  int64_t totPairs = totalClass[0] * totalClass[1];
  std::vector<int64_t> totalClassCopy = totalClass;

  long double expected = 0;
  for (auto s : samplesInfo) {
    int notBestClass = (s.bestClass + 1) % 2;
    for (int j = 0; j < attribSize; j++) {
      double p = distrib[j];
      expected += s.diff * totalClass[notBestClass] * p * (1 - p);
    }
    totalClass[s.bestClass]--;
  }
  expected = expected /totPairs;
  
  totalClass = totalClassCopy;
  long double var = 0;
  for (auto s : samplesInfo) {
    int notBestClass = (s.bestClass + 1) % 2;
    for (int j = 0; j < attribSize; j++) {
      double p = distrib[j];
      var += (((long double)s.diff) - expected) * (((long double)s.diff) - expected) * totalClass[notBestClass] * p * (1 - p);
    }
    totalClass[s.bestClass]--;
  }
  var = var / totPairs;
  long double std = sqrt(var);

  return std::make_pair(expected*totPairs, std);
}


long double PairTree::getProbBound(int64_t attribInx,
                                   std::vector<PairTree::SampleInfo>& samplesInfo,
                                   long double t) {
  auto vars = calcXstarSumsq(attribInx, samplesInfo);
  return applyBound(t, vars.first, vars.second);
}


std::pair<long double, long double> PairTree::calcXstarSumsq(int64_t attribInx,
                                                             std::vector<PairTree::SampleInfo>& samplesInfo) {
  std::vector<int64_t> totalClass(2, 0); // totalClass[0] = number of samples whose best class is 0

  for (auto s : samplesInfo) {
    totalClass[s.bestClass]++;
  }
  long double xstar = std::max(totalClass[0], totalClass[1]);

  long double sumSqBounds = 0;
  for (auto s : samplesInfo) {
    int notBestClass = (s.bestClass + 1) % 2;
    sumSqBounds += (s.diff * s.diff) * totalClass[notBestClass];
    totalClass[s.bestClass]--;
  }

  return std::pair<long double, long double>(xstar, sumSqBounds);
}


long double PairTree::applyBound(long double t, long double xstar, long double sumSqBounds) {
  // Tries to avoid division by 0
  if (CompareUtils::compare(t, 0) <= 0 || CompareUtils::compare(xstar * sumSqBounds, 0) == 0) {
    return 1;
  }
  return std::exp((-2.0 * t * t) / (xstar * sumSqBounds));
}


bool compareSampleInfo(const PairTree::SampleInfo& a, const PairTree::SampleInfo& b) {
  return CompareUtils::compare(a.diff, b.diff) < 0;
}

void PairTree::initSampleInfo(DataSet& ds, std::vector<PairTree::SampleInfo>& samplesInfo) {
  samplesInfo.resize(ds.samples_.size());
  int64_t count = 0;
  for (auto s : ds.samples_) {
    SampleInfo info;
    info.ptr = s;
    if (CompareUtils::compare(s->benefit_[0], s->benefit_[1]) >= 0) {
      info.bestClass = 0;
    }
    else {
      info.bestClass = 1;
    }
    info.diff = std::abs(s->benefit_[0] - s->benefit_[1]);

    samplesInfo[count++] = info;
  }
  std::sort(samplesInfo.begin(), samplesInfo.end(), compareSampleInfo);
}
